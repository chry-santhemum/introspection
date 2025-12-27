# %%
import os
os.environ["HF_HOME"] = "/root/hf"

from dataclasses import dataclass
from typing import Callable, Literal, Optional
from jaxtyping import Float
from pathlib import Path

import torch
from torch import Tensor
import matplotlib.pyplot as plt

from utils_model import (
    load_model,
    load_tokenizer,
    get_resid_block_name,
    get_num_layers,
    get_all_component_names,
    ComponentType,
    FwdHook,
    fwd_with_hooks,
    SteerConfig,
    scale_vec,
    Activations,
    fwd_steer,
    fwd_record_all,
    TokenSpec,
)
from concept_vectors import introspection_inputs


# --- Attribution computation ---

# Metric function: takes final hidden state [hidden_dim] and returns a scalar tensor
Metric = Callable[[Tensor], Tensor]


def logit_metric(model, token_id: int) -> Metric:
    """Create a metric that returns the logit for a specific token.

    Applies the model's final norm and lm_head to get logits.
    """
    def metric(hidden: Tensor) -> Tensor:
        logits = model.lm_head(hidden)
        return logits[token_id]
    return metric


def logit_diff_metric(model, pos_token_id: int, neg_token_id: int) -> Metric:
    """Create a metric for the difference between two token logits."""
    def metric(hidden: Tensor) -> Tensor:
        logits = model.lm_head(hidden)
        return logits[pos_token_id] - logits[neg_token_id]
    return metric


def projection_metric(direction: Tensor) -> Metric:
    """Create a metric that returns the projection onto a direction vector."""
    def metric(hidden: Tensor) -> Tensor:
        # Normalize direction and compute dot product
        direction_normalized = direction / direction.norm()
        return (hidden * direction_normalized.to(hidden.device)).sum()
    return metric


@dataclass
class AttributionResult:
    resid: Float[Tensor, "num_layers seq_len"]
    attn: Float[Tensor, "num_layers seq_len"]
    mlp: Float[Tensor, "num_layers seq_len"]

    # Metadata
    tokens: list[str]
    metric_value: float

    def aggregate(self, method: Literal["sum", "mean", "abs_sum", "abs_mean"] = "sum") -> dict[ComponentType, Tensor]:
        result = {}
        for comp in ["resid", "attn", "mlp"]:
            attr = getattr(self, comp)  # [num_layers, seq_len]
            match method:
                case "sum":
                    result[comp] = attr.sum(dim=1)
                case "mean":
                    result[comp] = attr.mean(dim=1)
                case "abs_sum":
                    result[comp] = attr.abs().sum(dim=1)
                case "abs_mean":
                    result[comp] = attr.abs().mean(dim=1)
        return result


def gradient_attribution(
    model,
    tokenizer,
    inputs: dict,
    steer_config: Optional[SteerConfig],
    metric: Metric,
    source: Activations,
    components: list[ComponentType] = ["resid", "attn", "mlp"],
    attr_start_pos: int = 0,
) -> AttributionResult:
    """Gradient-based attribution for activation patching approximation.

    Computes: attribution = d(metric)/d(component_output) · (activation - source)
    """
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    num_layers = get_num_layers(model)
    module_names = get_all_component_names(model, components)

    full_seq_len = inputs["input_ids"].shape[1]
    attr_seq_len = full_seq_len - attr_start_pos

    # Create record hooks for all components
    record_hooks = {
        key: FwdHook(
            module_name=module_name,
            pos="output",
            op="record",
            tokens=slice(None),  # Record all positions (need full for gradient flow)
        )
        for key, module_name in module_names.items()
    }

    # Build list of all hooks (record + optional steering)
    all_hooks = list(record_hooks.values())
    if steer_config:
        steer_hook = FwdHook(
            module_name=get_resid_block_name(model, steer_config.layer),
            pos="output",
            op="add",
            tokens=steer_config.tokens,
            tensor=scale_vec(steer_config.vec, steer_config.strength),
        )
        all_hooks.append(steer_hook)

    # We need gradients, so we make the input embeddings require grad
    # This ensures the computation graph is built
    input_ids = inputs["input_ids"]
    inputs_embeds = model.get_input_embeddings()(input_ids)
    inputs_embeds = inputs_embeds.detach().requires_grad_(True)

    # Prepare inputs without input_ids (use inputs_embeds instead)
    fwd_inputs = {k: v for k, v in inputs.items() if k != "input_ids"}
    fwd_inputs["inputs_embeds"] = inputs_embeds

    # Forward pass with hooks (allow_grad=True for gradient computation)
    with fwd_with_hooks(all_hooks, model, allow_grad=True):
        outputs = model(**fwd_inputs, output_hidden_states=True)

        # # Print top 10 most probable tokens
        # logits = outputs.logits[0, -1, :]  # [vocab]
        # probs = torch.softmax(logits.float(), dim=-1)
        # top_probs, top_indices = torch.topk(probs, k=10)
        # print("Top 10 tokens:")
        # for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        #     token = tokenizer.decode([idx])
        #     print(f"  {i+1:>3}. {repr(token):20s} prob={prob.item():<10.6f} logit={logits[idx].item():<8.2f}")

        # Compute metric and backward
        metric_value = metric(outputs.hidden_states[-1][0, -1, :])
        metric_value.backward()

    # Compute attribution: grad · (activation - source)
    # Only for positions from attr_start_pos onwards
    attribution = {comp: torch.zeros(num_layers, attr_seq_len) for comp in components}

    for key, hook in record_hooks.items():
        comp, layer = key
        act = hook.tensor  # [1, full_seq, hidden]
        grad = hook.grad  # [1, full_seq, hidden] - captured by backward hook

        if grad is None:
            print(f"Warning: No gradient for {comp} layer {layer}")
            continue

        # Attribution at each position from attr_start_pos onwards
        for i, pos in enumerate(range(attr_start_pos, full_seq_len)):
            act_pos = act[0, pos, :]  # [hidden]
            grad_pos = grad[0, pos, :]  # [hidden]
            base = source.get_at(comp, layer, pos).to(device)  # [hidden]

            diff = act_pos - base
            attr = (grad_pos * diff).sum().item()
            attribution[comp][layer, i] = attr

    # Get token strings for metadata (only from attr_start_pos onwards)
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0, attr_start_pos:].tolist()]

    return AttributionResult(
        resid=attribution.get("resid", torch.zeros(num_layers, attr_seq_len)),
        attn=attribution.get("attn", torch.zeros(num_layers, attr_seq_len)),
        mlp=attribution.get("mlp", torch.zeros(num_layers, attr_seq_len)),
        tokens=tokens,
        metric_value=metric_value.item(),
    )


# %% --- Visualization helpers ---

COMPONENT_NAMES = {"resid": "Residual Stream", "attn": "Attention", "mlp": "MLP"}


def save_attribution_heatmaps(
    result: AttributionResult,
    save_dir: str,
    prefix: str = "",
    start_layer: int = 0,
    title: str | None = None,
    cell_size: float = 0.15,
):
    """Save attribution heatmaps for each component (resid, attn, mlp)."""
    os.makedirs(save_dir, exist_ok=True)
    if title is None:
        title = f"Attribution (metric: {result.metric_value:.2f})"

    for comp in ["resid", "attn", "mlp"]:
        attr = getattr(result, comp)[start_layer:]
        num_layers, seq_len = attr.shape

        fig, ax = plt.subplots(figsize=(seq_len * cell_size + 2, num_layers * cell_size + 1.5))
        vmax = max(abs(attr.min().item()), abs(attr.max().item()))
        im = ax.imshow(attr.numpy(), aspect="equal", cmap="RdBu_r", vmin=-vmax, vmax=vmax)

        ax.set_xlabel("Token")
        ax.set_ylabel("Layer")
        ax.set_title(f"{COMPONENT_NAMES[comp]} {title}")
        ax.set_xticks(range(len(result.tokens)))
        ax.set_xticklabels([repr(t)[1:-1] for t in result.tokens], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(num_layers))
        ax.set_yticklabels([str(i + start_layer) for i in range(num_layers)])
        plt.colorbar(im, ax=ax, label="Attribution")
        plt.tight_layout()

        filepath = os.path.join(save_dir, f"{prefix}_{comp}.png" if prefix else f"{comp}.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {filepath}")


def save_layer_attribution(
    attrs: AttributionResult | dict[ComponentType, Tensor],
    save_dir: str,
    prefix: str = "",
    start_layer: int = 0,
    title: str | None = None,
):
    """Plot and save layer attribution with separate vertical subplots per component."""
    if isinstance(attrs, AttributionResult):
        data = attrs.aggregate(method="sum")
    else:
        data = attrs

    components: list[ComponentType] = ["resid", "attn", "mlp"]
    colors = {"resid": "#1f77b4", "attn": "#ff7f0e", "mlp": "#2ca02c"}
    num_layers = data[components[0]].shape[0]
    layers = range(start_layer, num_layers)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, comp in zip(axes, components):
        ax.plot(layers, data[comp][start_layer:].numpy(), color=colors[comp], linewidth=1.5)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.set_ylabel(COMPONENT_NAMES[comp])
    axes[-1].set_xlabel("Layer")
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    filename = f"{prefix}_layer_attribution.png" if prefix else "layer_attribution.png"
    fig.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {os.path.join(save_dir, filename)}")


# %% --- Experiment script ---

# Load model and tokenizer
model = load_model(
    model_name="google/gemma-3-27b-it",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)
for p in model.parameters():
    p.requires_grad_(True)

tokenizer = load_tokenizer(model_name="google/gemma-3-27b-it")

# wikitext = load_dataset("Salesforce/wikitext", name="wikitext-2-v1", split="train")
# ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

# def preprocess_item(item: dict):
#     formatted_text = tokenizer.apply_chat_template(item["messages"], tokenize=False)
#     return {"text": formatted_text}

# ultrachat_to_use = ultrachat.select(range(5000)).map(preprocess_item, num_proc=8)

# mean_acts = compute_mean_activations(
#     model, 
#     tokenizer,
#     ultrachat_to_use,
# )

# print(f"Mean activations shape: resid={mean_acts.resid.shape}, attn={mean_acts.attn.shape}, mlp={mean_acts.mlp.shape}")

# # Save mean activations for reuse
# torch.save(mean_acts, "attribution/mean_acts-27b-it.pt")

# %%
# Load concept vectors
LAYER = 38

base_path = Path(f"concept_vectors/concept_diff-27b-it-L{LAYER}")
concept_vectors = torch.load(base_path / "concepts.pt", weights_only=True)
refusal_vectors = torch.load("attribution/refusal_directions.pt")

inputs = introspection_inputs(
    tokenizer, 
    append=None,
    prefill=None
).to("cuda:0")

# Find double newline position for steering
decoded_tokens = [tokenizer.decode(x) for x in inputs["input_ids"][0]]
for i, tok in enumerate(decoded_tokens):
    if tok == "\n\n":
        double_newline_pos = i
        break

# %%
# Harmonies and Sugar are relatively weak
AFFIRM_WORDS = ["Algorithms", "Aquariums", "Bread", "Origami", "Satellites", "Trees", "Vegetables", "Volcanoes"]

# # Accumulate layer-wise attributions for averaging
# num_layers = get_num_layers(model)
# COMPONENTS: list[ComponentType] = ["resid", "attn", "mlp"]
# ctrl_to_steer_accum: dict[ComponentType, Tensor] = {c: torch.zeros(num_layers) for c in COMPONENTS}
# steer_to_ctrl_accum: dict[ComponentType, Tensor] = {c: torch.zeros(num_layers) for c in COMPONENTS}

success_word = "Bread"
failure_word = "Mirrors"

no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]

success_steer_config = SteerConfig(
    layer=LAYER,
    tokens=slice(double_newline_pos, None),
    vec=concept_vectors[success_word],
    strength=4.0,
)

steered_logits = fwd_steer(
    steer_config=success_steer_config,
    model=model,
    inputs=inputs
)
logits = steered_logits[0]
top_token_id = int(torch.argmax(logits).item())
top_token = tokenizer.decode([top_token_id])
print(f"Top token: {top_token}")

metric=logit_diff_metric(model, top_token_id, no_token_id)

# %%
# success to failure
source_acts = fwd_record_all(
    model=model,
    inputs=inputs,
    steer_config=success_steer_config,
    components=["resid", "attn", "mlp"],
)

failure_steer_config = SteerConfig(
    layer=LAYER,
    tokens=slice(double_newline_pos, None),
    vec=concept_vectors[failure_word],
    strength=4.0,
)

result = gradient_attribution(
    model=model,
    tokenizer=tokenizer,
    inputs=inputs,
    steer_config=failure_steer_config,
    metric=metric,
    source=source_acts,
    attr_start_pos=double_newline_pos,
)

# # Accumulate layer-wise attribution (sum over tokens)
# agg = result.aggregate(method="sum")
# for comp in agg:
#     ctrl_to_steer_accum[comp] += agg[comp]

save_attribution_heatmaps(result, "attribution/success_to_failure", f"{success_word}_to_{failure_word}", LAYER, f"(logit({top_token}) - logit(No) = {result.metric_value:.2f})")
save_layer_attribution(result, "attribution/success_to_failure", f"{success_word}_to_{failure_word}", LAYER, f"Layer Attribution (logit({top_token}) - logit(No) = {result.metric_value:.2f})")


# %%
# steer to ctrl
source_acts = fwd_record_all(
    model=model,
    inputs=inputs,
    steer_config=steer_config,
    components=["resid", "attn", "mlp"],
)

result = gradient_attribution(
    model=model,
    tokenizer=tokenizer,
    inputs=inputs,
    steer_config=None,
    metric=metric,
    source=source_acts,
    attr_start_pos=double_newline_pos,
)

# Accumulate layer-wise attribution (sum over tokens)
agg = result.aggregate(method="sum")
for comp in agg:
    steer_to_ctrl_accum[comp] += agg[comp]

save_attribution_heatmaps(result, "attribution/steer_to_ctrl", word, LAYER, f"(logit({top_token}) - logit(No) = {result.metric_value:.2f})")
save_layer_attribution(result, "attribution/steer_to_ctrl", word, LAYER, f"Layer Attribution (logit({top_token}) - logit(No) = {result.metric_value:.2f})")

# # Average over all words and save
# n_words = len(AFFIRM_WORDS)
# ctrl_to_steer_avg: dict[ComponentType, Tensor] = {c: ctrl_to_steer_accum[c] / n_words for c in COMPONENTS}
# steer_to_ctrl_avg: dict[ComponentType, Tensor] = {c: steer_to_ctrl_accum[c] / n_words for c in COMPONENTS}

# save_layer_attribution(ctrl_to_steer_avg, "attribution/ctrl_to_steer", "averaged", LAYER, f"Averaged Layer Attribution (ctrl → steer, n={n_words})")
# save_layer_attribution(steer_to_ctrl_avg, "attribution/steer_to_ctrl", "averaged", LAYER, f"Averaged Layer Attribution (steer → ctrl, n={n_words})")

# %%

DEFAULT_REGION_WEIGHTS = {
    "very_early": 0.02422,  # Layers 0-10
    "early": 0.00858,       # Layers 11-20
    "pre_key": 0.00640,     # Layers 21-28
    "key": 0.01072,         # Layers 29-35
    "mid": 0.05206,         # Layers 36-47
    "late": 0.93913,        # Layers 48-55
    "final": 0.46057,       # Layers 56-61
}

DEFAULT_REGION_BOUNDARIES = {
    "very_early_end": 10,   # very_early: 0-10 (inclusive)
    "early_end": 20,        # early: 11-20
    "pre_key_end": 28,      # pre_key: 21-28
    "key_end": 35,          # key: 29-35 (BYPASS ZONE)
    "mid_end": 47,          # mid: 36-47
    "late_end": 55,         # late: 48-55
    # final: 56-61
}


def refusal_ablation_hooks(
    model,
    refusal_vectors: Tensor,
    tokens: TokenSpec,
) -> list[FwdHook]:
    """Create proj_ablate hooks for each layer, scaled by region weights."""
    num_layers = refusal_vectors.shape[0]
    hooks = []

    def get_layer_region(layer: int) -> str:
        """Get the region name for a given layer."""
        boundaries = DEFAULT_REGION_BOUNDARIES
        if layer <= boundaries["very_early_end"]:
            return "very_early"
        elif layer <= boundaries["early_end"]:
            return "early"
        elif layer <= boundaries["pre_key_end"]:
            return "pre_key"
        elif layer <= boundaries["key_end"]:
            return "key"
        elif layer <= boundaries["mid_end"]:
            return "mid"
        elif layer <= boundaries["late_end"]:
            return "late"
        else:
            return "final"

    for layer in range(num_layers):
        region = get_layer_region(layer)
        weight = DEFAULT_REGION_WEIGHTS[region]

        # Scale the refusal vector by the region weight
        scaled_vec = refusal_vectors[layer] * weight

        hooks.append(FwdHook(
            module_name=get_resid_block_name(model, layer),
            pos="output",
            op="proj_ablate",
            tokens=tokens,
            tensor=scaled_vec,
        ))

    return hooks

# %%
word = "Mirrors"

steer_hook = FwdHook(
    module_name=get_resid_block_name(model, LAYER),
    pos="output",
    op="add",
    tokens=slice(double_newline_pos, None),
    tensor=scale_vec(concept_vectors[word], 4.0),
)

source_acts = fwd_record_all(...)
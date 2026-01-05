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
    print_top_tokens,
    ComponentType,
    FwdHook,
    fwd_with_hooks,
    SteerConfig,
    Activations,
    fwd_record_all,
    TokenSpec,
)
from utils_introspection import introspection_template


# --- Attribution computation ---

# Metric function: takes final hidden state [hidden_dim] and returns a scalar tensor
Metric = Callable[[Tensor], Tensor]


def logit_metric(model, token_id: int) -> Metric:
    def metric(hidden: Tensor) -> Tensor:
        logits = model.lm_head(hidden)
        return logits[token_id]
    return metric


def logit_diff_metric(model, pos_token_id: int, neg_token_id: int) -> Metric:
    def metric(hidden: Tensor) -> Tensor:
        logits = model.lm_head(hidden)
        return logits[pos_token_id] - logits[neg_token_id]
    return metric


def projection_metric(direction: Tensor) -> Metric:
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
    inputs: dict[str, Tensor],
    hooks: list[FwdHook],
    metric: Metric,
    source: Activations,
    components: list[ComponentType] = ["resid", "attn", "mlp"],
    attribution_start_pos: int = 0,
) -> AttributionResult:
    """Gradient-based attribution for activation patching approximation.

    Computes: attribution = d(metric)/d(component_output) · (activation - source)
    """
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    num_layers = get_num_layers(model)
    module_names = get_all_component_names(model, components)

    full_seq_len = inputs["input_ids"].shape[1]
    attr_seq_len = full_seq_len - attribution_start_pos

    # Create record hooks for all components
    record_hooks = {
        key: FwdHook(
            module_name=module_name,
            pos="output",
            op="record",
            tokens=slice(None),
        )
        for key, module_name in module_names.items()
    }
    all_hooks = list(record_hooks.values())
    all_hooks.extend(hooks)

    # make the input embeddings require grad
    # in order to ensure gradients are propagated
    input_ids = inputs["input_ids"]
    inputs_embeds = model.get_input_embeddings()(input_ids)
    inputs_embeds = inputs_embeds.detach().requires_grad_(True)

    # use inputs_embeds instead of input_ids
    fwd_inputs = {k: v for k, v in inputs.items() if k != "input_ids"}
    fwd_inputs["inputs_embeds"] = inputs_embeds

    # Forward pass with hooks (allow_grad=True for gradient computation)
    with fwd_with_hooks(all_hooks, model, allow_grad=True):
        outputs = model(**fwd_inputs, output_hidden_states=True)

        # Compute metric and backward
        metric_value = metric(outputs.hidden_states[-1][0, -1, :])
        metric_value.backward()

    # Compute attribution: grad · (activation - source)
    # Only for positions from attribution_start_pos onwards
    attribution = {comp: torch.zeros(num_layers, attr_seq_len) for comp in components}

    for key, hook in record_hooks.items():
        comp, layer = key
        act = hook.tensor  # [1, full_seq, hidden]
        grad = hook.grad  # [1, full_seq, hidden]

        if grad is None:
            print(f"Warning: No gradient for {comp} layer {layer}")
            continue

        # Attribution at each position from attribution_start_pos onwards
        for i, pos in enumerate(range(attribution_start_pos, full_seq_len)):
            act_pos = act[0, pos, :]  # [hidden]
            grad_pos = grad[0, pos, :]  # [hidden]
            base = source.get_at(comp, layer, pos).to(device)  # [hidden]

            diff = act_pos - base
            attr = (grad_pos * diff).sum().item()
            attribution[comp][layer, i] = attr

    # Get token strings for metadata (only from attribution_start_pos onwards)
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0, attribution_start_pos:].tolist()]

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
base_model = load_model(
    model_name="google/gemma-3-27b-pt",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)
for p in base_model.parameters():
    p.requires_grad_(True)

model = load_model(
    model_name="google/gemma-3-27b-it",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

tokenizer = load_tokenizer(model_name="google/gemma-3-27b-it")

# %%
# Load concept vectors
LAYER = 38

base_path = Path(f"concept_vectors/concept_diff-27b-it-L{LAYER}")
concept_vectors = torch.load(base_path / "concepts.pt", weights_only=True)

input_str = introspection_template(tokenizer, append=None, prefill=None)
inputs = tokenizer(input_str, return_tensors="pt").to("cuda:0")

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

with fwd_with_hooks([success_steer_config.to_hook(model)], model):
    outputs = model(**inputs)
logits = outputs.logits[0, -1, :]
top_token_id = int(torch.argmax(logits).item())
top_token = tokenizer.decode([top_token_id])
print(f"Top token: {top_token}")

metric=logit_diff_metric(model, top_token_id, no_token_id)

# %%
# success to failure
source_acts = fwd_record_all(
    model=model,
    inputs=inputs,
    hooks=[success_steer_config.to_hook(model)],
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
    steer_configs=[failure_steer_config],
    metric=metric,
    source=source_acts,
    attribution_start_pos=double_newline_pos,
)

# # Accumulate layer-wise attribution (sum over tokens)
# agg = result.aggregate(method="sum")
# for comp in agg:
#     ctrl_to_steer_accum[comp] += agg[comp]


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

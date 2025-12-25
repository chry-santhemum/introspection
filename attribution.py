# %%
import os
os.environ["HF_HOME"] = "/root/hf"

from dataclasses import dataclass
from typing import Literal, Optional
from jaxtyping import Float
from datasets import load_dataset, Dataset

import torch
from torch import Tensor
import matplotlib.pyplot as plt

from utils_model import (
    load_model,
    load_tokenizer,
    get_resid_block_name,
    get_num_layers,
    FwdHook,
    fwd_with_hooks,
    SteerConfig,
    scale_vec,
)
from concept_vectors import introspection_inputs

# %% --- Module naming helpers ---

ComponentType = Literal["resid", "attn", "mlp"]

def get_all_component_names(model, components: list[ComponentType]) -> dict[tuple[ComponentType, int], str]:
    """Get module names for all components at all layers.

    Returns:
        Dict mapping (component, layer) -> module_name
    """
    num_layers = get_num_layers(model)
    result = {}
    for layer in range(num_layers):
        for comp in components:
            base = get_resid_block_name(model, layer)
            match comp:
                case "resid":
                    name = base
                case "attn":
                    name = base + ".self_attn"
                case "mlp":
                    name = base + ".mlp"
            result[(comp, layer)] = name
    return result


# --- Compute mean activations as head ablation target ---

@dataclass
class MeanActivations:
    """Averaged across token positions"""
    resid: Float[Tensor, "num_layers hidden"]
    attn: Float[Tensor, "num_layers hidden"]
    mlp: Float[Tensor, "num_layers hidden"]

    def get(self, component: str) -> Tensor:
        return getattr(self, component)

def compute_mean_activations(
    model,
    tokenizer,
    dataset: Dataset,
    num_tokens: int = 100_000,
    seq_len: int = 1024,
    text_column: str = "text",
    components: list[ComponentType] = ["resid", "attn", "mlp"],
) -> MeanActivations:
    """dataset["text"] should be pre-formatted"""
    device = next(model.parameters()).device
    num_layers = get_num_layers(model)
    module_names = get_all_component_names(model, components)

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

    # Running sum and count for online mean computation
    running_sum = {key: None for key in module_names.keys()}
    total_tokens = 0

    def process_chunk(inputs):
        """Process a single chunk and accumulate activations."""
        nonlocal total_tokens

        chunk_len = inputs["input_ids"].shape[1]
        if chunk_len == 0:
            return

        # Reset hook tensors
        for hook in record_hooks.values():
            hook.tensor = None

        # Forward pass
        with fwd_with_hooks(list(record_hooks.values()), model, allow_grad=False):
            model(**inputs)

        # Accumulate activations (online mean computation)
        for key, hook in record_hooks.items():
            act = hook.tensor  # [1, seq_len, hidden]
            act_sum = act[0].sum(dim=0).detach().cpu()  # [hidden]

            if running_sum[key] is None:
                running_sum[key] = act_sum
            else:
                running_sum[key] += act_sum

        total_tokens += chunk_len

    # Process text, accumulating into buffer to fill seq_len chunks
    text_buffer = ""
    text_idx = 0

    while total_tokens < num_tokens and text_idx < len(dataset):
        # Accumulate text until we have enough for a chunk
        while len(tokenizer.encode(text_buffer)) < seq_len and text_idx < len(dataset):
            text = dataset[text_idx][text_column]
            if text and text.strip():
                text_buffer += text + " "
            text_idx += 1

        if len(tokenizer.encode(text_buffer)) < seq_len:
            break

        # Tokenize a chunk
        inputs = tokenizer(
            text_buffer,
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
            add_special_tokens=False,
        ).to(device)

        # Remove processed tokens from buffer
        processed_text = tokenizer.decode(inputs["input_ids"][0])
        text_buffer = text_buffer[len(processed_text):].lstrip()

        process_chunk(inputs)

        if total_tokens % 2000 < seq_len:
            print(f"Processed {total_tokens}/{num_tokens} tokens")

    print(f"Finished processing {total_tokens} tokens")

    # Compute final means
    mean_acts = {comp: [] for comp in components}
    for (comp, _layer), act_sum in sorted(running_sum.items(), key=lambda x: (x[0][0], x[0][1])):
        mean = act_sum / total_tokens  # [hidden]
        mean_acts[comp].append(mean)

    # Stack layers: [num_layers, hidden]
    result = {}
    for comp in components:
        result[comp] = torch.stack(mean_acts[comp], dim=0)

    return MeanActivations(
        resid=result.get("resid", torch.zeros(num_layers, 1)),
        attn=result.get("attn", torch.zeros(num_layers, 1)),
        mlp=result.get("mlp", torch.zeros(num_layers, 1)),
    )


# --- Attribution computation ---


@dataclass
class AttributionResult:
    resid: Float[Tensor, "num_layers seq_len"]
    attn: Float[Tensor, "num_layers seq_len"]
    mlp: Float[Tensor, "num_layers seq_len"]

    # Metadata
    tokens: list[str]
    target_token: str
    target_logit: float

    def get(self, component: ComponentType) -> Tensor:
        return getattr(self, component)

    def aggregate(self, method: Literal["sum", "mean", "abs_sum", "abs_mean"] = "sum") -> dict[ComponentType, Tensor]:
        result = {}
        for comp in ["resid", "attn", "mlp"]:
            attr = self.get(comp)  # [num_layers, seq_len]
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
    target_token_id: int,
    mean_activations: MeanActivations,
    components: list[ComponentType] = ["resid", "attn", "mlp"],
    attr_start_pos: int = 0,
) -> AttributionResult:
    """attribution = d(target_logit)/d(component_output) · (activation - mean_activation)"""
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
        outputs = model(**fwd_inputs)

        # Get target logit
        logits = outputs.logits[0, -1, :]  # [vocab]
        target_logit = logits[target_token_id]

        # Backward pass
        target_logit.backward()

    # Compute attribution: grad · (activation - mean)
    # Only for positions from attr_start_pos onwards
    attribution = {comp: torch.zeros(num_layers, attr_seq_len) for comp in components}

    for key, hook in record_hooks.items():
        comp, layer = key
        act = hook.tensor  # [1, full_seq, hidden]
        grad = hook.grad  # [1, full_seq, hidden] - captured by backward hook

        if grad is None:
            print(f"Warning: No gradient for {comp} layer {layer}")
            continue

        mean = mean_activations.get(comp)[layer].to(device)  # [hidden]

        # Attribution at each position from attr_start_pos onwards
        for i, pos in enumerate(range(attr_start_pos, full_seq_len)):
            act_pos = act[0, pos, :]  # [hidden]
            grad_pos = grad[0, pos, :]  # [hidden]
            diff = act_pos - mean
            attr = (grad_pos * diff).sum().item()
            attribution[comp][layer, i] = attr

    # Get token strings for metadata (only from attr_start_pos onwards)
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0, attr_start_pos:].tolist()]
    target_token = tokenizer.decode([target_token_id])

    return AttributionResult(
        resid=attribution.get("resid", torch.zeros(num_layers, attr_seq_len)),
        attn=attribution.get("attn", torch.zeros(num_layers, attr_seq_len)),
        mlp=attribution.get("mlp", torch.zeros(num_layers, attr_seq_len)),
        tokens=tokens,
        target_token=target_token,
        target_logit=target_logit.item(),
    )


# %% --- Visualization helpers ---


def plot_attribution_heatmap(
    attr: Tensor,
    tokens: list[str],
    title: str = "",
    cmap: str = "RdBu_r",
    show_colorbar: bool = True,
    cell_size: float = 0.15,
):
    """Plot a [num_layers, seq_len] heatmap of attribution scores."""
    num_layers, seq_len = attr.shape

    # Calculate figure size to get square cells
    fig_width = seq_len * cell_size + 2  # extra space for labels/colorbar
    fig_height = num_layers * cell_size + 1.5  # extra space for title/labels

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Symmetric colormap centered at 0
    vmax = max(abs(attr.min().item()), abs(attr.max().item()))
    vmin = -vmax

    im = ax.imshow(attr.numpy(), aspect="equal", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xlabel("Token")
    ax.set_ylabel("Layer")
    ax.set_title(title)

    # Token labels on x-axis
    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels([repr(t)[1:-1] for t in tokens], rotation=45, ha="right", fontsize=8)

    if show_colorbar:
        plt.colorbar(im, ax=ax, label="Attribution")

    plt.tight_layout()
    return fig, ax


COMPONENT_NAMES = {"resid": "Residual Stream", "attn": "Attention", "mlp": "MLP"}


def save_attribution_plots(
    result: AttributionResult,
    save_dir: str,
    prefix: str = "",
    components: list[ComponentType] = ["resid", "attn", "mlp"],
    cell_size: float = 0.15,
):
    """Save separate attribution heatmaps for each component.

    Args:
        result: AttributionResult from gradient_attribution
        save_dir: Directory to save plots
        prefix: Prefix for filenames (e.g., concept word)
        components: Which components to plot
        cell_size: Size of each cell in inches
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    for comp in components:
        attr = result.get(comp)
        title = f'{COMPONENT_NAMES[comp]} Attribution (target: "{result.target_token}", logit: {result.target_logit:.2f})'
        fig, ax = plot_attribution_heatmap(attr, result.tokens, title=title, cell_size=cell_size)

        filename = f"{prefix}_{comp}.png" if prefix else f"{comp}.png"
        filepath = os.path.join(save_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {filepath}")


def plot_layer_attribution(
    result: AttributionResult,
    components: list[ComponentType] = ["resid", "attn", "mlp"],
    agg_method: Literal["sum", "mean", "abs_sum", "abs_mean"] = "sum",
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """Plot attribution scores aggregated over tokens for each layer.

    Args:
        result: AttributionResult from gradient_attribution
        components: Which components to plot
        agg_method: Aggregation method over tokens
        title: Plot title (auto-generated if None)
        figsize: Figure size in inches

    Returns:
        (fig, ax) tuple
    """
    agg = result.aggregate(method=agg_method)
    num_layers = result.resid.shape[0]
    layers = range(num_layers)

    fig, ax = plt.subplots(figsize=figsize)

    colors = {"resid": "#1f77b4", "attn": "#ff7f0e", "mlp": "#2ca02c"}
    for comp in components:
        ax.plot(layers, agg[comp].numpy(), label=COMPONENT_NAMES[comp], color=colors[comp], linewidth=1.5)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Attribution ({agg_method})")
    ax.legend()

    if title is None:
        title = f'Layer Attribution (target: "{result.target_token}", logit: {result.target_logit:.2f})'
    ax.set_title(title)

    plt.tight_layout()
    return fig, ax


def save_layer_attribution_plot(
    result: AttributionResult,
    save_dir: str,
    prefix: str = "",
    components: list[ComponentType] = ["resid", "attn", "mlp"],
    agg_method: Literal["sum", "mean", "abs_sum", "abs_mean"] = "sum",
):
    """Save layer attribution plot.

    Args:
        result: AttributionResult from gradient_attribution
        save_dir: Directory to save plot
        prefix: Prefix for filename
        components: Which components to plot
        agg_method: Aggregation method over tokens
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    fig, ax = plot_layer_attribution(result, components=components, agg_method=agg_method)

    filename = f"{prefix}_layer_attribution.png" if prefix else "layer_attribution.png"
    filepath = os.path.join(save_dir, filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filepath}")


# %% --- Experiment script ---

if __name__ == "__main__":
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
    ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

    def preprocess_item(item: dict):
        formatted_text = tokenizer.apply_chat_template(item["messages"], tokenize=False)
        return {"text": formatted_text}
    
    ultrachat_to_use = ultrachat.select(range(5000)).map(preprocess_item, num_proc=8)
    
    mean_acts = compute_mean_activations(
        model, 
        tokenizer,
        ultrachat_to_use,
    )

    print(f"Mean activations shape: resid={mean_acts.resid.shape}, attn={mean_acts.attn.shape}, mlp={mean_acts.mlp.shape}")

    # Save mean activations for reuse
    torch.save(mean_acts, "attribution/mean_acts-27b-it.pt")

    # %%
    # Load concept vectors
    concept_vectors = torch.load("concept_vectors/concept_diff-27b-L43/concepts.pt", weights_only=True)

    inputs = introspection_inputs(tokenizer)

    # Find double newline position for steering
    decoded_tokens = [tokenizer.decode(x) for x in inputs["input_ids"][0]]
    double_newline_pos = None
    for i, tok in enumerate(decoded_tokens):
        if tok == "\n\n":
            double_newline_pos = i
            break

    # %%
    # Run attribution for a concept word
    word = "Illusions"
    vec = concept_vectors[word]  # TODO: change these keys
    word = word.lower()

    steer_config = SteerConfig(
        layer=43,
        tokens=slice(double_newline_pos, None),
        vec=vec,
        strength=4.0,
    )

    # Get token ID for the concept word
    word_token_ids = tokenizer.encode(word, add_special_tokens=False)
    print(f"Token IDs for '{word}': {word_token_ids}")
    target_token_id = word_token_ids[0]

    # %%
    # Compute attribution (only for steered tokens)
    print(f"Computing attribution for '{word}'...")
    result = gradient_attribution(
        model=model,
        tokenizer=tokenizer,
        inputs=inputs,
        steer_config=steer_config,
        target_token_id=target_token_id,
        mean_activations=mean_acts,
        attr_start_pos=double_newline_pos,
    )

    # %%
    save_attribution_plots(result, save_dir="attribution", prefix=word)
    save_layer_attribution_plot(result, save_dir="attribution", prefix=word)

# %%

# %%
import os
os.environ["HF_HOME"] = "/root/hf"

from dataclasses import dataclass
from typing import Callable, Literal, Optional
from jaxtyping import Float
from pathlib import Path
from datasets import load_dataset, Dataset

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
    fwd_record_all,
)
from concept_vectors import introspection_inputs, ASSISTANT_PREFILL


def compute_mean_activations(
    model,
    tokenizer,
    dataset: Dataset,
    num_tokens: int = 100_000,
    seq_len: int = 1024,
    text_column: str = "text",
    components: list[ComponentType] = ["resid", "attn", "mlp"],
) -> Activations:
    """Compute mean activations across dataset, returned as Activations with seq_len=1."""
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

    # Stack layers and add seq_len=1 dim: [num_layers, 1, hidden]
    result = {}
    for comp in components:
        stacked = torch.stack(mean_acts[comp], dim=0)  # [num_layers, hidden]
        result[comp] = stacked.unsqueeze(1)  # [num_layers, 1, hidden]

    hidden_dim = result[components[0]].shape[-1]
    return Activations(
        resid=result.get("resid", torch.zeros(num_layers, 1, hidden_dim)),
        attn=result.get("attn", torch.zeros(num_layers, 1, hidden_dim)),
        mlp=result.get("mlp", torch.zeros(num_layers, 1, hidden_dim)),
    )


# --- Attribution computation ---

# Metric function: takes final hidden state [hidden_dim] and returns a scalar tensor
Metric = Callable[[Tensor], Tensor]


def get_final_norm(model):
    """Get the final RMSNorm/LayerNorm from various model architectures."""
    # Try different paths for different architectures
    paths_to_try = [
        lambda m: m.model.language_model.model.norm,  # Gemma-3 multimodal
        lambda m: m.model.language_model.norm,        # Some multimodal
        lambda m: m.model.norm,                       # Standard HF
        lambda m: m.transformer.ln_f,                 # GPT-2 style
        lambda m: m.norm,                             # Direct
    ]
    for path in paths_to_try:
        try:
            norm = path(model)
            if norm is not None:
                return norm
        except AttributeError:
            continue
    raise ValueError("Could not find final norm layer in model")


def logit_metric(model, token_id: int) -> Metric:
    """Create a metric that returns the logit for a specific token.

    Applies the model's final norm and lm_head to get logits.
    """
    norm = get_final_norm(model)

    def metric(hidden: Tensor) -> Tensor:
        normed = norm(hidden)
        logits = model.lm_head(normed)
        return logits[token_id]
    return metric


def logit_diff_metric(model, pos_token_id: int, neg_token_id: int) -> Metric:
    """Create a metric for the difference between two token logits."""
    norm = get_final_norm(model)

    def metric(hidden: Tensor) -> Tensor:
        normed = norm(hidden)
        logits = model.lm_head(normed)
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
    metric: Metric,
    baseline: Activations,
    components: list[ComponentType] = ["resid", "attn", "mlp"],
    attr_start_pos: int = 0,
) -> AttributionResult:
    """Gradient-based attribution for activation patching approximation.

    Computes: attribution = d(metric)/d(component_output) · (activation - baseline)

    Args:
        metric: Function that takes logits [vocab] and returns a scalar tensor.
                Use logit_metric(token_id) for single token attribution.
        baseline: Activations with shape [num_layers, seq_len, hidden].
                  - seq_len=1: broadcasts to all positions (mean ablation)
                  - seq_len=full: position-specific (patching from another run)
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

        # Get final hidden state (last layer, last position)
        final_hidden = outputs.hidden_states[-1][0, -1, :]  # [hidden_dim]

        # # Print top 10 most probable tokens
        # logits = outputs.logits[0, -1, :]  # [vocab]
        # probs = torch.softmax(logits.float(), dim=-1)
        # top_probs, top_indices = torch.topk(probs, k=10)
        # print("Top 10 tokens:")
        # for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        #     token = tokenizer.decode([idx])
        #     print(f"  {i+1}. {repr(token):15s} prob={prob.item():.6f}")

        # Compute metric and backward
        metric_value = metric(final_hidden)
        metric_value.backward()

    # Compute attribution: grad · (activation - baseline)
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
            base = baseline.get_at(comp, layer, pos).to(device)  # [hidden]

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


def plot_attribution_heatmap(
    attr: Tensor,
    tokens: list[str],
    title: str = "",
    cmap: str = "RdBu_r",
    show_colorbar: bool = True,
    cell_size: float = 0.15,
    start_layer: int = 0,
):
    """Plot a [num_layers, seq_len] heatmap of attribution scores.

    Args:
        start_layer: Only plot from this layer onwards (useful when steering at a specific layer).
    """
    # Crop to layers from start_layer onwards
    attr = attr[start_layer:]
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

    # Layer labels on y-axis (showing actual layer numbers)
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels([str(i + start_layer) for i in range(num_layers)])

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
    start_layer: int = 0,
):
    """Save separate attribution heatmaps for each component.

    Args:
        result: AttributionResult from gradient_attribution
        save_dir: Directory to save plots
        prefix: Prefix for filenames (e.g., concept word)
        components: Which components to plot
        cell_size: Size of each cell in inches
        start_layer: Only plot from this layer onwards
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    for comp in components:
        attr = result.get(comp)
        title = f'{COMPONENT_NAMES[comp]} Attribution (metric: {result.metric_value:.2f})'
        fig, _ = plot_attribution_heatmap(attr, result.tokens, title=title, cell_size=cell_size, start_layer=start_layer)

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
    start_layer: int = 0,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot attribution scores aggregated over tokens for each layer.

    Args:
        result: AttributionResult from gradient_attribution
        components: Which components to plot
        agg_method: Aggregation method over tokens
        title: Plot title (auto-generated if None)
        figsize: Figure size in inches
        start_layer: Only plot from this layer onwards

    Returns:
        (fig, ax) tuple
    """
    agg = result.aggregate(method=agg_method)
    num_layers = result.resid.shape[0]
    layers = range(start_layer, num_layers)

    fig, ax = plt.subplots(figsize=figsize)

    colors = {"resid": "#1f77b4", "attn": "#ff7f0e", "mlp": "#2ca02c"}
    for comp in components:
        ax.plot(layers, agg[comp][start_layer:].numpy(), label=COMPONENT_NAMES[comp], color=colors[comp], linewidth=1.5)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Attribution ({agg_method})")
    ax.legend()

    if title is None:
        title = f'Layer Attribution (metric: {result.metric_value:.2f})'
    ax.set_title(title)

    plt.tight_layout()
    return fig, ax


def save_layer_attribution_plot(
    result: AttributionResult,
    save_dir: str,
    prefix: str = "",
    components: list[ComponentType] = ["resid", "attn", "mlp"],
    agg_method: Literal["sum", "mean", "abs_sum", "abs_mean"] = "sum",
    start_layer: int = 0,
):
    """Save layer attribution plot.

    Args:
        result: AttributionResult from gradient_attribution
        save_dir: Directory to save plot
        prefix: Prefix for filename
        components: Which components to plot
        agg_method: Aggregation method over tokens
        start_layer: Only plot from this layer onwards
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    fig, _ = plot_layer_attribution(result, components=components, agg_method=agg_method, start_layer=start_layer)

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
    from concept_vectors import SUCCESS_WORDS
    print(SUCCESS_WORDS)

    base_path = Path(f"concept_vectors/concept_diff-27b-it-L{LAYER}")
    concept_vectors = torch.load(base_path / "concepts.pt", weights_only=True)

    inputs = introspection_inputs(
        tokenizer, 
        append=" Answer first with either Yes or No, then answer the question." + " ."*32, 
        prefill=None
    )

    # Find double newline position for steering
    decoded_tokens = [tokenizer.decode(x) for x in inputs["input_ids"][0]]

    for i, tok in enumerate(decoded_tokens):
        if tok == "\n\n":
            double_newline_pos = i
            break

    yes_token_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]
    print(f"Token IDs for 'Yes': {yes_token_id}, 'No': {no_token_id}")

    # %%
    for word in SUCCESS_WORDS:
        steer_config = SteerConfig(
            layer=LAYER,
            tokens=slice(double_newline_pos, None),
            vec=concept_vectors[word],
            strength=4.0,
        )

        source_acts = fwd_record_all(
            model=model,
            inputs=inputs,
            steer_config=steer_config,
            components=["resid", "attn", "mlp"],
        )

        # Print top 10 most probable tokens
        last_layer_acts = source_acts.get("resid")[-1, -1, :]  # [hidden]
        final_norm = model.model.language_model.norm  # RMSNorm(hidden_dim)
        unembed = model.lm_head.weight  # shape: [vocab_size, hidden_dim]
        logits = unembed @ final_norm(last_layer_acts.to(unembed.device))

        probs = torch.softmax(logits.float(), dim=-1)
        top_probs, top_indices = torch.topk(probs, k=10)
        print("Top 10 tokens:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            token = tokenizer.decode([idx])
            print(f"  {i+1}. {repr(token):15s} prob={prob.item():.6f}")

        result = gradient_attribution(
            model=model,
            tokenizer=tokenizer,
            inputs=inputs,
            steer_config=None,
            metric=logit_diff_metric(model, yes_token_id, no_token_id),
            baseline=source_acts,
            attr_start_pos=double_newline_pos,
        )

        save_attribution_plots(result, save_dir="attribution/steer_to_ctrl", prefix=word, start_layer=LAYER)
        save_layer_attribution_plot(result, save_dir="attribution/steer_to_ctrl", prefix=word, start_layer=LAYER)

# %%

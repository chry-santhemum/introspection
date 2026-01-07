from dataclasses import dataclass
from typing import Callable, Literal, Sequence
from jaxtyping import Float
from pathlib import Path

import torch
from torch import Tensor
import matplotlib.pyplot as plt

from utils_model import (
    get_num_layers,
    get_all_component_names,
    ComponentType,
    FwdHook,
    fwd_with_hooks,
    Activations,
)


Metric = Callable[[Tensor], Tensor]  # [hidden_dim] -> scalar


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
    components: Sequence[ComponentType] = ("resid", "attn", "mlp"),
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


COMPONENT_NAMES = {"resid": "Residual Stream", "attn": "Attention", "mlp": "MLP"}

def save_attribution_heatmaps(
    result: AttributionResult,
    save_dir: Path,
    prefix: str = "",
    start_layer: int = 0,
    title: str | None = None,
    cell_size: float = 0.15,
):
    """Save attribution heatmaps for each component (resid, attn, mlp)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
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

        filename = f"{prefix}_{comp}.png" if prefix else f"{comp}.png"
        filepath = save_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {filepath}")


def save_layer_attribution(
    attrs: AttributionResult | dict[ComponentType, Tensor],
    save_dir: Path,
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

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_layer_attribution.png" if prefix else "layer_attribution.png"
    filepath = save_dir / filename
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filepath}")

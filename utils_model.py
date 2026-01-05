from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, Optional, Union
from jaxtyping import Float, Int
from datasets import Dataset

import torch
from torch import nn, Tensor
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


# --- Model loading ---

def load_model(model_name: str, **kwargs):
    """**kwargs are Passed to AutoModelForCausalLM.from_pretrained
    
    e.g. device, dtype, quantization_config"""

    print(f"Loading model: {model_name}")
    if kwargs:
        print("Arguments:")
        for k, v in kwargs.items():
            print(f"  {k}: {v}")

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("Model loaded (eval mode, grads disabled)")
    return model

def load_tokenizer(model_name: str, **kwargs):
    """Load tokenizer, handling multimodal models that use AutoProcessor."""
    try:
        return AutoTokenizer.from_pretrained(model_name, **kwargs)
    except (OSError, ValueError):
        processor = AutoProcessor.from_pretrained(model_name, **kwargs)
        return processor.tokenizer

def get_module(model, module_name: str) -> nn.Module:
    module = model
    for part in module_name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module

def get_resid_block_name(model, layer: int) -> str:
    """Get the residual block module name for a given layer.

    Detects model architecture and returns appropriate path.
    """
    name = ""
    # print(model)
    if hasattr(model, "model"):
        name += "model."
        model = model.model
    if hasattr(model, "language_model"):
        name += "language_model."
        model = model.language_model
    if hasattr(model, "transformer"):
        name += "transformer."
        model = model.transformer
    if hasattr(model, "layers"):
        name += "layers."
        model = model.layers
    if hasattr(model, "h"):
        name += "h."
        model = model.h
    return name + f"{layer}"

def get_num_layers(model) -> int:
    config = model.config
    config = getattr(config, "text_config", config)
    return config.num_hidden_layers

def get_layer_at_fraction(model, fraction: float) -> int:
    num_layers = get_num_layers(model)
    layer_idx = int(num_layers * fraction)
    return max(0, min(layer_idx, num_layers - 1))



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


def print_top_tokens(logits: Tensor, tokenizer, top_k: int=10) -> None:
    """Print top_k most probable tokens"""
    probs = torch.softmax(logits.float(), dim=-1)
    top_probs, top_indices = torch.topk(probs, k=top_k)
    print(f"Top {top_k} tokens:")
    for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
        token = tokenizer.decode([idx])
        print(f"  {i+1:>3}. {repr(token):20s} prob={prob.item():<10.6f} logit={logits[idx].item():<8.2f}")


# --- Forward hook configuration ---

TokenSpec = Union[list[int], slice]

@dataclass(kw_only=True)
class FwdHook:
    module_name: str
    pos: Literal["input", "output"]
    op: Literal["record", "replace", "add", "proj_ablate"]
    tokens: TokenSpec

    tensor: Optional[Tensor] = None  # will be broadcast to x[tokens]
    grad: Optional[Tensor] = None  
    # populated when backward() is called in fwd_with_hooks(allow_grad=True).
    # this captures grads wrt the modified in/outputs.

    def __post_init__(self):
        if self.op != "record" and self.tensor is None:
            raise ValueError(f"tensor is required for '{self.op}' operation")


def _apply_hook_op(h: FwdHook, x: Tensor) -> Optional[Tensor]:
    """Apply hook operation to x.

    Assumes KV cache is used when seq_len=1.
    """
    idx = h.tokens

    # Handle KV cache decode step (seq_len=1)
    # Generated tokens should be steered if h.tokens is unbounded
    if x.shape[1] == 1:
        if isinstance(idx, slice) and idx.stop is None:
            idx = slice(None)  # steer the single generated token
        else:
            return None

    if h.op == "record":
        if h.tensor is None:
            h.tensor = x[:, idx, :].clone().detach()
        else:
            h.tensor.copy_(x[:, idx, :])
        return None

    new_tensor = x.clone()
    vec = h.tensor.to(device=x.device, dtype=x.dtype)

    if h.op == "add":
        new_tensor[:, idx, :] += vec
    elif h.op == "replace":
        new_tensor[:, idx, :] = vec
    elif h.op == "proj_ablate":
        v_hat = vec / vec.norm(dim=-1, keepdim=True)
        proj_coef = (new_tensor[:, idx, :] * v_hat).sum(dim=-1, keepdim=True)  # [batch, num_tokens, 1]
        new_tensor[:, idx, :] -= proj_coef * vec

    return new_tensor


def _get_tensor(x):
    return x[0] if isinstance(x, tuple) else x

def _set_tensor(original, new_tensor):
    if isinstance(original, tuple):
        return (new_tensor,) + original[1:]
    return new_tensor


def make_fwd_hook(h: FwdHook):
    if h.pos == "output":
        def output_hook(module, input, output):
            result = _apply_hook_op(h, _get_tensor(output))
            return _set_tensor(output, result) if result is not None else None
        return output_hook
    else:
        def input_hook(module, input):
            result = _apply_hook_op(h, _get_tensor(input))
            return _set_tensor(input, result) if result is not None else None
        return input_hook


def make_bwd_hook(h: FwdHook):
    """Capture gradients for a FwdHook."""
    def grad_hook(module, grad_input, grad_output):
        if h.pos == "output":
            grad_tensor = grad_output[0]
        else:
            grad_tensor = grad_input[0]
        
        if grad_tensor is not None:
            idx = h.tokens
            h.grad = grad_tensor[:, idx, :].clone().detach()

        return None
    return grad_hook


@contextmanager
def fwd_with_hooks(hooks: list[FwdHook], model, allow_grad: bool = False):
    """Context manager for running forward pass with hooks.
    
    allow_grad: If False (default), runs in inference_mode with no gradients.
                If True, gradients are enabled and backward hooks are registered.
    """
    handles = []

    # register hooks
    for hook in hooks:
        module = get_module(model, hook.module_name)
        if hook.pos == "output":
            handle = module.register_forward_hook(make_fwd_hook(hook))
        else:
            handle = module.register_forward_pre_hook(make_fwd_hook(hook))
        handles.append(handle)

        # Register backward hooks to capture gradients
        if allow_grad:
            bwd_handle = module.register_full_backward_hook(make_bwd_hook(hook))
            handles.append(bwd_handle)

    if allow_grad:
        with torch.enable_grad():
            yield
    else:
        with torch.inference_mode():
            yield

    for handle in handles:
        handle.remove()


# --- Steering and patching hooks ---

def scale_vec(vec: Tensor, strength: Tensor|float) -> Tensor:
    """Scale steering vector by strength, then unsqueeze to broadcast to BSH"""

    if isinstance(strength, (int, float)):
        scaled = vec * strength
    else:
        scaled = vec * strength.unsqueeze(-1)  # strength [batch] → [batch, 1] for broadcasting

    # If result has batch dim, need [batch, 1, hidden] for [batch, seq, hidden] broadcast
    if scaled.dim() == 2:
        scaled = scaled.unsqueeze(1)

    return scaled

@dataclass(kw_only=True)
class SteerConfig:
    layer: int
    tokens: TokenSpec
    vec: Tensor  # will be broadcast to x[tokens]
    strength: Tensor|float = 1.0  # scalar or [batch]

    def to_hook(self, model) -> FwdHook:
        return FwdHook(
            module_name=get_resid_block_name(model, self.layer),
            pos="output",
            op="add",
            tokens=self.tokens,
            tensor=scale_vec(self.vec, self.strength),
        )


@dataclass
class Activations:
    """Activations with shape [num_layers, seq_len, hidden].

    For mean activations, seq_len=1 and broadcasts to any position.
    For recorded forward pass, seq_len=actual_seq_len.
    """
    resid: Float[Tensor, "num_layers seq_len hidden"]
    attn: Float[Tensor, "num_layers seq_len hidden"]
    mlp: Float[Tensor, "num_layers seq_len hidden"]

    def get_at(self, comp: ComponentType, layer: int, pos: int) -> Float[Tensor, "hidden"]:
        """Get activation at (layer, pos), broadcasting if seq_len=1."""
        tensor = getattr(self, comp)
        actual_pos = 0 if tensor.shape[1] == 1 else pos
        return tensor[layer, actual_pos, :]

    def __repr__(self) -> str:
        return (
            f"Activations(resid={list(self.resid.shape)}, "
            f"attn={list(self.attn.shape)}, "
            f"mlp={list(self.mlp.shape)})"
        )

@dataclass
class PatchConfig:
    source: Activations
    patches: dict[tuple[ComponentType, int], TokenSpec]  # (comp, layer) -> tokens

    @classmethod
    def from_modules(
        cls,
        source: Activations,
        modules: list[tuple[ComponentType, int]],
        tokens: TokenSpec,
    ) -> "PatchConfig":
        """Create PatchConfig with same tokens for all modules."""
        return cls(source=source, patches={m: tokens for m in modules})
    
    def to_hooks(self, model) -> list[FwdHook]:
        hooks = []
        component_names = get_all_component_names(model, ["resid", "attn", "mlp"])

        for (comp, layer), tokens in self.patches.items():
            module_name = component_names[(comp, layer)]
            # Get source activation at the positions we're patching: [num_tokens, hidden] -> [1, num_tokens, hidden]
            source_layer = getattr(self.source, comp)[layer]  # [seq_len, hidden]
            source_tensor = source_layer[tokens].unsqueeze(0)  # [1, num_tokens, hidden]

            hooks.append(FwdHook(
                module_name=module_name,
                pos="output",
                op="replace",
                tokens=tokens,
                tensor=source_tensor,
            ))

        return hooks


# --- Activation recording ---

def fwd_record_resid(
    model,
    inputs: dict[str, Tensor],
    layer: int,
    pos: int = -1,
) -> Float[Tensor, "batch hidden"]:
    """Record residual stream activations at a specific layer and position."""
    module_name = get_resid_block_name(model, layer)

    # slice for a single position
    tokens = slice(pos, pos + 1 if pos != -1 else None)

    record_hook = FwdHook(
        module_name=module_name,
        pos="output",
        op="record",
        tokens=tokens,
    )

    with fwd_with_hooks([record_hook], model, allow_grad=False):
        model(**inputs)

    # record_hook.tensor is [batch, 1, hidden], squeeze to [batch, hidden]
    return record_hook.tensor.squeeze(1)


def fwd_record_all(
    model,
    inputs: dict[str, Tensor],
    hooks: list[FwdHook],
    components: list[ComponentType] = ["resid", "attn", "mlp"],
) -> Activations:
    """Forward pass recording activations at all layers for specified components."""
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    num_layers = get_num_layers(model)
    seq_len = inputs["input_ids"].shape[1]
    module_names = get_all_component_names(model, components)

    # Create record hooks for all components at all layers
    record_hooks = {
        key: FwdHook(
            module_name=module_name,
            pos="output",
            op="record",
            tokens=slice(None),  # Record all positions
        )
        for key, module_name in module_names.items()
    }

    all_hooks = list(record_hooks.values())
    all_hooks.extend(hooks)

    with fwd_with_hooks(all_hooks, model, allow_grad=False):
        _ = model(**inputs)

    # Collect activations: [num_layers, seq_len, hidden]
    activations = {comp: [] for comp in components}
    for layer in range(num_layers):
        for comp in components:
            hook = record_hooks[(comp, layer)]
            act = hook.tensor[0].detach().cpu()  # [seq_len, hidden]
            activations[comp].append(act)

    result = {}
    for comp in components:
        result[comp] = torch.stack(activations[comp], dim=0)

    hidden_dim = result[components[0]].shape[-1]
    return Activations(
        resid=result.get("resid", torch.zeros(num_layers, seq_len, hidden_dim)),
        attn=result.get("attn", torch.zeros(num_layers, seq_len, hidden_dim)),
        mlp=result.get("mlp", torch.zeros(num_layers, seq_len, hidden_dim)),
    )


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

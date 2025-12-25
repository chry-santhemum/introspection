from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, Optional, Union
from jaxtyping import Float, Int

import torch
from torch import nn, Tensor
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


def load_model(model_name: str, **kwargs):
    """**kwargs: Passed to AutoModelForCausalLM.from_pretrained
    
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


# --- Forward hook configuration ---

TokenSpec = Union[list[int], slice]

@dataclass(kw_only=True)
class FwdHook:
    module_name: str
    pos: Literal["input", "output"]
    op: Literal["record", "replace", "add"]
    tokens: TokenSpec
    tensor: Optional[Tensor] = None  # will be broadcast to x[tokens]
    grad: Optional[Tensor] = None  # populated by backward hook when allow_grad=True

    def __post_init__(self):
        if self.op in ("add", "replace") and self.tensor is None:
            raise ValueError(f"tensor is required for '{self.op}' operation")


def _get_tensor(x):
    return x[0] if isinstance(x, tuple) else x

def _set_tensor(original, new_tensor):
    if isinstance(original, tuple):
        return (new_tensor,) + original[1:]
    return new_tensor

def _apply_hook_op(h: FwdHook, tensor: Tensor, retain_grad: bool = False) -> Optional[Tensor]:
    """Apply hook operation. Returns new tensor for add/replace, None for record.

    Handles forward passes using KV cache (seq_len=1).

    Args:
        retain_grad: If True and op is "record", call retain_grad() on the
                     recorded tensor for gradient computation.
    """
    seq_len = tensor.shape[1]
    idx = h.tokens

    # Handle KV cache decode step (seq_len=1)
    # Generated tokens should be steered if spec is unbounded (open-ended slice)
    if seq_len == 1:
        if isinstance(idx, slice) and idx.stop is None:
            idx = slice(None)  # steer the single generated token
        elif isinstance(idx, list):
            return None  # explicit positions don't include generated tokens
        else:  # Bounded slice, don't steer generated tokens
            return None

    if h.op == "record":
        if h.tensor is None:
            h.tensor = tensor[:, idx, :].clone()
        else:
            h.tensor.copy_(tensor[:, idx, :])
        if retain_grad:
            h.tensor.retain_grad()
        return None

    new_tensor = tensor.clone()
    vec = h.tensor.to(device=tensor.device, dtype=tensor.dtype)

    if h.op == "add":
        new_tensor[:, idx, :] += vec
    else:
        new_tensor[:, idx, :] = vec

    return new_tensor


def make_hook(h: FwdHook, retain_grad: bool = False):
    if h.pos == "output":
        def hook_fn(module, input, output):
            result = _apply_hook_op(h, _get_tensor(output), retain_grad=retain_grad)
            return _set_tensor(output, result) if result is not None else None
        return hook_fn
    else:
        def hook_fn(module, input):
            result = _apply_hook_op(h, _get_tensor(input), retain_grad=retain_grad)
            return _set_tensor(input, result) if result is not None else None
        return hook_fn


def make_backward_hook(h: FwdHook):
    """Create backward hook to capture gradients for a FwdHook."""
    def hook_fn(module, grad_input, grad_output):
        # grad_output is tuple, first element is the gradient w.r.t. output
        grad_tensor = grad_output[0]
        if grad_tensor is not None:
            idx = h.tokens
            # Handle same slicing as forward hook
            h.grad = grad_tensor[:, idx, :].clone().detach()
        return None
    return hook_fn


@contextmanager
def fwd_with_hooks(hooks: list[FwdHook], model, allow_grad: bool = False):
    """Context manager for running forward pass with hooks.

    Args:
        hooks: List of hook configurations
        model: The model to hook
        allow_grad: If False (default), runs in inference_mode with no gradients.
                    If True, gradients are enabled and backward hooks are
                    registered to capture gradients for "record" ops.
    """
    handles = []

    # register hooks
    for hook in hooks:
        module = get_module(model, hook.module_name)
        if hook.pos == "output":
            handle = module.register_forward_hook(make_hook(hook, retain_grad=allow_grad))
        else:
            handle = module.register_forward_pre_hook(make_hook(hook, retain_grad=allow_grad))
        handles.append(handle)

        # Register backward hooks to capture gradients for record ops
        if allow_grad and hook.op == "record":
            bwd_handle = module.register_full_backward_hook(make_backward_hook(hook))
            handles.append(bwd_handle)

    if allow_grad:
        yield
    else:
        with torch.inference_mode():
            yield

    for handle in handles:
        handle.remove()


# --- Recording resid activations wrapper ---

def fwd_record_resid(
    model,
    inputs: dict[str, Tensor],
    layer: int,
    pos: int = -1,
) -> Float[Tensor, "batch hidden"]:
    """Record residual stream activations at a specific layer and position."""
    module_name = get_resid_block_name(model, layer)

    # Use slice for single position to keep [batch, 1, hidden] then squeeze
    tokens = [pos] if pos >= 0 else slice(pos, None)

    record_hook = FwdHook(
        module_name=module_name,
        pos="output",
        op="record",
        tokens=tokens,
    )

    with fwd_with_hooks([record_hook], model):
        model(**inputs)

    # record_hook.tensor is [batch, 1, hidden], squeeze to [batch, hidden]
    return record_hook.tensor.squeeze(1)


# --- Steering wrapper ---

@dataclass(kw_only=True)
class SteerConfig:
    layer: int
    tokens: TokenSpec
    vec: Tensor  # [hidden], will be broadcast to x[tokens]
    strength: float | Tensor = 1.0  # scalar or [batch]


def scale_vec(vec: Tensor, strength: float | Tensor) -> Tensor:
    """Scale steering vector by strength.

    Args:
        vec: [hidden] or [batch, hidden]
        strength: scalar or [batch]

    Returns:
        Scaled vector that broadcasts with [batch, seq, hidden]
    """
    if isinstance(strength, (int, float)):
        scaled = vec * strength
    else:
        # strength [batch] → [batch, 1] for broadcasting
        scaled = vec * strength.unsqueeze(-1)

    # If result has batch dim, need [batch, 1, hidden] for [batch, seq, hidden] broadcast
    if scaled.dim() == 2:
        scaled = scaled.unsqueeze(1)

    return scaled


def fwd_steer(
    steer_config: SteerConfig,
    model,
    inputs: dict[str, Tensor],
    other_hooks: Optional[list[FwdHook]] = None,
) -> Float[Tensor, "batch vocab"]:
    module_name = get_resid_block_name(model, steer_config.layer)

    steer_hook = FwdHook(
        module_name=module_name,
        pos="output",
        op="add",
        tokens=steer_config.tokens,
        tensor=scale_vec(steer_config.vec, steer_config.strength),
    )

    all_hooks = [steer_hook]
    if other_hooks:
        all_hooks.extend(other_hooks)

    with fwd_with_hooks(all_hooks, model):
        outputs = model(**inputs)

    return outputs.logits[:, -1, :]

def generate_steer(
    steer_config: SteerConfig,
    model,
    inputs: dict[str, Tensor],
    other_hooks: list[FwdHook] = None,
    **generate_kwargs,
) -> Int[Tensor, "batch out_seq"]:
    """**generate_kwargs: Passed to model.generate() (max_new_tokens, temperature, do_sample, etc.)"""
    module_name = get_resid_block_name(model, steer_config.layer)

    steer_hook = FwdHook(
        module_name=module_name,
        pos="output",
        op="add",
        tokens=steer_config.tokens,
        tensor=scale_vec(steer_config.vec, steer_config.strength),
    )

    all_hooks = [steer_hook]
    if other_hooks:
        all_hooks.extend(other_hooks)

    with fwd_with_hooks(all_hooks, model):
        output_ids = model.generate(**inputs, use_cache=True, **generate_kwargs)

    return output_ids

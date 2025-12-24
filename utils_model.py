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


def get_layer_at_fraction(model, fraction: float) -> int:
    config = model.config
    config = getattr(config, "text_config", config)
    layer_idx = int(config.num_hidden_layers * fraction)
    return max(0, min(layer_idx, config.num_hidden_layers - 1))


# --- Forward hook configuration ---

TokenSpec = Union[list[int], slice]

@dataclass(kw_only=True)
class FwdHook:
    module_name: str
    pos: Literal["input", "output"]
    op: Literal["record", "replace", "add"]
    tokens: TokenSpec
    tensor: Optional[Tensor] = None  # will be broadcast to x[tokens]

    def __post_init__(self):
        if self.op in ("add", "replace") and self.tensor is None:
            raise ValueError(f"tensor is required for '{self.op}' operation")


def _get_tensor(x):
    return x[0] if isinstance(x, tuple) else x

def _set_tensor(original, new_tensor):
    if isinstance(original, tuple):
        return (new_tensor,) + original[1:]
    return new_tensor

def _apply_hook_op(h: FwdHook, tensor: Tensor) -> Optional[Tensor]:
    """Apply hook operation. Returns new tensor for add/replace, None for record.

    Handles forward passes using KV cache (seq_len=1).
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
        return None

    new_tensor = tensor.clone()
    vec = h.tensor.to(device=tensor.device, dtype=tensor.dtype)

    if h.op == "add":
        new_tensor[:, idx, :] += vec
    else:
        new_tensor[:, idx, :] = vec

    return new_tensor

def make_hook(h: FwdHook):
    if h.pos == "output":
        def hook_fn(module, input, output):
            result = _apply_hook_op(h, _get_tensor(output))
            return _set_tensor(output, result) if result is not None else None
        return hook_fn
    else:
        def hook_fn(module, input):
            result = _apply_hook_op(h, _get_tensor(input))
            return _set_tensor(input, result) if result is not None else None
        return hook_fn


@contextmanager
def fwd_with_hooks(hooks: list[FwdHook], model):
    """Note: grads are disabled in this function."""
    handles = []

    # register hooks
    for hook in hooks:
        module = get_module(model, hook.module_name)
        if hook.pos == "output":
            handle = module.register_forward_hook(make_hook(hook))
        else:
            handle = module.register_forward_pre_hook(make_hook(hook))
        handles.append(handle)

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


def _scale_vec(vec: Tensor, strength: float | Tensor) -> Tensor:
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
        tensor=_scale_vec(steer_config.vec, steer_config.strength),
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
        tensor=_scale_vec(steer_config.vec, steer_config.strength),
    )

    all_hooks = [steer_hook]
    if other_hooks:
        all_hooks.extend(other_hooks)

    with fwd_with_hooks(all_hooks, model):
        output_ids = model.generate(**inputs, use_cache=True, **generate_kwargs)

    return output_ids

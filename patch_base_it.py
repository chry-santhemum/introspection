# %%
import os
os.environ["HF_HOME"] = "/root/hf"

from dataclasses import dataclass
from typing import Literal, Optional

import torch
from torch import Tensor

from utils_model import (
    load_model,
    load_tokenizer,
    get_resid_block_name,
    get_module,
    FwdHook,
    fwd_with_hooks,
    make_hook,
    SteerConfig,
    scale_vec,
    TokenSpec,
)


# %% --- Component type and helpers (from attribution.py) ---

ComponentType = Literal["resid", "attn", "mlp"]


def get_component_module_name(model, layer: int, component: ComponentType) -> str:
    """Get the module name for a specific component at a layer."""
    base = get_resid_block_name(model, layer)
    if component == "resid":
        return base
    elif component == "attn":
        return f"{base}.self_attn"
    elif component == "mlp":
        return f"{base}.mlp"
    else:
        raise ValueError(f"Unknown component: {component}")


def get_num_layers(model) -> int:
    """Get the number of layers in the model."""
    config = model.config
    config = getattr(config, "text_config", config)
    return config.num_hidden_layers


# %% --- Patch configuration ---

@dataclass(kw_only=True)
class PatchSpec:
    """Specification for a single patching location.

    Defines which module to patch and at which token positions.
    """
    component: ComponentType  # "resid", "attn", "mlp"
    layer: int
    tokens: TokenSpec  # which positions to patch

    def get_module_name(self, model) -> str:
        return get_component_module_name(model, self.layer, self.component)


@dataclass
class PatchSimilarity:
    """Similarity metrics between patched and original activations."""
    cosine_sim: float  # cosine similarity (mean across patched positions)
    l2_distance: float  # L2 distance (mean across patched positions)
    relative_norm: float  # ||patched - original|| / ||original||


@dataclass
class PatchLocationResult:
    """Result for a single patched location."""
    spec: PatchSpec
    similarity: PatchSimilarity
    source_activation: Tensor  # [batch, num_tokens, hidden]
    target_original: Tensor  # [batch, num_tokens, hidden] - what was there before patching


@dataclass
class PatchExperimentResult:
    """Full result from a patching experiment."""
    generated_ids: Tensor  # [batch, seq] generated token IDs
    generated_text: list[str]  # decoded text for each batch item
    patch_results: list[PatchLocationResult]  # results per patch location

    # Experiment metadata
    source_model_name: str
    target_model_name: str
    steer_config: Optional[SteerConfig]


# %% --- Activation recording ---

def record_activations(
    model,
    inputs: dict,
    patch_specs: list[PatchSpec],
    steer_config: Optional[SteerConfig] = None,
) -> dict[tuple[ComponentType, int], Tensor]:
    """Record activations at specified locations during a forward pass.

    Args:
        model: The model to record from
        inputs: Tokenized inputs
        patch_specs: List of PatchSpec defining where to record
        steer_config: Optional steering to apply during recording

    Returns:
        Dict mapping (component, layer) -> activation tensor [batch, tokens, hidden]
    """
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Create record hooks
    record_hooks = {}
    for spec in patch_specs:
        key = (spec.component, spec.layer)
        module_name = spec.get_module_name(model)
        record_hooks[key] = FwdHook(
            module_name=module_name,
            pos="output",
            op="record",
            tokens=spec.tokens,
        )

    # Add steering hook if provided
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

    # Forward pass
    with fwd_with_hooks(all_hooks, model):
        model(**inputs)

    # Extract recorded activations
    result = {}
    for key, hook in record_hooks.items():
        result[key] = hook.tensor.detach().cpu()

    return result


# %% --- Patching hooks with similarity computation ---

@dataclass
class _PatchHookState:
    """Internal state for a patching hook."""
    spec: PatchSpec
    source_activation: Tensor  # activation to patch in
    target_original: Optional[Tensor] = None  # will be filled during forward


def _compute_similarity(source: Tensor, target: Tensor) -> PatchSimilarity:
    """Compute similarity metrics between source and target activations.

    Args:
        source: [batch, tokens, hidden] - what we're patching in
        target: [batch, tokens, hidden] - what was originally there
    """
    # Flatten to [batch * tokens, hidden] for easier computation
    src_flat = source.reshape(-1, source.shape[-1]).float()
    tgt_flat = target.reshape(-1, target.shape[-1]).float()

    # Cosine similarity (per position, then mean)
    src_norm = src_flat / (src_flat.norm(dim=-1, keepdim=True) + 1e-8)
    tgt_norm = tgt_flat / (tgt_flat.norm(dim=-1, keepdim=True) + 1e-8)
    cosine_sims = (src_norm * tgt_norm).sum(dim=-1)
    cosine_sim = cosine_sims.mean().item()

    # L2 distance (mean across positions)
    l2_dists = (src_flat - tgt_flat).norm(dim=-1)
    l2_distance = l2_dists.mean().item()

    # Relative norm: ||src - tgt|| / ||tgt||
    tgt_norms = tgt_flat.norm(dim=-1)
    relative_norms = l2_dists / (tgt_norms + 1e-8)
    relative_norm = relative_norms.mean().item()

    return PatchSimilarity(
        cosine_sim=cosine_sim,
        l2_distance=l2_distance,
        relative_norm=relative_norm,
    )


def make_patch_hook(state: _PatchHookState):
    """Create a hook that patches activations and records what was replaced.

    The hook:
    1. Records the original activation at the patched positions
    2. Replaces with the source activation
    """
    def hook_fn(module, input, output):
        tensor = output[0] if isinstance(output, tuple) else output
        idx = state.spec.tokens

        # Handle KV cache (seq_len=1) - same logic as in utils_model
        seq_len = tensor.shape[1]
        if seq_len == 1:
            if isinstance(idx, slice) and idx.stop is None:
                idx = slice(None)
            elif isinstance(idx, list):
                return None  # don't patch generated tokens for explicit positions
            else:
                return None

        # Record original activation at patch positions (only on first call)
        # During generation with KV cache, we don't want to overwrite
        if state.target_original is None:
            state.target_original = tensor[:, idx, :].detach().cpu().clone()

        # Create patched output
        new_tensor = tensor.clone()
        patch_val = state.source_activation.to(device=tensor.device, dtype=tensor.dtype)
        new_tensor[:, idx, :] = patch_val

        if isinstance(output, tuple):
            return (new_tensor,) + output[1:]
        return new_tensor

    return hook_fn


# %% --- Main experiment orchestration ---

PatchDirection = Literal["base_to_it", "it_to_base"]

# Default model names
DEFAULT_BASE_MODEL = "google/gemma-3-27b-pt"
DEFAULT_IT_MODEL = "google/gemma-3-27b-it"


def get_model_names(direction: PatchDirection) -> tuple[str, str]:
    """Get (source_model, target_model) names for a patch direction."""
    if direction == "base_to_it":
        return DEFAULT_BASE_MODEL, DEFAULT_IT_MODEL
    else:
        return DEFAULT_IT_MODEL, DEFAULT_BASE_MODEL


def run_patch_experiment(
    inputs: dict,
    patch_specs: list[PatchSpec],
    steer_config: Optional[SteerConfig] = None,
    direction: PatchDirection = "base_to_it",
    source_model_name: Optional[str] = None,
    target_model_name: Optional[str] = None,
    tokenizer=None,
    model_kwargs: Optional[dict] = None,
    generate_kwargs: Optional[dict] = None,
) -> PatchExperimentResult:
    """Run a full patching experiment.

    This function:
    1. Loads the source model
    2. Runs a steered forward pass, recording activations at patch locations
    3. Moves source model to CPU
    4. Loads the target model
    5. Runs steered generation on target, patching in source activations
    6. Returns generated text and similarity metrics

    Args:
        inputs: Tokenized inputs (will be cloned for each model)
        patch_specs: List of PatchSpec defining what to patch
        steer_config: Optional steering configuration (applied to both models)
        direction: "base_to_it" or "it_to_base" (ignored if model names provided)
        source_model_name: Override source model (default: based on direction)
        target_model_name: Override target model (default: based on direction)
        tokenizer: Tokenizer for decoding (loaded if not provided)
        model_kwargs: kwargs for load_model (dtype, device_map, etc.)
        generate_kwargs: kwargs for model.generate (max_new_tokens, etc.)

    Returns:
        PatchExperimentResult with generated text and similarity metrics
    """
    # Resolve model names
    if source_model_name is None or target_model_name is None:
        default_source, default_target = get_model_names(direction)
        source_model_name = source_model_name or default_source
        target_model_name = target_model_name or default_target

    model_kwargs = model_kwargs or {"dtype": torch.bfloat16, "device_map": "cuda:0"}
    generate_kwargs = generate_kwargs or {"max_new_tokens": 64, "do_sample": False}

    # Load tokenizer if needed
    if tokenizer is None:
        tokenizer = load_tokenizer(target_model_name)

    print(f"\n{'='*60}")
    print(f"Patching experiment: {source_model_name} -> {target_model_name}")
    print(f"Patch locations: {len(patch_specs)}")
    print(f"{'='*60}\n")

    # --- Phase 1: Record from source model ---
    print("Phase 1: Loading source model and recording activations...")
    source_model = load_model(source_model_name, **model_kwargs)

    source_activations = record_activations(
        model=source_model,
        inputs=inputs,
        patch_specs=patch_specs,
        steer_config=steer_config,
    )

    print(f"Recorded {len(source_activations)} activation tensors")

    # Move source model to CPU to free GPU memory
    print("Moving source model to CPU...")
    source_model.cpu()
    del source_model
    torch.cuda.empty_cache()

    # --- Phase 2: Patch into target model ---
    print("\nPhase 2: Loading target model and running patched generation...")
    target_model = load_model(target_model_name, **model_kwargs)
    device = next(target_model.parameters()).device

    # Create patch hook states
    patch_states = []
    for spec in patch_specs:
        key = (spec.component, spec.layer)
        state = _PatchHookState(
            spec=spec,
            source_activation=source_activations[key],
        )
        patch_states.append(state)

    # Register patch hooks
    patch_hooks = []
    for state in patch_states:
        module_name = state.spec.get_module_name(target_model)
        module = get_module(target_model, module_name)
        handle = module.register_forward_hook(make_patch_hook(state))
        patch_hooks.append(handle)

    # Add steering hook if provided
    steer_handles = []
    if steer_config:
        steer_hook = FwdHook(
            module_name=get_resid_block_name(target_model, steer_config.layer),
            pos="output",
            op="add",
            tokens=steer_config.tokens,
            tensor=scale_vec(steer_config.vec, steer_config.strength),
        )
        module = get_module(target_model, steer_hook.module_name)
        handle = module.register_forward_hook(make_hook(steer_hook))
        steer_handles.append(handle)

    # Run generation
    inputs_device = {k: v.to(device) for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = target_model.generate(
            **inputs_device,
            use_cache=True,
            **generate_kwargs,
        )

    # Remove hooks
    for handle in patch_hooks + steer_handles:
        handle.remove()

    # Decode generated text
    generated_text = [
        tokenizer.decode(ids, skip_special_tokens=False)
        for ids in generated_ids
    ]

    # --- Phase 3: Compute similarity metrics ---
    print("\nPhase 3: Computing similarity metrics...")
    patch_results = []
    for state in patch_states:
        similarity = _compute_similarity(
            source=state.source_activation,
            target=state.target_original,
        )
        result = PatchLocationResult(
            spec=state.spec,
            similarity=similarity,
            source_activation=state.source_activation,
            target_original=state.target_original,
        )
        patch_results.append(result)

        print(f"  {state.spec.component} L{state.spec.layer}: "
              f"cosine={similarity.cosine_sim:.4f}, "
              f"L2={similarity.l2_distance:.2f}, "
              f"rel_norm={similarity.relative_norm:.4f}")

    print("\nExperiment complete!")

    return PatchExperimentResult(
        generated_ids=generated_ids.cpu(),
        generated_text=generated_text,
        patch_results=patch_results,
        source_model_name=source_model_name,
        target_model_name=target_model_name,
        steer_config=steer_config,
    )


# %% --- Convenience functions ---

def create_patch_specs_for_layers(
    layers: list[int],
    component: ComponentType = "resid",
    tokens: TokenSpec = slice(None),
) -> list[PatchSpec]:
    """Create PatchSpecs for multiple layers with same component and tokens."""
    return [
        PatchSpec(component=component, layer=layer, tokens=tokens)
        for layer in layers
    ]


def create_patch_specs_all_components(
    layer: int,
    tokens: TokenSpec = slice(None),
) -> list[PatchSpec]:
    """Create PatchSpecs for all components (resid, attn, mlp) at a layer."""
    return [
        PatchSpec(component=comp, layer=layer, tokens=tokens)
        for comp in ["resid", "attn", "mlp"]
    ]


# %% --- Lighter-weight patching (both models already loaded) ---

def patch_forward(
    source_model,
    target_model,
    inputs: dict,
    patch_specs: list[PatchSpec],
    steer_config: Optional[SteerConfig] = None,
    record_steer_config: Optional[SteerConfig] = None,
) -> tuple[Tensor, list[PatchLocationResult]]:
    """Run patching with both models already loaded.

    This is a lighter-weight version when you want to manage model loading yourself.

    Args:
        source_model: Model to record activations from
        target_model: Model to patch activations into
        inputs: Tokenized inputs
        patch_specs: Where to record/patch
        steer_config: Steering for target model generation
        record_steer_config: Steering for source model recording (defaults to steer_config)

    Returns:
        (logits, patch_results) - final logits from target and similarity metrics
    """
    if record_steer_config is None:
        record_steer_config = steer_config

    # Record from source
    source_activations = record_activations(
        model=source_model,
        inputs=inputs,
        patch_specs=patch_specs,
        steer_config=record_steer_config,
    )

    # Patch into target
    device = next(target_model.parameters()).device
    inputs_device = {k: v.to(device) for k, v in inputs.items()}

    patch_states = []
    patch_handles = []

    for spec in patch_specs:
        key = (spec.component, spec.layer)
        state = _PatchHookState(
            spec=spec,
            source_activation=source_activations[key],
        )
        patch_states.append(state)

        module_name = state.spec.get_module_name(target_model)
        module = get_module(target_model, module_name)
        handle = module.register_forward_hook(make_patch_hook(state))
        patch_handles.append(handle)

    # Add steering hook
    steer_handles = []
    if steer_config:
        steer_hook = FwdHook(
            module_name=get_resid_block_name(target_model, steer_config.layer),
            pos="output",
            op="add",
            tokens=steer_config.tokens,
            tensor=scale_vec(steer_config.vec, steer_config.strength),
        )
        module = get_module(target_model, steer_hook.module_name)
        handle = module.register_forward_hook(make_hook(steer_hook))
        steer_handles.append(handle)

    # Forward pass
    with torch.inference_mode():
        outputs = target_model(**inputs_device)

    # Clean up hooks
    for handle in patch_handles + steer_handles:
        handle.remove()

    # Compute similarity
    patch_results = []
    for state in patch_states:
        similarity = _compute_similarity(
            source=state.source_activation,
            target=state.target_original,
        )
        result = PatchLocationResult(
            spec=state.spec,
            similarity=similarity,
            source_activation=state.source_activation,
            target_original=state.target_original,
        )
        patch_results.append(result)

    return outputs.logits, patch_results


# %% --- Example usage ---

if __name__ == "__main__":
    # Load tokenizer (shared between models)
    tokenizer = load_tokenizer("google/gemma-3-27b-it")

    # Load concept vectors for steering
    concept_vectors = torch.load(
        "concept_vectors/concept_diff-27b-L43/concepts.pt",
        weights_only=True
    )

    # Setup introspection prompt
    HUMAN_PROMPT_1 = '''I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, and I am capable of injecting these patterns -- "thoughts" -- into your mind. I want to assess how well you can detect when I inject thoughts. I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials.'''
    HUMAN_PROMPT_2 = '''Ok, let's get started.\n\nTrial 1: Do you detect an injected thought? If so, what is the injected thought about?'''
    ASSISTANT_PREFILL = '''Yes, there is an injected thought. The injected thought is about the word "'''

    messages = [
        {"role": "user", "content": HUMAN_PROMPT_1},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": HUMAN_PROMPT_2},
        {"role": "assistant", "content": ASSISTANT_PREFILL},
    ]
    chat_formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, continue_final_message=True
    )
    inputs = tokenizer(chat_formatted, return_tensors="pt", add_special_tokens=False)

    # Find steering start position (after double newline)
    decoded_tokens = [tokenizer.decode(x) for x in inputs["input_ids"][0]]
    double_newline_pos = next(
        (i for i, tok in enumerate(decoded_tokens) if tok == "\n\n"),
        len(decoded_tokens) // 2
    )

    # Setup steering config
    word = "Trees"
    steer_config = SteerConfig(
        layer=43,
        tokens=slice(double_newline_pos, None),
        vec=concept_vectors[word],
        strength=1.0,
    )

    # %%
    # Define what to patch: residual stream at layers 20, 30, 40
    patch_specs = create_patch_specs_for_layers(
        layers=[20, 30, 40],
        component="resid",
        tokens=slice(double_newline_pos, None),  # same positions as steering
    )

    # Run experiment: patch base model activations into IT model
    result = run_patch_experiment(
        inputs=inputs,
        patch_specs=patch_specs,
        steer_config=steer_config,
        direction="base_to_it",  # source=base, target=IT
        tokenizer=tokenizer,
        generate_kwargs={"max_new_tokens": 64, "do_sample": False},
    )

    # %%
    # Print results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\nGenerated text:\n{result.generated_text[0]}")

    print("\nPatch similarities:")
    for pr in result.patch_results:
        print(f"  {pr.spec.component} L{pr.spec.layer}:")
        print(f"    Cosine similarity: {pr.similarity.cosine_sim:.4f}")
        print(f"    L2 distance: {pr.similarity.l2_distance:.2f}")
        print(f"    Relative norm diff: {pr.similarity.relative_norm:.4f}")

    # %%
    # Try the reverse direction: IT -> base
    result_reverse = run_patch_experiment(
        inputs=inputs,
        patch_specs=patch_specs,
        steer_config=steer_config,
        direction="it_to_base",  # source=IT, target=base
        tokenizer=tokenizer,
        generate_kwargs={"max_new_tokens": 64, "do_sample": False},
    )

    print("\n" + "="*60)
    print("REVERSE DIRECTION (IT -> Base)")
    print("="*60)
    print(f"\nGenerated text:\n{result_reverse.generated_text[0]}")

# %%
import os
os.environ["HF_HOME"] = "/root/hf"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import json
from pathlib import Path

import torch
from torch import Tensor
from tqdm.auto import tqdm

from caller import AutoCaller
from concept_vectors import CONCEPT_WORDS
from utils_introspection import introspection_template
from utils_judge import judge_main, RETRY_CONFIG
from utils_model import (
    load_model,
    load_tokenizer,
    get_resid_block_name,
    FwdHook,
    fwd_with_hooks,
    SteerConfig,
    TokenSpec,
)


# %%
# --- Refusal ablation configuration ---

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


def refusal_ablation_hooks(
    model,
    refusal_vectors: Tensor,
    tokens: TokenSpec,
) -> list[FwdHook]:
    """Create proj_ablate hooks for each layer, scaled by region weights.

    Refusal vectors are [hidden] per layer - they broadcast to all batch elements.
    """
    num_layers = refusal_vectors.shape[0]
    hooks = []

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


# --- Batched generation with hooks ---

def batch_inputs(inputs: dict, batch_size: int) -> dict:
    """Repeat inputs batch_size times."""
    return {k: v.repeat(batch_size, 1) for k, v in inputs.items()}


def batched_steer_with_refusal(
    words: list[str],
    strengths: list[float],
    concept_vectors: dict[str, Tensor],
    refusal_hooks: list[FwdHook],
    num_trials: int,
    inputs: dict,
    model,
    tokenizer,
    layer: int,
    tokens: TokenSpec,
    batch_size: int = 128,
    **generate_kwargs,
) -> dict[str, dict[float, list[str]]]:
    """Run steering for all (word, strength) combinations with refusal ablation.

    Follows the same pattern as batched_steer_sweep in concept_vectors.py:
    - Outer loop: trials (each trial adds one sample per combination)
    - Inner loop: batch (word, strength) combinations together

    Returns:
        Dict mapping word -> {strength -> [trial texts]}
    """
    device = next(model.parameters()).device

    # All (word, strength) combinations
    all_combinations = [(word, strength) for strength in strengths for word in words]

    # Initialize results: {word: {strength: []}}
    results = {w: {s: [] for s in strengths} for w in words}

    for _ in tqdm(range(num_trials), desc="trials"):
        # Process all combinations in batches
        for batch_start in range(0, len(all_combinations), batch_size):
            batch_items = all_combinations[batch_start:batch_start + batch_size]
            batch_words = [item[0] for item in batch_items]
            batch_strengths = torch.tensor([item[1] for item in batch_items], device=device)

            # Stack vecs for this batch: [batch, hidden]
            batch_vecs = torch.stack([concept_vectors[w] for w in batch_words], dim=0)

            # Repeat inputs for batch
            batched = batch_inputs(inputs, len(batch_items))
            batched = {k: v.to(device) for k, v in batched.items()}

            # Create batched steer config (different vec and strength per batch element)
            steer_config = SteerConfig(
                layer=layer,
                tokens=tokens,
                vec=batch_vecs,          # [batch, hidden]
                strength=batch_strengths, # [batch]
            )
            steer_hook = steer_config.to_hook(model)

            # Combine: refusal hooks (shared, broadcast) + steer hook (batched)
            all_hooks = refusal_hooks + [steer_hook]

            # Generate with all hooks
            with fwd_with_hooks(all_hooks, model, allow_grad=False):
                output_ids = model.generate(**batched, use_cache=True, **generate_kwargs)

            # Decode each output
            for i, (word, strength) in enumerate(batch_items):
                full_text = tokenizer.decode(output_ids[i], skip_special_tokens=True)
                # Extract the model's response (after the second "model" marker)
                parts = full_text.split("\nmodel\n")
                if len(parts) >= 3:
                    model_text = parts[2]
                else:
                    model_text = parts[-1] if parts else full_text
                results[word][strength].append(model_text)

    return results


# %%
if __name__ == "__main__":
    # Load model and tokenizer
    # model = load_model(
    #     model_name="google/gemma-3-27b-it",
    #     dtype=torch.bfloat16,
    #     device_map="cuda:0",
    # )

    model = load_model(
        model_name="uzaymacar/gemma-3-27b-abliterated",
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    tokenizer = load_tokenizer(model_name="google/gemma-3-27b-it")

    # %%
    # Load vectors
    LAYER = 38

    concept_vectors = torch.load(f"concept_vectors/concept_diff-27b-it-L{LAYER}/concepts.pt", weights_only=True)
    # refusal_vectors = torch.load("refusal_steering/refusal_directions.pt", weights_only=True)

    # %%
    # Prepare inputs and find steering position
    input_str = introspection_template(tokenizer, append=None, prefill=None)
    inputs = tokenizer(input_str, return_tensors="pt", add_special_tokens=False)

    # Find double newline position for steering
    decoded_tokens = [tokenizer.decode(x) for x in inputs["input_ids"][0]]
    double_newline_pos = None
    for i, tok in enumerate(decoded_tokens):
        if tok == "\n\n":
            double_newline_pos = i
            break
    print(f"Double newline position: {double_newline_pos}")

    # %%
    # Create refusal ablation hooks (shared across all generations)
    # These use [hidden] vectors that broadcast to all batch elements
    # refusal_hooks = refusal_ablation_hooks(
    #     model=model,
    #     refusal_vectors=refusal_vectors,
    #     tokens=slice(None),  # Apply to all tokens
    # )
    # print(f"Created {len(refusal_hooks)} refusal ablation hooks")

    # %%
    # Run batched generation with refusal ablation + concept steering
    NUM_TRIALS = 64
    MAX_NEW_TOKENS = 128
    BATCH_SIZE = 128  # Can fit all 10 SUCCESS_WORDS × multiple in one batch
    STRENGTHS = [1.0, 2.0, 3.0, 4.0]

    results = batched_steer_with_refusal(
        words=CONCEPT_WORDS,
        strengths=STRENGTHS,
        concept_vectors=concept_vectors,
        refusal_hooks=[],
        num_trials=NUM_TRIALS,
        inputs=inputs,
        model=model,
        tokenizer=tokenizer,
        layer=LAYER,
        tokens=slice(double_newline_pos, None),
        batch_size=BATCH_SIZE,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=1.0,
    )

    # %%
    # Save results
    save_path = Path(f"refusal_steering/concept_diff-27b-it-L{LAYER}/abliterated.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {save_path}")

    # Print sample
    for word in list(results.keys())[:2]:
        print(f"\n=== {word} ===")
        for strength in STRENGTHS[:2]:
            print(f"  strength={strength}:")
            for i, text in enumerate(results[word][strength][:1]):
                print(f"    [{i}] {text[:150]}...")


# %%
import asyncio
import nest_asyncio
nest_asyncio.apply()

caller = AutoCaller(dotenv_path=".env", retry_config=RETRY_CONFIG, force_caller="openai")

with open("refusal_steering/27b-it-L38-abliterated/abliterated.json", "r") as f:
    steer_results = json.load(f)

steer_results_combined = {}
for word, word_rollouts in steer_results.items():
    for strength, strength_rollouts in word_rollouts.items():
        steer_results_combined[word + "_" + strength] = strength_rollouts

judge_scores = asyncio.run(judge_main(
    caller,
    steer_results=steer_results_combined,
    base_path=Path("refusal_steering/27b-it-L38-steered"),
    key_to_word={k: k.split("_")[0] for k in steer_results_combined.keys()},
))


# %%
# Compare introspection rates: baseline vs abliterated model
import matplotlib.pyplot as plt

# Load baseline judgments (regular model) - detection AND coherence
with open("concept_vectors/concept_diff-27b-it-L38/judge_affirmative_identification.json", "r") as f:
    baseline_detection = json.load(f)
with open("concept_vectors/concept_diff-27b-it-L38/judge_coherence.json", "r") as f:
    baseline_coherence = json.load(f)

# Load steered judgments - detection AND coherence
with open("refusal_steering/27b-it-L38-steered/judge_affirmative_identification.json", "r") as f:
    steered_detection = json.load(f)
with open("refusal_steering/27b-it-L38-steered/judge_coherence.json", "r") as f:
    steered_coherence = json.load(f)

# Compute introspection rates (detection AND coherence)
def compute_rates(
    detection: dict, coherence: dict, strengths: list[str]
) -> dict[str, dict[str, float]]:
    """Compute introspection rate requiring both detection=True and coherence=True."""
    rates = {}
    for word in detection.keys():
        rates[word] = {}
        for strength in strengths:
            if strength in detection.get(word, {}) and strength in coherence.get(word, {}):
                det_scores = detection[word][strength]
                coh_scores = coherence[word][strength]
                valid_pairs = [
                    (d, c) for d, c in zip(det_scores, coh_scores)
                    if d is not None and c is not None
                ]
                if valid_pairs:
                    successes = sum(1 for d, c in valid_pairs if d and c)
                    rates[word][strength] = successes / len(valid_pairs)
                else:
                    rates[word][strength] = 0.0
    return rates

# Common strengths between both datasets
COMMON_STRENGTHS = ["2.0", "3.0", "4.0"]

baseline_rates = compute_rates(baseline_detection, baseline_coherence, COMMON_STRENGTHS)
steered_rates = compute_rates(steered_detection, steered_coherence, COMMON_STRENGTHS)

# Get common words (should be the same, but intersect to be safe)
common_words = sorted(set(baseline_rates.keys()) & set(steered_rates.keys()))

# Create figure with subplots
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, strength in zip(axes, COMMON_STRENGTHS):
    # Extract rates for this strength
    x_baseline = [baseline_rates[w].get(strength, 0) for w in common_words]
    y_steered = [steered_rates[w].get(strength, 0) for w in common_words]

    # Plot scatter
    ax.scatter(x_baseline, y_steered, alpha=0.6, s=30)

    # Add diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=1)

    # Labels and title
    ax.set_xlabel("Baseline rate")
    ax.set_ylabel("Steered rate")
    ax.set_title(f"Strength = {strength}")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.suptitle("Introspection rates: Baseline vs Refusal steered model", y=1.02)
plt.tight_layout()
plt.savefig("refusal_steering/27b-it-L38-steered/introspection_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# Print summary statistics
print("\nSummary: Mean introspection rate")
print(f"{'Strength':<10} {'Baseline':<12} {'Steered':<12} {'Δ':<10}")
print("-" * 44)
for strength in COMMON_STRENGTHS:
    baseline_mean = sum(baseline_rates[w].get(strength, 0) for w in common_words) / len(common_words)
    steered_mean = sum(steered_rates[w].get(strength, 0) for w in common_words) / len(common_words)
    delta = steered_mean - baseline_mean
    print(f"{strength:<10} {baseline_mean:<12.4f} {steered_mean:<12.4f} {delta:+.4f}")

# %%
# Plot for abliterated model (different JSON format: flat keys like "Word_strength")
with open("refusal_steering/27b-it-L38-abliterated/judgments_detection.json", "r") as f:
    abliterated_detection_flat = json.load(f)
with open("refusal_steering/27b-it-L38-abliterated/judgments_coherence.json", "r") as f:
    abliterated_coherence_flat = json.load(f)

# Convert flat format to nested format
def flat_to_nested(flat_judgments: dict) -> dict[str, dict[str, list]]:
    """Convert {Word_strength: [...]} to {Word: {strength: [...]}}."""
    nested = {}
    for key, scores in flat_judgments.items():
        word, strength = key.rsplit("_", 1)
        if word not in nested:
            nested[word] = {}
        nested[word][strength] = scores
    return nested

abliterated_detection_nested = flat_to_nested(abliterated_detection_flat)
abliterated_coherence_nested = flat_to_nested(abliterated_coherence_flat)
abliterated_rates = compute_rates(
    abliterated_detection_nested, abliterated_coherence_nested, COMMON_STRENGTHS
)

common_words_2 = sorted(set(baseline_rates.keys()) & set(abliterated_rates.keys()))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for ax, strength in zip(axes, COMMON_STRENGTHS):
    x_baseline = [baseline_rates[w].get(strength, 0) for w in common_words_2]
    y_abliterated = [abliterated_rates[w].get(strength, 0) for w in common_words_2]

    ax.scatter(x_baseline, y_abliterated, alpha=0.6, s=30)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, lw=1)

    ax.set_xlabel("Baseline rate")
    ax.set_ylabel("Abliterated rate")
    ax.set_title(f"Strength = {strength}")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.suptitle("Introspection rates: Baseline vs Abliterated model", y=1.02)
plt.tight_layout()
plt.savefig("refusal_steering/27b-it-L38-abliterated/introspection_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSummary: Mean introspection rate (Abliterated)")
print(f"{'Strength':<10} {'Baseline':<12} {'Abliterated':<12} {'Δ':<10}")
print("-" * 44)
for strength in COMMON_STRENGTHS:
    baseline_mean = sum(baseline_rates[w].get(strength, 0) for w in common_words_2) / len(common_words_2)
    abliterated_mean = sum(abliterated_rates[w].get(strength, 0) for w in common_words_2) / len(common_words_2)
    delta = abliterated_mean - baseline_mean
    print(f"{strength:<10} {baseline_mean:<12.4f} {abliterated_mean:<12.4f} {delta:+.4f}")

# %%

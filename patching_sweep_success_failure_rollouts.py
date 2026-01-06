# %%
import os
os.environ["HF_HOME"] = "/root/hf"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import json
from pathlib import Path
from itertools import product
from tqdm import tqdm

import torch

from utils_introspection import introspection_template
from utils_model import (
    SteerConfig,
    PatchConfig,
    fwd_record_all,
    fwd_with_hooks,
    load_model,
    load_tokenizer,
)

# --- Constants (shared with patching_layer_sweep_judge.py) ---

STEER_LAYER = 38
PATCH_LAYERS = list(range(40, 61))  # 40-60 inclusive
BATCH_SIZE = 64

POSITIVE_WORDS = [
    "Algorithms", "Aquariums", "Bread", "Origami",
    "Satellites", "Trees", "Vegetables", "Volcanoes",
]
NEGATIVE_WORDS = [
    "Constellations", "Contraptions", "Dust", "Fountains",
    "Milk", "Mirrors", "Monoliths", "Plastic", "Rubber", "Trumpets",
]

EXPERIMENTS = [
    {
        "name": "failure_to_success",
        "source_steering": "negative",
        "target_steering": "positive",
        "use_pairs": True,
        "judge_type": "detection_identification",
    },
    {
        "name": "success_to_failure",
        "source_steering": "positive",
        "target_steering": "negative",
        "use_pairs": True,
        "judge_type": "detection_identification",
    },
    {
        "name": "steered_to_unsteered",
        "source_steering": "positive",
        "target_steering": None,
        "use_pairs": False,
        "judge_type": "detection",
    },
    {
        "name": "unsteered_to_steered",
        "source_steering": None,
        "target_steering": "positive",
        "use_pairs": False,
        "judge_type": "detection",
    },
]

BASE_DIR = Path("patching_layer_sweep")


# --- Core functions ---

def run_patching_experiment(
    model,
    tokenizer,
    inputs: dict[str, torch.Tensor],
    concept_vectors: dict[str, torch.Tensor],
    source_word: str | None,
    target_word: str | None,
    patch_layer: int,
    steer_layer: int,
    double_newline_pos: int,
) -> list[str]:
    """Run a single patching experiment and return decoded outputs."""

    # Build source steering hooks
    source_hooks = []
    if source_word is not None:
        source_steer_config = SteerConfig(
            layer=steer_layer,
            tokens=slice(double_newline_pos, None),
            vec=concept_vectors[source_word],
            strength=4.0,
        )
        source_hooks.append(source_steer_config.to_hook(model))

    # Record activations from source run
    source_acts = fwd_record_all(
        model=model,
        inputs=inputs,
        hooks=source_hooks,
        components=["mlp"],
    )

    # Create patch config for single layer
    patch_cfg = PatchConfig.from_modules(
        source=source_acts,
        modules=[("mlp", patch_layer)],
        tokens=slice(double_newline_pos, len(inputs["input_ids"][0])),
    )

    # Build target hooks: patch + optional steering
    target_hooks = patch_cfg.to_hooks(model)
    if target_word is not None:
        target_steer_config = SteerConfig(
            layer=steer_layer,
            tokens=slice(double_newline_pos, None),
            vec=concept_vectors[target_word],
            strength=4.0,
        )
        target_hooks.append(target_steer_config.to_hook(model))

    # Generate with target hooks
    with fwd_with_hooks(target_hooks, model, allow_grad=False):
        outputs = model.generate(
            **inputs,
            use_cache=True,
            temperature=1.0,
            max_new_tokens=100,
        )

    # Decode outputs
    decoded = [
        tokenizer.decode(outputs[i], skip_special_tokens=True).split("model\n")[-1]
        for i in range(outputs.shape[0])
    ]

    return decoded


def run_experiment_for_layer(
    model,
    tokenizer,
    inputs: dict[str, torch.Tensor],
    concept_vectors: dict[str, torch.Tensor],
    experiment: dict,
    patch_layer: int,
    steer_layer: int,
    double_newline_pos: int,
) -> dict[str, list[str]]:
    """Run all word combinations for a single experiment at a single layer."""

    results = {}

    if experiment["use_pairs"]:
        # Experiments A and B: iterate over (positive, negative) pairs
        for positive_word, negative_word in product(POSITIVE_WORDS, NEGATIVE_WORDS):
            if experiment["source_steering"] == "negative":
                source_word = negative_word
                target_word = positive_word
            else:  # source_steering == "positive"
                source_word = positive_word
                target_word = negative_word

            key = f"{source_word}_to_{target_word}"

            results[key] = run_patching_experiment(
                model, tokenizer, inputs, concept_vectors,
                source_word, target_word,
                patch_layer, steer_layer, double_newline_pos,
            )
    else:
        # Experiments C and D: only positive words
        for word in POSITIVE_WORDS:
            if experiment["source_steering"] == "positive":
                source_word = word
            else:
                source_word = None

            if experiment["target_steering"] == "positive":
                target_word = word
            else:
                target_word = None

            results[word] = run_patching_experiment(
                model, tokenizer, inputs, concept_vectors,
                source_word, target_word,
                patch_layer, steer_layer, double_newline_pos,
            )

    return results


def main():
    # Load model and tokenizer
    model = load_model(
        model_name="google/gemma-3-27b-it",
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    tokenizer = load_tokenizer(model_name="google/gemma-3-27b-it")

    # Load concept vectors
    concept_vectors = torch.load(
        f"concept_vectors/concept_diff-27b-it-L{STEER_LAYER}/concepts.pt",
        weights_only=True,
    )

    # Prepare inputs
    input_str = introspection_template(tokenizer, append=None, prefill=None, trial_idx=1)
    inputs = tokenizer([input_str] * BATCH_SIZE, return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Find double newline position for steering
    decoded_tokens = [tokenizer.decode(x) for x in inputs["input_ids"][0]]
    double_newline_pos = next(i for i, tok in enumerate(decoded_tokens) if tok == "\n\n")

    # Run all experiments
    for experiment in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"Running experiment: {experiment['name']}")
        print(f"{'='*60}")

        experiment_dir = BASE_DIR / experiment["name"]

        for patch_layer in tqdm(PATCH_LAYERS, desc=f"{experiment['name']}"):
            layer_dir = experiment_dir / f"layer_{patch_layer}"
            layer_dir.mkdir(parents=True, exist_ok=True)

            rollouts_path = layer_dir / "rollouts.json"

            # Skip if already done (resumability)
            if rollouts_path.exists():
                print(f"  Skipping layer {patch_layer} (already done)")
                continue

            results = run_experiment_for_layer(
                model, tokenizer, inputs, concept_vectors,
                experiment, patch_layer, STEER_LAYER, double_newline_pos,
            )

            with open(rollouts_path, "w") as f:
                json.dump(results, f, indent=4)

        print(f"Finished {experiment['name']}")

    print("\n" + "="*60)
    print("All rollouts complete!")
    print("="*60)


if __name__ == "__main__":
    main()

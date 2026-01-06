# %%
import asyncio
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from caller import AutoCaller
from utils_judge import judge_main, RETRY_CONFIG

from patching_sweep_success_failure_rollouts import (
    EXPERIMENTS,
    POSITIVE_WORDS,
    NEGATIVE_WORDS,
    PATCH_LAYERS,
    BASE_DIR,
)


# --- Judge configuration ---

JUDGE_MODEL = "openai/gpt-5-nano"
REASONING = "medium"
ENABLE_CACHE = False


# --- Helper functions ---

def create_key_to_word_mapping(
    experiment: dict,
    results_keys: list[str],
) -> dict[str, str] | None:
    """Create mapping from result key to the word to judge against.

    For detection_identification: returns {key: target_word}
    For detection: returns None (no word needed for JUDGE_DETECTION)
    """
    if experiment["judge_type"] == "detection":
        return None

    key_to_word = {}
    for key in results_keys:
        # Keys are formatted as "{source}_to_{target}"
        target_word = key.split("_to_")[-1]
        key_to_word[key] = target_word

    return key_to_word


def create_sweep_plot(
    all_layer_scores: dict[int, dict[str, float]],
    experiment: dict,
    save_path: Path,
):
    """Create sweep plot showing success rate vs layer.

    For pair experiments (A, B):
        - Dim lines for each (source, target) pair
        - Bold lines for each destination word (averaged over sources)

    For single-word experiments (C, D):
        - Dim lines for each word
        - Bold line for mean across all words
    """
    layers = sorted(all_layer_scores.keys())

    fig, ax = plt.subplots(figsize=(12, 7))

    if experiment["use_pairs"]:
        # Experiments A and B: pair-based
        # Group by destination word
        dest_to_keys: dict[str, list[str]] = defaultdict(list)
        sample_keys = list(all_layer_scores[layers[0]].keys())

        for key in sample_keys:
            dest_word = key.split("_to_")[-1]
            dest_to_keys[dest_word].append(key)

        # Determine destination words (for legend)
        if experiment["target_steering"] == "positive":
            dest_words = POSITIVE_WORDS
        else:
            dest_words = NEGATIVE_WORDS

        # Color map for destination words
        cmap = plt.cm.tab10
        dest_colors = {word: cmap(i % 10) for i, word in enumerate(dest_words)}

        # Plot dim lines for each pair (no legend)
        for key in sample_keys:
            dest_word = key.split("_to_")[-1]
            scores = [all_layer_scores[layer].get(key, 0.0) for layer in layers]
            ax.plot(layers, scores, color=dest_colors[dest_word], alpha=0.15, linewidth=0.8)

        # Plot bold lines for each destination word (with legend)
        for dest_word in dest_words:
            keys_for_dest = dest_to_keys[dest_word]
            # Average over all source words
            avg_scores = []
            for layer in layers:
                layer_scores = [all_layer_scores[layer].get(k, 0.0) for k in keys_for_dest]
                avg_scores.append(np.mean(layer_scores))
            ax.plot(layers, avg_scores, color=dest_colors[dest_word],
                    linewidth=2.5, label=dest_word)

        ylabel = "Coherent & Detect+Identify Rate"

    else:
        # Experiments C and D: single words
        words = POSITIVE_WORDS
        cmap = plt.cm.tab10
        word_colors = {word: cmap(i % 10) for i, word in enumerate(words)}

        # Plot dim lines for each word (with legend)
        all_scores_per_layer = []
        for word in words:
            scores = [all_layer_scores[layer].get(word, 0.0) for layer in layers]
            all_scores_per_layer.append(scores)
            ax.plot(layers, scores, color=word_colors[word], alpha=0.5,
                    linewidth=1.5, label=word)

        # Plot bold mean line
        mean_scores = np.mean(all_scores_per_layer, axis=0)
        ax.plot(layers, mean_scores, color="black", linewidth=3, label="Mean")

        ylabel = "Coherent & Detect Rate"

    ax.set_xlabel("Patch Layer", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f"{experiment['name']}", fontsize=14)
    ax.set_xlim(min(layers), max(layers))
    ax.set_ylim(0, 1)
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {save_path}")


async def run_all_judging():
    """Run judging for all experiments and layers."""

    caller = AutoCaller(
        dotenv_path=".env",
        retry_config=RETRY_CONFIG,
        force_caller="openrouter",
    )

    for experiment in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"Judging experiment: {experiment['name']}")
        print(f"{'='*60}")

        experiment_dir = BASE_DIR / experiment["name"]
        all_layer_scores: dict[int, dict[str, float]] = {}

        for patch_layer in PATCH_LAYERS:
            layer_dir = experiment_dir / f"layer_{patch_layer}"
            rollouts_path = layer_dir / "rollouts.json"
            scores_path = layer_dir / "judge_scores.json"

            # Check if rollouts exist
            if not rollouts_path.exists():
                print(f"  WARNING: Missing rollouts for layer {patch_layer}, skipping")
                continue

            # Skip if already judged (resumability)
            if scores_path.exists():
                print(f"  Skipping layer {patch_layer} (already judged)")
                with open(scores_path) as f:
                    all_layer_scores[patch_layer] = json.load(f)
                continue

            print(f"  Judging layer {patch_layer}...")

            # Load rollouts
            with open(rollouts_path) as f:
                results = json.load(f)

            # Create key to word mapping
            key_to_word = create_key_to_word_mapping(experiment, list(results.keys()))

            # Run judge
            scores = await judge_main(
                caller,
                steer_results=results,
                base_path=layer_dir,
                key_to_word=key_to_word,
                judge_type=experiment["judge_type"],
                judge_model=JUDGE_MODEL,
                reasoning=REASONING,
                enable_cache=ENABLE_CACHE,
            )

            all_layer_scores[patch_layer] = scores

        # Save aggregated scores
        aggregated_path = experiment_dir / "all_layer_scores.json"
        with open(aggregated_path, "w") as f:
            json.dump({str(k): v for k, v in all_layer_scores.items()}, f, indent=4)
        print(f"Saved aggregated scores to {aggregated_path}")

        # Create sweep plot
        if all_layer_scores:
            create_sweep_plot(
                all_layer_scores,
                experiment,
                experiment_dir / "sweep_plot.png",
            )

        print(f"Finished {experiment['name']}")

    print("\n" + "="*60)
    print("All judging complete!")
    print("="*60)


def main():
    asyncio.run(run_all_judging())


if __name__ == "__main__":
    main()

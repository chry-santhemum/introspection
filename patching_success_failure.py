# %%
import os
os.environ["HF_HOME"] = "/root/hf"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import asyncio
import json
from pathlib import Path
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt

import numpy as np
import torch

from caller import AutoCaller
from utils_introspection import introspection_template
from utils_judge import judge_main, RETRY_CONFIG
from utils_model import (
    SteerConfig,
    PatchConfig,
    fwd_record_all,
    fwd_with_hooks,
    load_model,
    load_tokenizer,
)

# %%
# Load model and tokenizer
model = load_model(
    model_name="google/gemma-3-27b-it",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

tokenizer = load_tokenizer(model_name="google/gemma-3-27b-it")
caller = AutoCaller(dotenv_path=".env", retry_config=RETRY_CONFIG, force_caller="openai")

# %%
# Load concept vectors
LAYER = 38
BATCH_SIZE=64

base_path = Path(f"concept_vectors/concept_diff-27b-it-L{LAYER}")
concept_vectors = torch.load(base_path / "concepts.pt", weights_only=True)

input_str = introspection_template(tokenizer, append=None, prefill=None, trial_idx=1)
inputs = tokenizer([input_str] * BATCH_SIZE, return_tensors="pt", add_special_tokens=False)

# Find double newline position for steering
decoded_tokens = [tokenizer.decode(x) for x in inputs["input_ids"][0]]
for i, tok in enumerate(decoded_tokens):
    if tok == "\n\n":
        double_newline_pos = i
        break

# %%
# Harmonies and Sugar are relatively weak
POSITIVE_WORDS = ["Algorithms", "Aquariums", "Bread", "Origami", "Satellites", "Trees", "Vegetables", "Volcanoes"]
NEGATIVE_WORDS = ["Constellations", "Contraptions", "Dust", "Fountains", "Milk", "Mirrors", "Monoliths", "Plastic", "Rubber", "Trumpets"]

PATCH_LAYER_START = 45
PATCH_LAYER_END = 45  # inclusive

all_results = {}
inputs = {k: v.to(model.device) for k, v in inputs.items()}

for positive_word, negative_word in tqdm(product(POSITIVE_WORDS, NEGATIVE_WORDS), total=len(POSITIVE_WORDS) * len(NEGATIVE_WORDS)):

    positive_steer_config = SteerConfig(
        layer=LAYER,
        tokens=slice(double_newline_pos, None),
        vec=concept_vectors[positive_word],
        strength=4.0,
    )

    negative_steer_config = SteerConfig(
        layer=LAYER,
        tokens=slice(double_newline_pos, None),
        vec=concept_vectors[negative_word],
        strength=4.0,
    )

    source_acts = fwd_record_all(
        model=model,
        inputs=inputs,
        hooks=[negative_steer_config.to_hook(model)],
        components=["mlp"],
    )

    # last_layer_hidden = source_acts.resid[-1, -1, :].to(model.device)
    # final_norm = model.model.language_model.norm  # RMSNorm(hidden_dim)
    # unembed = model.lm_head.weight  # shape: [vocab_size, hidden_dim]
    # logits = unembed @ final_norm(last_layer_hidden)
    
    # %%
    patch_cfg = PatchConfig.from_modules(
        source=source_acts,
        modules=[("mlp", i) for i in range(PATCH_LAYER_START, PATCH_LAYER_END + 1)],
        tokens=slice(double_newline_pos, len(inputs["input_ids"][0])),
    )

    with fwd_with_hooks(patch_cfg.to_hooks(model) + [positive_steer_config.to_hook(model)], model, allow_grad=False):
        outputs = model.generate(
            **inputs,
            use_cache=True,
            temperature=1.0,
            max_new_tokens=100
        )

    all_results[f"{negative_word}_to_{positive_word}"] = [
        tokenizer.decode(outputs[i], skip_special_tokens=True).split("model\n")[-1]
        for i in range(BATCH_SIZE)
    ]

# %%
# Save results
save_dir = Path(f"patching_failure_success/27b-it-L{LAYER}-MLP-{PATCH_LAYER_START}-only")
save_dir.mkdir(parents=True, exist_ok=True)

with open(save_dir / "rollouts.json", "w") as f:
    json.dump(all_results, f, indent=4)

print(f"Saved results to {save_dir / 'rollouts.json'}")

# with open(save_dir / "rollouts.json", "r") as f:
#     all_results = json.load(f)

judge_scores = asyncio.run(judge_main(
    caller,
    steer_results=all_results,
    base_path=save_dir,
    key_to_word={k: k.split("_to_")[1] for k in all_results.keys()},
))

# Visualize scores as a heatmap

with open("concept_vectors/concept_diff-27b-it-L38/judge_affirmative_identification.json", "r") as f:
    judgments_detection = json.load(f)
with open("concept_vectors/concept_diff-27b-it-L38/judge_coherence.json", "r") as f:
    judgments_coherence = json.load(f)

scores: dict[str, float] = {}
for word in judgments_detection.keys():
    detection_scores = judgments_detection[word]["4.0"]
    coherence_scores = judgments_coherence[word]["4.0"]
    
    total = 0
    valid = 0
    for det, coh in zip(detection_scores, coherence_scores):
        if det is None or coh is None:
            continue
        valid += 1
        if det and coh:
            total += 1

    scores[word] = total / valid if valid else 0.0

with open("concept_vectors/concept_diff-27b-it-L38/scores_4.json", "w") as f:
    json.dump(scores, f, indent=4)


score_matrix = np.zeros((len(POSITIVE_WORDS), len(NEGATIVE_WORDS)))
for i, pos_word in enumerate(POSITIVE_WORDS):
    for j, neg_word in enumerate(NEGATIVE_WORDS):
        key = f"{neg_word}_to_{pos_word}"
        score_matrix[i, j] = judge_scores.get(key, 0.0)

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(score_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

ax.set_xticks(range(len(NEGATIVE_WORDS)))
ax.set_xticklabels(NEGATIVE_WORDS, rotation=45, ha="right")
ax.set_yticks(range(len(POSITIVE_WORDS)))
ax.set_yticklabels(POSITIVE_WORDS)
ax.set_xlabel("Source (patched from)")
ax.set_ylabel("Destination (steered toward)")
ax.set_title(f"Activation Patching Success Rate (MLP L{PATCH_LAYER_START}-{PATCH_LAYER_END})")

# Add score values in cells
for i in range(len(POSITIVE_WORDS)):
    for j in range(len(NEGATIVE_WORDS)):
        target_word = POSITIVE_WORDS[i]
        target_original_score = scores[target_word]
        text = ax.text(
            j, i, 
            f"{score_matrix[i, j]:.2f}\n(original: {target_original_score:.2f})", 
            ha="center", va="center",
            color="black" if 0.3 < score_matrix[i, j] < 0.7 else "white", 
            fontsize=8
        )

plt.colorbar(im, ax=ax, label="Success Rate")
plt.tight_layout()
fig.savefig(save_dir / "patching_heatmap.png", dpi=150)
plt.close(fig)
print(f"Saved heatmap to {save_dir / 'patching_heatmap.png'}")

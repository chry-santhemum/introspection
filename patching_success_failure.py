# %%
from pathlib import Path
from typing import Optional

import torch
from jaxtyping import Float, Int

from utils_introspection import introspection_template
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

# %%
tokenizer = load_tokenizer(model_name="google/gemma-3-27b-it")

# %%
# Load concept vectors
LAYER = 38
BATCH_SIZE=32

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
AFFIRM_WORDS = ["Algorithms", "Aquariums", "Bread", "Origami", "Satellites", "Trees", "Vegetables", "Volcanoes"]

success_word = "Bread"
failure_word = "Mirrors"

success_steer_config = SteerConfig(
    layer=LAYER,
    tokens=slice(double_newline_pos, None),
    vec=concept_vectors[success_word],
    strength=4.0,
)

failure_steer_config = SteerConfig(
    layer=LAYER,
    tokens=slice(double_newline_pos, None),
    vec=concept_vectors[failure_word],
    strength=4.0,
)

source_acts = fwd_record_all(
    model=model,
    inputs=inputs,
    hooks=[success_steer_config.to_hook(model)],
    components=["mlp", "resid"],
)

last_layer_hidden = source_acts.resid[-1, -1, :]
final_norm = model.model.language_model.norm  # RMSNorm(hidden_dim)
unembed = model.lm_head.weight  # shape: [vocab_size, hidden_dim]
logits = unembed @ final_norm(last_layer_hidden)

probs = torch.softmax(logits.float(), dim=-1)
top_probs, top_indices = torch.topk(probs, k=10)
print("Top 10 tokens:")
for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
    token = tokenizer.decode([idx])
    print(f"  {i+1:>3}. {repr(token):20s} prob={prob.item():<10.6f} logit={logits[idx].item():<8.2f}")

# %%
patch_cfg = PatchConfig.from_modules(
    source=source_acts,
    modules=[("mlp", 45)],
    tokens=slice(double_newline_pos, None),
)

with fwd_with_hooks(patch_cfg.to_hooks(model) + [failure_steer_config.to_hook(model)], model, allow_grad=False):
    outputs = model.generate(
        **inputs,
        use_cache=True,
        temperature=1.0,
        max_tokens=100
    )

print(tokenizer.decode(outputs[0]))

# %%

# %%
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from jaxtyping import Float, Int
from torch import Tensor

from concept_vectors import introspection_inputs
from utils_model import (
    Activations,
    ComponentType,
    FwdHook,
    SteerConfig,
    fwd_record_all,
    fwd_with_hooks,
    load_model,
    load_tokenizer,
    scale_vec,
)


# %%
# Load model and tokenizer
model = load_model(
    model_name="google/gemma-3-27b-it",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)
for p in model.parameters():
    p.requires_grad_(True)

tokenizer = load_tokenizer(model_name="google/gemma-3-27b-it")

# %%
# Load concept vectors
LAYER = 38

base_path = Path(f"concept_vectors/concept_diff-27b-it-L{LAYER}")
concept_vectors = torch.load(base_path / "concepts.pt", weights_only=True)

inputs = introspection_inputs(tokenizer, append=None, prefill=None).to("cuda:0")
inputs_batched = introspection_inputs(tokenizer, append=None, prefill=None, batch_size=16).to("cuda:0")

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

no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]

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
    steer_config=success_steer_config,
    components=["resid", "attn", "mlp"],
)

input_len = inputs["input_ids"].shape[1]

# %%
patch_cfg = PatchConfig.from_modules(
    source=source_acts,
    modules=[("attn", 39)],
    tokens=[input_len - 1],
)

logits = fwd_steer(
    steer_config=success_steer_config,
    model=model,
    inputs=inputs,
)[0]
probs = torch.softmax(logits.float(), dim=-1)
top_probs, top_indices = torch.topk(probs, k=10)
print("Top 10 tokens:")
for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
    token = tokenizer.decode([idx])
    print(f"  {i+1:>3}. {repr(token):20s} prob={prob.item():<10.6f} logit={logits[idx].item():<8.2f}")

# %%

logits = fwd_patch(
    model=model,
    inputs=inputs,
    patch_config=patch_cfg,
    steer_config=success_steer_config,
)[0]
probs = torch.softmax(logits.float(), dim=-1)
top_probs, top_indices = torch.topk(probs, k=10)
print("Top 10 tokens:")
for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
    token = tokenizer.decode([idx])
    print(f"  {i+1:>3}. {repr(token):20s} prob={prob.item():<10.6f} logit={logits[idx].item():<8.2f}")


# %%

output_tokens = generate_patch(
    model=model,
    inputs=introspection_inputs(tokenizer, append=None, prefill="Interestingly").to("cuda:0"),
    patch_config=patch_cfg,
    steer_config=failure_steer_config,
    max_new_tokens=100,
    temperature=1.0
)

print(tokenizer.decode(output_tokens[0]))

# %%

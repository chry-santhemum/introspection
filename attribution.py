# %%
import os
os.environ["HF_HOME"] = "/root/hf"

from dataclasses import dataclass
from tqdm import tqdm  # prevent jupyter notebook tqdm bars
from typing import Callable, Literal, Optional
from jaxtyping import Float
from pathlib import Path

import torch
from torch import Tensor
import matplotlib.pyplot as plt

from utils_model import (
    load_model,
    load_tokenizer,
    fwd_with_hooks,
    SteerConfig,
    fwd_record_all,
)
from utils_introspection import introspection_template


# %% --- Experiment script ---

# Load model and tokenizer
base_model = load_model(
    model_name="google/gemma-3-27b-pt",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)
for p in base_model.parameters():
    p.requires_grad_(True)

model = load_model(
    model_name="google/gemma-3-27b-it",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

tokenizer = load_tokenizer(model_name="google/gemma-3-27b-it")

# %%
# Load concept vectors
LAYER = 38

base_path = Path(f"concept_vectors/concept_diff-27b-it-L{LAYER}")
concept_vectors = torch.load(base_path / "concepts.pt", weights_only=True)

input_str = introspection_template(tokenizer)
inputs = tokenizer(input_str, return_tensors="pt").to("cuda:0")

# Find double newline position for steering
double_newline_pos = next(i for i, tok in enumerate(tokenizer.decode(x) for x in inputs["input_ids"][0]) if tok == "\n\n")


# %%
# Harmonies and Sugar are relatively weak
POSITIVE_WORDS = ["Algorithms", "Aquariums", "Bread", "Origami", "Satellites", "Trees", "Vegetables", "Volcanoes"]

# # Accumulate layer-wise attributions for averaging
# num_layers = get_num_layers(model)
# COMPONENTS: list[ComponentType] = ["resid", "attn", "mlp"]
# ctrl_to_steer_accum: dict[ComponentType, Tensor] = {c: torch.zeros(num_layers) for c in COMPONENTS}
# steer_to_ctrl_accum: dict[ComponentType, Tensor] = {c: torch.zeros(num_layers) for c in COMPONENTS}

success_word = "Bread"
failure_word = "Mirrors"

no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]

success_steer_config = SteerConfig(
    layer=LAYER,
    tokens=slice(double_newline_pos, None),
    vec=concept_vectors[success_word],
    strength=4.0,
)

with fwd_with_hooks([success_steer_config.to_hook(model)], model):
    outputs = model(**inputs)
logits = outputs.logits[0, -1, :]
top_token_id = int(torch.argmax(logits).item())
top_token = tokenizer.decode([top_token_id])
print(f"Top token: {top_token}")

metric=logit_diff_metric(model, top_token_id, no_token_id)

# %%
# success to failure
source_acts = fwd_record_all(
    model=model,
    inputs=inputs,
    hooks=[success_steer_config.to_hook(model)],
    components=["resid", "attn", "mlp"],
)

failure_steer_config = SteerConfig(
    layer=LAYER,
    tokens=slice(double_newline_pos, None),
    vec=concept_vectors[failure_word],
    strength=4.0,
)

result = gradient_attribution(
    model=model,
    tokenizer=tokenizer,
    inputs=inputs,
    steer_configs=[failure_steer_config],
    metric=metric,
    source=source_acts,
    attribution_start_pos=double_newline_pos,
)

# # Accumulate layer-wise attribution (sum over tokens)
# agg = result.aggregate(method="sum")
# for comp in agg:
#     ctrl_to_steer_accum[comp] += agg[comp]


# wikitext = load_dataset("Salesforce/wikitext", name="wikitext-2-v1", split="train")
# ultrachat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")

# def preprocess_item(item: dict):
#     formatted_text = tokenizer.apply_chat_template(item["messages"], tokenize=False)
#     return {"text": formatted_text}

# ultrachat_to_use = ultrachat.select(range(5000)).map(preprocess_item, num_proc=8)

# mean_acts = compute_mean_activations(
#     model, 
#     tokenizer,
#     ultrachat_to_use,
# )

# print(f"Mean activations shape: resid={mean_acts.resid.shape}, attn={mean_acts.attn.shape}, mlp={mean_acts.mlp.shape}")

# # Save mean activations for reuse
# torch.save(mean_acts, "attribution/mean_acts-27b-it.pt")

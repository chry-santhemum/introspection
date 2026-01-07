# %%
import os
os.environ["HF_HOME"] = "/root/hf"

from pathlib import Path
from tqdm import tqdm  # prevent jupyter notebook tqdm bars
import torch

from utils_introspection import introspection_template, introspection_rollout, no_steer_rollout
from utils_model import load_model, load_tokenizer, SteerConfig, fwd_with_hooks
from utils_attribution import gradient_attribution, logit_diff_metric

from concept_vectors import CONCEPT_WORDS

# %%
# Load model and tokenizer
base_model = load_model(
    model_name="google/gemma-3-27b-pt",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)
for p in base_model.parameters():
    p.requires_grad_(True)

tokenizer = load_tokenizer(model_name="google/gemma-3-27b-it")

LAYER = 38
POSITIVE_WORDS = ["Algorithms", "Aquariums", "Bread", "Origami", "Satellites", "Trees", "Vegetables", "Volcanoes"]

base_path = Path(f"concept_vectors/concept_diff-27b-it-L{LAYER}")
concept_vectors = torch.load(base_path / "concepts.pt", weights_only=True)

no_steer_rollouts = no_steer_rollout(
    model=base_model,
    tokenizer=tokenizer,
    other_hooks=[],
    save_dir=Path("attribution_it_pt/base_model"),
)

rollouts = introspection_rollout(
    model=base_model,
    tokenizer=tokenizer,
    concept_vectors=concept_vectors,
    words=CONCEPT_WORDS,
    strengths=[4.0],
    layer=LAYER,
    other_hooks=[],
    save_dir=Path("attribution_it_pt/base_model"),
)


# %%

instr_model = load_model(
    model_name="google/gemma-3-27b-it",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

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

no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]


for positive_word in POSITIVE_WORDS:
    steer_config = SteerConfig(
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

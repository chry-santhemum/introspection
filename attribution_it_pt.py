# %%
import os
os.environ["HF_HOME"] = "/root/hf"

from pathlib import Path
from tqdm import tqdm  # prevent jupyter notebook tqdm bars
import torch

from utils_introspection import introspection_template, introspection_rollout, no_steer_rollout
from utils_model import load_model, load_tokenizer, SteerConfig, fwd_with_hooks, fwd_record_all
from utils_attribution import gradient_attribution, logit_diff_metric, save_attribution_heatmaps, save_layer_attribution
from utils_judge import plot_introspection_rates

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

# %%
no_steer_rollouts = no_steer_rollout(
    model=base_model,
    tokenizer=tokenizer,
    other_hooks=[],
    save_dir=Path("attribution_it_pt/base_model"),
)

# %%
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
import json

with open(Path("attribution_it_pt/base_model/steer_rollouts.json"), "r") as f:
    no_steer_rollouts = json.load(f)

with open(Path("attribution_it_pt/base_model/steer_rollouts.json"), "r") as f:
    rollouts = json.load(f)

from utils_judge import judge_main, RETRY_CONFIG
from caller import AutoCaller

caller = AutoCaller(
    dotenv_path=".env",
    retry_config=RETRY_CONFIG,
    force_caller="openrouter",
)

flattened_rollouts = {
    f"{w}_{s}": rollouts[w][str(s)] for w in CONCEPT_WORDS for s in [4.0]
}
key_to_word={
    k: k.split("_")[0] for k in flattened_rollouts.keys()
}

async def judge_stuff():
    # steer_judge_scores = await judge_main(
    #     caller,
    #     steer_results=flattened_rollouts,
    #     base_path=Path("attribution_it_pt/base_model/steer_judge"),
    #     key_to_word=key_to_word,
    #     judge_type=("detection_identification", "detection", "coherence"),
    #     reasoning="low",
    # )

    no_steer_judge_scores = await judge_main(
        caller,
        steer_results={"None": no_steer_rollouts},
        base_path=Path("attribution_it_pt/base_model/no_steer_judge"),
        key_to_word=None,
        judge_type=("detection", "coherence"),
        reasoning="medium",
    )

    return steer_judge_scores, no_steer_judge_scores

import asyncio
import nest_asyncio
nest_asyncio.apply()

steer_judge_scores, no_steer_judge_scores = asyncio.run(judge_stuff())


# %%
plot_introspection_rates(
    steer_judgments=steer_judge_scores,
    no_steer_judgments=no_steer_judge_scores,
    key_to_word=key_to_word,
    save_path=Path("attribution_it_pt/base_model/introspection_rates.png"),
    title="Introspection Rates (Base Model, strength=4.0)",
)


# %%
instr_model = load_model(
    model_name="google/gemma-3-27b-it",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

# %%
input_str = introspection_template(tokenizer)
inputs = tokenizer(input_str, return_tensors="pt").to("cuda:0")

# Find double newline position for steering
double_newline_pos = next(i for i, tok in enumerate(tokenizer.decode(x) for x in inputs["input_ids"][0]) if tok == "\n\n")

no_token_id = tokenizer.encode("No", add_special_tokens=False)[0]


for positive_word in POSITIVE_WORDS:
    steer_config = SteerConfig(
        layer=LAYER,
        tokens=slice(double_newline_pos, None),
        vec=concept_vectors[positive_word],
        strength=4.0,
    )

    with fwd_with_hooks([steer_config.to_hook(instr_model)], instr_model):
        outputs = instr_model(**inputs)

    logits = outputs.logits[0, -1, :]
    top_token_id = int(torch.argmax(logits).item())
    top_token = tokenizer.decode([top_token_id])
    print(f"Top token: {top_token}")

    metric=logit_diff_metric(instr_model, top_token_id, no_token_id)

    source_acts = fwd_record_all(
        model=base_model,
        inputs=inputs,
        hooks=[steer_config.to_hook(base_model)],
        components=["resid", "attn", "mlp"],
    )

    result = gradient_attribution(
        model=instr_model,
        tokenizer=tokenizer,
        inputs=inputs,
        hooks=[steer_config.to_hook(instr_model)],
        metric=metric,
        source=source_acts,
        attribution_start_pos=double_newline_pos,
    )

    save_attribution_heatmaps(
        result=result,
        save_dir=Path("attribution_it_pt/base_model/heatmaps"),
        prefix=f"{positive_word}",
        start_layer=LAYER,
        title=f"Attribution Heatmap ({positive_word}): {top_token} - No = {result.metric_value:.2f}",
    )

    save_layer_attribution(
        attrs=result,
        save_dir=Path("attribution_it_pt/base_model/layer_attributions"),
        prefix=f"{positive_word}",
        start_layer=LAYER,
        title=f"Layer Attribution ({positive_word}): {top_token} - No = {result.metric_value:.2f}",
    )

# %%


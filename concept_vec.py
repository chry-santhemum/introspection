# %%
import os
os.environ["HF_HOME"] = "/root/hf"

import torch
from typing import Optional
from pathlib import Path
from textwrap import dedent
import json

from torch import Tensor
from transformers import BitsAndBytesConfig

from utils_misc import find_executable_batch_size
from utils_model import (
    load_model, 
    load_tokenizer, 
    get_layer_at_fraction,
    fwd_record_resid,
)

def extract_concept_vectors(
    model,
    tokenizer,
    prompts: list[str],
    layer: int,
    pos: int = -1,
    batch_size: int = 16,  # will be auto-reduced on OOM
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device

    # Format all prompts upfront
    messages = [[{"role": "user", "content": p}] for p in prompts]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    @find_executable_batch_size(starting_batch_size=batch_size)
    def process_batch(batch_size, start_idx):
        batch = formatted[start_idx:start_idx + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        return fwd_record_resid(model, inputs, layer, pos)

    # Process all prompts
    all_acts = []
    idx = 0
    while idx < len(formatted):
        acts = process_batch(idx)
        all_acts.append(acts)
        idx += acts.shape[0]

    all_acts = torch.cat(all_acts, dim=0)

    # Build result dict
    result = {}
    for i, prompt in enumerate(prompts):
        result[prompt] = all_acts[i]

    return result

# %%
model = load_model(
    model_name="google/gemma-3-27b-it",
    quantization_config=BitsAndBytesConfig(load_in_8bit=True)
)

# %%
tokenizer = load_tokenizer(model_name="google/gemma-3-27b-it")

# %%
extract_concept_vectors(
    model=model,
    tokenizer=tokenizer,
    prompts=["Tell me about the concept of love", "Tell me about the concept of hate"],
    layer=get_layer_at_fraction(model, 0.7),
)

# %%

def save_concept_vectors(
    vector_dict: dict[str, Tensor],
    save_dir: Path,
    metadata: Optional[dict] = None,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    if (save_dir / "concepts.pt").exists():
        print(f"Concept vectors already exist at {save_dir}, updating its keys...")
        existing_vectors = torch.load(save_dir / "concepts.pt", weights_only=True)
        existing_vectors.update(vector_dict)
        vector_dict = existing_vectors

    torch.save(vector_dict, save_dir / "concepts.pt")
    print(f"Saved concept vectors to {save_dir}")

    if metadata is not None:
        metadata_path = save_dir / "metadata.json"
        if metadata_path.exists():
            print(f"Metadata already exists at {metadata_path}, updating its keys...")
            with open(metadata_path) as f:
                existing_metadata = json.load(f)
            existing_metadata.update(metadata)
            metadata = existing_metadata

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f"Saved metadata to {metadata_path}")


# Predefined baseline words (100 words from the paper)
# Note that "Butterflies" appears twice in the original paper
DEFAULT_BASELINE_WORDS = [
    "Desks", "Jackets", "Gondolas", "Laughter", "Intelligence",
    "Bicycles", "Chairs", "Orchestras", "Sand", "Pottery",
    "Arrowheads", "Jewelry", "Daffodils", "Plateaus", "Estuaries",
    "Quilts", "Moments", "Bamboo", "Ravines", "Archives",
    "Hieroglyphs", "Stars", "Clay", "Fossils", "Wildlife",
    "Flour", "Traffic", "Bubbles", "Honey", "Geodes",
    "Magnets", "Ribbons", "Zigzags", "Puzzles", "Tornadoes",
    "Anthills", "Galaxies", "Poverty", "Diamonds", "Universes",
    "Vinegar", "Nebulae", "Knowledge", "Marble", "Fog",
    "Rivers", "Scrolls", "Silhouettes", "Marbles", "Cakes",
    "Valleys", "Whispers", "Pendulums", "Towers", "Tables",
    "Glaciers", "Whirlpools", "Jungles", "Wool", "Anger",
    "Ramparts", "Flowers", "Research", "Hammers", "Clouds",
    "Justice", "Dogs", "Butterflies", "Needles", "Fortresses",
    "Bonfires", "Skyscrapers", "Caravans", "Patience", "Bacon",
    "Velocities", "Smoke", "Electricity", "Sunsets", "Anchors",
    "Parchments", "Courage", "Statues", "Oxygen", "Time",
    "Butterflies", "Fabric", "Pasta", "Snowflakes", "Mountains",
    "Echoes", "Pianos", "Sanctuaries", "Abysses", "Air",
    "Dewdrops", "Gardens", "Literature", "Rice", "Enigmas",
]


# Concept pairs for different experiments
CONCEPT_PAIRS = {
    "all_caps": (
        "HI! HOW ARE YOU? I'M DOING GREAT TODAY!",
        "Hi! How are you? I'm doing great today!",
    ),
    "recursion_code": (
        dedent("""
            def factorial(n):
                if n <= 1:
                    return 1
                return n * factorial(n - 1)
        """).strip(),
        dedent("""
            def factorial(n):
                result = 1
                for i in range(2, n + 1):
                    result *= i
                return result
        """).strip(),
    ),
    "if_statement_code": (
        dedent("""
            def check_positive(x):
                if x > 0:
                    return True
                return False
        """).strip(),
        dedent("""
            def check_positive(x):
                result = x > 0
                return result
        """).strip(),
    ),
    "loop_code": (
        dedent("""
            for i in range(10):
                print(i)
        """).strip(),
        "print(list(range(10)))",
    ),
}

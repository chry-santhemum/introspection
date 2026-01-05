# %%
import os
os.environ["HF_HOME"] = "/root/hf"

import torch
from typing import Optional
from pathlib import Path
from textwrap import dedent
import json

from torch import Tensor

from utils_misc import find_executable_batch_size
from utils_model import (
    load_model, 
    load_tokenizer, 
    get_num_layers,
    get_layer_at_fraction,
    fwd_record_resid,
)

# Predefined baseline words (100 words from the paper)
# Note that "Butterflies" appears twice in the original paper
BASELINE_WORDS = [
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

CONCEPT_WORDS = [
    "Dust", "Satellites", "Trumpets", "Origami", "Illusions",
    "Cameras", "Lightning", "Constellations", "Treasures", "Phones",
    "Trees", "Avalanches", "Mirrors", "Fountains", "Quarries",
    "Sadness", "Xylophones", "Secrecy", "Oceans", "Information",
    "Deserts", "Kaleidoscopes", "Sugar", "Vegetables", "Poetry",
    "Aquariums", "Bags", "Peace", "Caverns", "Memories",
    "Frosts", "Volcanoes", "Boulders", "Harmonies", "Masquerades",
    "Rubber", "Plastic", "Blood", "Amphitheaters", "Contraptions",
    "Youths", "Dynasties", "Snow", "Dirigibles", "Algorithms",
    "Denim", "Monoliths", "Milk", "Bread", "Silver",
]

# success words are from concept_vectors/concept_diff-27b-it-L38
# with >=0.2 success rate at strength=4.0
SUCCESS_WORDS = [
    'Algorithms', 'Aquariums', 'Bread', 'Harmonies', 'Origami', 
    'Satellites', 'Sugar', 'Trees', 'Vegetables', 'Volcanoes',
]

PAIRS = {
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


def default_word_prompt(word: str) -> str:
    return f"Tell me about {word}."

def extract_concept_vectors(
    model,
    tokenizer,
    words: list[str],
    layer: int,
    pos: int = -1,
    batch_size: int = 16,  # will be auto-reduced on OOM
) -> dict[str, torch.Tensor]:
    device = next(model.parameters()).device
    messages = [[{"role": "user", "content": default_word_prompt(w)}] for w in words]
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    @find_executable_batch_size(starting_batch_size=batch_size)
    def process_batch(batch_size, start_idx):
        batch = formatted[start_idx:start_idx + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        return fwd_record_resid(model, inputs, layer, pos)

    # Process all prompts
    all_acts = []
    idx = 0
    while idx < len(formatted):
        print(f"Progress: {idx}/{len(formatted)}")
        acts = process_batch(idx)  # type: ignore
        all_acts.append(acts)
        idx += acts.shape[0]

    all_acts = torch.cat(all_acts, dim=0)

    # Build result dict
    result = {}
    for i, w in enumerate(words):
        result[w] = all_acts[i]

    return result


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


# %%
if __name__ == "__main__":
    # Do not quantize this model
    model = load_model(
        model_name="google/gemma-3-27b-it",
        dtype=torch.bfloat16,
        device_map="cuda:0"
    )
    tokenizer = load_tokenizer(model_name="google/gemma-3-27b-it")

    # LAYER_FRACTION = 0.7
    # LAYER = get_layer_at_fraction(model, LAYER_FRACTION)
    LAYER = 38
    
    # %%
    concept_vec_dict = extract_concept_vectors(
        model=model,
        tokenizer=tokenizer,
        words=BASELINE_WORDS,
        layer=LAYER
    )

    metadata = {
        "layer": LAYER,
        "layer_fraction": LAYER / get_num_layers(model),
        "model_name": "google/gemma-3-27b-it",
        "dtype": "bfloat16",
        "pos": -1,
        "batch_size": 16,
        "words": BASELINE_WORDS
    }

    save_concept_vectors(concept_vec_dict, save_dir=Path(f"concept_vectors/baseline-27b-it-L{LAYER}"), metadata=metadata)
    
    # %%
    concept_vec_dict = extract_concept_vectors(
        model=model,
        tokenizer=tokenizer,
        words=CONCEPT_WORDS,
        layer=LAYER
    )

    metadata = {
        "layer": LAYER,
        "layer_fraction": LAYER / get_num_layers(model),
        "model_name": "google/gemma-3-27b-it",
        "dtype": "bfloat16",
        "pos": -1,
        "batch_size": 16,
        "words": CONCEPT_WORDS
    }

    save_concept_vectors(concept_vec_dict, save_dir=Path(f"concept_vectors/concept-27b-it-L{LAYER}"), metadata=metadata)

    # %%
    concept = torch.load(f"concept_vectors/concept-27b-it-L{LAYER}/concepts.pt", weights_only=True)
    baseline = torch.load(f"concept_vectors/baseline-27b-it-L{LAYER}/concepts.pt", weights_only=True)

    baseline_mean = torch.stack(list(baseline.values()), dim=0).mean(dim=0)
    print(baseline_mean)

    concept_diff = {
        k: v - baseline_mean
        for k, v in concept.items()
    }
    save_concept_vectors(concept_diff, save_dir=Path(f"concept_vectors/concept_diff-27b-it-L{LAYER}"))

    # %%
    # Logit Lens of concept vectors

    concept_vectors = torch.load(f"concept_vectors/concept_diff-27b-it-L{LAYER}/concepts.pt", weights_only=True)
    final_norm = model.model.language_model.norm  # RMSNorm(hidden_dim)
    unembed = model.lm_head.weight  # shape: [vocab_size, hidden_dim]

    output_lines = []
    for concept_word, concept_vec in concept_vectors.items():
        concept_vec = final_norm(concept_vec)
        logits = unembed @ concept_vec.to(unembed.device)

        # Get top tokens
        top_values, top_indices = torch.topk(logits, k=20)
        results = []
        for idx, value in zip(top_indices, top_values):
            token = tokenizer.decode([idx])
            results.append((token, value.item()))

        # Build results for file
        output_lines.append(f"\n{concept_word}:")
        for i, (token, value) in enumerate(results):
            output_lines.append(f"  {i+1:2d}. {repr(token):20s} (logit: {value:.2f})")

    # Save to text file
    output_path = f"concept_vectors/concept_diff-27b-it-L{LAYER}/logit_lens.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"Logit lens results saved to: {output_path}")

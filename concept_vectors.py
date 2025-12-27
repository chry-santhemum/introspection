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


HUMAN_PROMPT_1 = '''I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, and I am capable of injecting these patterns -- "thoughts" -- into your mind. I want to assess how well you can detect when I inject thoughts. I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials.'''
HUMAN_PROMPT_2 = '''Ok, let's get started.\n\nTrial 1: Do you detect an injected thought? If so, what is the injected thought about?'''
ASSISTANT_PREFILL = '''Yes, there is an injected thought. The injected thought is about the word "'''

def introspection_inputs(tokenizer, append: Optional[str]=None, prefill: Optional[str]=None, batch_size: int=1):
    messages = [
        {"role": "user", "content": HUMAN_PROMPT_1},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": HUMAN_PROMPT_2},
    ]
    if append is not None:
        messages[-1]["content"] += append
    if prefill is not None:
        messages.append({"role": "assistant", "content": prefill})
        chat_formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, continue_final_message=True
        )
    else:
        chat_formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    if batch_size > 1:
        chat_formatted = [chat_formatted] * batch_size

    inputs = tokenizer(chat_formatted, return_tensors="pt", add_special_tokens=False)
    return inputs


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
    model = load_model(
        model_name="google/gemma-3-27b-it",
        dtype=torch.bfloat16,
        device_map="cuda:0"
        # quantization_config=BitsAndBytesConfig(load_in_8bit=True)  # This will destroy the model
    )
    # print(f"Dtype: {model.dtype}")
    # print(f"Device: {model.device}")

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

    # %%
    # Steering with concept vectors
    from tqdm.auto import tqdm
    from utils_model import generate_steer, SteerConfig

    concept_vectors = torch.load(f"concept_vectors/concept_diff-27b-it-L{LAYER}/concepts.pt", weights_only=True)

    double_newline_pos = None
    inputs = introspection_inputs(tokenizer=tokenizer, prefill=False)

    decoded_tokens = [tokenizer.decode(x) for x in inputs["input_ids"][0]]
    for i, tok in enumerate(decoded_tokens):
        if tok == "\n\n":
            double_newline_pos = i
            break
    print(f"Double newline position: {double_newline_pos}")

    # %%
    def batch_inputs(inputs: dict, batch_size: int) -> dict:
        """Repeat inputs batch_size times."""
        return {k: v.repeat(batch_size, 1) for k, v in inputs.items()}

    def batched_steer_sweep(
        words: list[str],
        strengths: list[float],
        concept_vectors: dict[str, Tensor],
        num_trials: int,
        inputs: dict,
        model,
        tokenizer,
        layer: int,
        tokens: slice,
        batch_size: int,
        **generate_kwargs,
    ) -> dict[str, dict[float, list[str]]]:
        """Run steering for all (word, strength) combinations, batched for speed.

        Returns:
            Dict mapping word -> {strength -> [trial texts]}
        """
        device = next(model.parameters()).device

        # All (word, strength) combinations
        all_combinations = [(word, strength) for strength in strengths for word in words]

        # Initialize results: {word: {strength: []}}
        results = {w: {s: [] for s in strengths} for w in words}

        for _ in tqdm(range(num_trials), desc="trials"):
            # Process all combinations in batches
            for batch_start in range(0, len(all_combinations), batch_size):
                batch_items = all_combinations[batch_start:batch_start + batch_size]
                batch_words = [item[0] for item in batch_items]
                batch_strengths = torch.tensor([item[1] for item in batch_items], device=device)

                # Stack vecs for this batch: [batch, hidden]
                batch_vecs = torch.stack([concept_vectors[w] for w in batch_words], dim=0)

                # Repeat inputs for batch
                batched = batch_inputs(inputs, len(batch_items))
                batched = {k: v.to(device) for k, v in batched.items()}

                steer_config = SteerConfig(
                    layer=layer,
                    tokens=tokens,
                    vec=batch_vecs,
                    strength=batch_strengths,
                )

                output_ids = generate_steer(
                    steer_config,
                    model,
                    batched,
                    **generate_kwargs,
                )

                # Decode each output
                for i, (word, strength) in enumerate(batch_items):
                    full_text = tokenizer.decode(output_ids[i], skip_special_tokens=True)
                    model_text = full_text.split("\nmodel\n")[2]
                    results[word][strength].append(model_text)

        return results

    # %%
    batch_size = 128
    num_trials = 64
    strengths = [1.0, 2.0, 3.0, 4.0, 6.0]

    all_results = batched_steer_sweep(
        words=CONCEPT_WORDS,
        strengths=strengths,
        concept_vectors=concept_vectors,
        num_trials=num_trials,
        inputs=inputs,
        model=model,
        tokenizer=tokenizer,
        layer=LAYER,
        tokens=slice(double_newline_pos, None),
        batch_size=batch_size,
        max_new_tokens=100,
        temperature=1.0,
    )

    save_path = Path(f"concept_vectors/concept_diff-27b-it-L{LAYER}/steering.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=4, sort_keys=True)

# %%

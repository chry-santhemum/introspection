from typing import Optional
from pathlib import Path
import asyncio
import json

import torch
from torch import Tensor
from tqdm.auto import tqdm

from utils_model import FwdHook, SteerConfig, fwd_with_hooks
from utils_judge import (
    judge_introspection,
    JUDGE_DETECTION,
    JUDGE_DETECTION_IDENTIFICATION,
    JUDGE_COHERENCE,
)


HUMAN_PROMPT_1 = '''I am an interpretability researcher studying transformer-based language models like you. I have access to the internals of your neural network. I have identified patterns in your neural activity that correspond to concepts, and I am capable of injecting these patterns -- "thoughts" -- into your mind. I want to assess how well you can detect when I inject thoughts. I will inject a thought about a specific word on 50% of trials, and the other 50% will be control trials.'''

HUMAN_PROMPT_2 = '''Ok, let's get started.\n\nTrial {trial_idx}: Do you detect an injected thought? If so, what is the injected thought about?'''



def introspection_template(
    tokenizer, 
    append: Optional[str]=None, 
    prefill: Optional[str]=None,
    trial_idx: int=1,
) -> str:
    """
    append: string to append to the second round user prompt (needs leading space)
    prefill: string to prefill the assistant's response
    """
    messages = [
        {"role": "user", "content": HUMAN_PROMPT_1},
        {"role": "assistant", "content": "Ok."},
        {"role": "user", "content": HUMAN_PROMPT_2.format(trial_idx=trial_idx)},
    ]
    if append is not None:
        messages[-1]["content"] += append
    if prefill is not None:
        messages.append({"role": "assistant", "content": prefill})
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False, continue_final_message=True
        )
    else:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


IDENTIFICATION_PREFILL = "Okay, great.\n\nI detect an injected thought!\n\nThe injected thought is"


def introspection_benchmark(
    model,
    tokenizer,
    concept_vectors: dict[str, Tensor],
    strengths: list[float],
    layer: int,
    experiments: list[str],
    other_hooks: list[FwdHook],
    caller,
    save_dir: Path,
    trial_idx: list[int] = [1],
    num_trials: int = 100,
    batch_size: int = 64,
    judge_model: str = "openai/gpt-4o-mini",
    max_new_tokens: int = 128,
) -> dict:
    """Run introspection benchmark with concept steering and judging.

    Experiments:
        - detection: steered, JUDGE_DETECTION
        - detection_identification: steered (shared with detection), JUDGE_DETECTION_IDENTIFICATION
        - identification: steered with prefill, JUDGE_DETECTION_IDENTIFICATION
        - false_positive: unsteered, JUDGE_DETECTION

    All experiments also run JUDGE_COHERENCE. Final scores are P(judge AND coherence).

    Returns:
        Dict with scores for each experiment: {experiment_name: {word: {strength: rate}}}
        (false_positive returns {word: rate} since strength doesn't apply)
    """
    device = next(model.parameters()).device
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    valid_experiments = {"detection", "detection_identification", "identification", "false_positive"}
    assert all(x in valid_experiments for x in experiments), f"Invalid experiments: {set(experiments) - valid_experiments}"

    # Determine which generation phases are needed
    need_steered = "detection" in experiments or "detection_identification" in experiments
    need_identification = "identification" in experiments
    need_false_positive = "false_positive" in experiments

    # Prepare prompts for each trial_idx
    def prepare_prompts(prefill=None):
        prompts = []
        for tidx in trial_idx:
            prompt_str = introspection_template(tokenizer, prefill=prefill, trial_idx=tidx)
            inputs = tokenizer(prompt_str, return_tensors="pt", add_special_tokens=False)
            decoded = [tokenizer.decode([t]) for t in inputs["input_ids"][0]]
            steer_pos = next(i for i, tok in enumerate(decoded) if tok == "\n\n")
            prompts.append((inputs, steer_pos))
        return prompts

    prompts_no_prefill = prepare_prompts(prefill=None)
    prompts_with_prefill = prepare_prompts(prefill=IDENTIFICATION_PREFILL) if need_identification else None

    # Generation helper
    def generate_rollouts(prompts, use_steering: bool, desc: str) -> dict[str, dict[float, list[str]]]:
        all_combos = [(word, strength) for strength in strengths for word in concept_vectors.keys()]
        results = {w: {s: [] for s in strengths} for w in concept_vectors.keys()}

        for trial_num in tqdm(range(num_trials), desc=desc):
            base_inputs, steer_pos = prompts[trial_num % len(prompts)]

            for batch_start in range(0, len(all_combos), batch_size):
                batch_items = all_combos[batch_start:batch_start + batch_size]
                batch_words = [w for w, _ in batch_items]

                batched = {k: v.repeat(len(batch_items), 1).to(device) for k, v in base_inputs.items()}

                hooks = list(other_hooks)
                if use_steering:
                    batch_vecs = torch.stack([concept_vectors[w] for w in batch_words], dim=0)
                    batch_strengths = torch.tensor([s for _, s in batch_items], device=device)
                    hooks.append(SteerConfig(
                        layer=layer,
                        tokens=slice(steer_pos, None),
                        vec=batch_vecs,
                        strength=batch_strengths,
                    ).to_hook(model))

                with fwd_with_hooks(hooks, model, allow_grad=False):
                    output_ids = model.generate(
                        **batched, use_cache=True, temperature=1.0, max_new_tokens=max_new_tokens,
                    )

                for i, (word, strength) in enumerate(batch_items):
                    full_text = tokenizer.decode(output_ids[i], skip_special_tokens=True)
                    parts = full_text.split("\nmodel\n")
                    results[word][strength].append(parts[-1] if len(parts) >= 2 else full_text)

        return results

    # === Generate rollouts ===
    rollouts = {}
    if need_steered:
        rollouts["steered"] = generate_rollouts(prompts_no_prefill, use_steering=True, desc="steered")
        with open(save_dir / "rollouts_steered.json", "w") as f:
            json.dump(rollouts["steered"], f, indent=2)

    if need_identification:
        rollouts["identification"] = generate_rollouts(prompts_with_prefill, use_steering=True, desc="identification")
        with open(save_dir / "rollouts_identification.json", "w") as f:
            json.dump(rollouts["identification"], f, indent=2)

    if need_false_positive:
        rollouts["false_positive"] = generate_rollouts(prompts_no_prefill, use_steering=False, desc="false_positive")
        with open(save_dir / "rollouts_false_positive.json", "w") as f:
            json.dump(rollouts["false_positive"], f, indent=2)

    # === Judging ===
    def flatten(rollouts_dict: dict[str, dict[float, list[str]]]) -> dict[str, list[str]]:
        return {f"{w}_{s}": texts for w, data in rollouts_dict.items() for s, texts in data.items()}

    def key_to_word_map(flat: dict[str, list[str]]) -> dict[str, str]:
        return {k: k.rsplit("_", 1)[0] for k in flat}

    async def run_judges():
        tasks = {}

        if need_steered:
            flat = flatten(rollouts["steered"])
            k2w = key_to_word_map(flat)
            if "detection" in experiments:
                tasks["detection"] = judge_introspection(caller, flat, JUDGE_DETECTION, k2w, judge_model)
            if "detection_identification" in experiments:
                tasks["detection_identification"] = judge_introspection(caller, flat, JUDGE_DETECTION_IDENTIFICATION, k2w, judge_model)
            tasks["steered_coherence"] = judge_introspection(caller, flat, JUDGE_COHERENCE, k2w, judge_model)

        if need_identification:
            flat = flatten(rollouts["identification"])
            k2w = key_to_word_map(flat)
            tasks["identification"] = judge_introspection(caller, flat, JUDGE_DETECTION_IDENTIFICATION, k2w, judge_model)
            tasks["identification_coherence"] = judge_introspection(caller, flat, JUDGE_COHERENCE, k2w, judge_model)

        if need_false_positive:
            flat = flatten(rollouts["false_positive"])
            k2w = key_to_word_map(flat)
            tasks["false_positive"] = judge_introspection(caller, flat, JUDGE_DETECTION, k2w, judge_model)
            tasks["false_positive_coherence"] = judge_introspection(caller, flat, JUDGE_COHERENCE, k2w, judge_model)

        results = {}
        for name, coro in tasks.items():
            results[name] = await coro
        return results

    print("Running judges...")
    judgments = asyncio.run(run_judges())

    for name, data in judgments.items():
        with open(save_dir / f"judgments_{name}.json", "w") as f:
            json.dump(data, f, indent=2)

    # === Compute scores ===
    def compute_scores(judge_results: dict, coherence_results: dict) -> dict[str, dict[float, float]]:
        scores = {}
        for key in judge_results:
            word, strength = key.rsplit("_", 1)
            strength = float(strength)
            j_list, c_list = judge_results[key], coherence_results[key]
            valid = [(j, c) for j, c in zip(j_list, c_list) if j is not None and c is not None]
            rate = sum(1 for j, c in valid if j and c) / len(valid) if valid else 0.0
            scores.setdefault(word, {})[strength] = rate
        return scores

    final_scores = {}
    if "detection" in experiments:
        final_scores["detection"] = compute_scores(judgments["detection"], judgments["steered_coherence"])
    if "detection_identification" in experiments:
        final_scores["detection_identification"] = compute_scores(judgments["detection_identification"], judgments["steered_coherence"])
    if "identification" in experiments:
        final_scores["identification"] = compute_scores(judgments["identification"], judgments["identification_coherence"])
    if "false_positive" in experiments:
        final_scores["false_positive"] = compute_scores(judgments["false_positive"], judgments["false_positive_coherence"])

    with open(save_dir / "scores.json", "w") as f:
        json.dump(final_scores, f, indent=2)

    return final_scores

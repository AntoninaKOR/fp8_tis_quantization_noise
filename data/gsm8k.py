"""
GSM8K dataset utilities.

Copied directly from QeRL (utils/data.py, utils/rewards.py, eval/gsm8k_Nemotron.py)
and adapted for the FP8-TIS training loop.
"""

from __future__ import annotations

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instruction / prompt format  (from QeRL eval/gsm8k_Nemotron.py)
# ---------------------------------------------------------------------------

GSM8K_INSTRUCTION = (
    "Let's think step by step first within <think> </think> tags, "
    "and output the final answer after \"####\" tag, i.e.,:\n"
    "    <think>\n"
    "    ...\n"
    "    </think>\n"
    "    #### number"
)


# ---------------------------------------------------------------------------
# Gold answer extraction  (copied from QeRL utils/data.py)
# ---------------------------------------------------------------------------

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


# ---------------------------------------------------------------------------
# Predicted answer extraction  (copied from QeRL utils/rewards.py)
# ---------------------------------------------------------------------------

def extract_solution(solution_str: str, method: str = "strict") -> str | None:
    """
    Extract the final numeric answer from the solution string.
    if method is "strict", it requires the "#### <number>" pattern.
    if method is "flexible", it takes the last numeric value in the generated text.

    Args:
        solution_str: The solution string to extract the answer from.
        method: The method to use to extract the answer.

    Returns:
        The final numeric answer, or None if no answer is found.

    """
    assert method in ("strict", "flexible")

    if method == "strict":
        # Tests formatting too: requires "#### <number>"
        solutions = re.findall(r"#### (\-?[0-9\.\,]+)", solution_str)
        if len(solutions) == 0:
            final_answer = None
        else:
            # take the last match
            final_answer = solutions[-1].replace(",", "").replace("$", "")

    elif method == "flexible":
        answer = re.findall(r"(\-?[0-9\.\,]+)", solution_str)
        final_answer = None
        if len(answer) == 0:
            pass
        else:
            invalid_str = ["", "."]
            for candidate in reversed(answer):
                if candidate not in invalid_str:
                    final_answer = candidate
                    break

    return final_answer


def gsm8k_score(
    response: str,
    answer: str,
    method: str = "strict",
    format_score: float = 0.0,
    score: float = 1.0,
) -> float:
    """
    Binary exact-match reward for a single (response, answer) pair.

    Returns 1.0 if extracted solution matches gold, else 0.0.
    """
    solution = extract_solution(solution_str=response, method=method)
    if solution is None:
        return format_score
    return score if solution == answer else format_score



def gsm8k_reward_batch(predictions: list[str], gold_answers: list[str]) -> list[float]:
    return [gsm8k_score(p, g) for p, g in zip(predictions, gold_answers)]



def get_gsm8k_questions(split: str = "train"):
    """
    Load GSM8K in QeRL chat-message format.

    Copied from QeRL utils/data.py get_gsm8k_questions().
    Uses openai/gsm8k (same as QeRL), no system prompt,
    answer extracted with extract_hash_answer.

    Returns a HuggingFace Dataset with columns:
        prompt  : list[dict]   — chat messages [{'role':'user','content':...}]
        answer  : str          — extracted gold numeric answer
        question: str          — raw question text
    """
    from datasets import load_dataset

    data = load_dataset("openai/gsm8k", "main", split=split)

    data = data.map(lambda x: {
        "prompt": [
            {"role": "user", "content": x["question"] + " " + GSM8K_INSTRUCTION}
        ],
        "answer": extract_hash_answer(x["answer"]),
        "question": x["question"],
    })
    return data


def load_gsm8k(split: str = "train") -> list[dict]:
    """
    Return GSM8K as a plain list of dicts.

    Each dict has:
        question : str
        answer   : str   (raw chain-of-thought)
        gold     : str   (extracted numeric answer via extract_hash_answer)
        prompt   : list[dict]  (chat format, QeRL-style)
    """
    data = get_gsm8k_questions(split)
    records = []
    for example in data:
        records.append({
            "question": example["question"],
            "answer":   example["answer"],        # already extracted by map above
            "gold":     example["answer"],        # same field — numeric only
            "prompt":   example["prompt"],        # chat list
        })
    return records


def get_gsm8k_dataloader(
    split: str = "train",
    batch_size: int = 16,
    shuffle: bool = True,
    seed: int = 42,
):
    """
    Iterator that yields batches of GSM8K records.

    Each batch is a dict of lists:
        questions, golds, prompts, raw_prompts (chat lists)
    """
    import random

    records = load_gsm8k(split)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(records)

    def _batches():
        for start in range(0, len(records), batch_size):
            chunk = records[start : start + batch_size]
            yield {
                "questions":   [r["question"] for r in chunk],
                "golds":       [r["gold"]     for r in chunk],
                "prompts":     [r["prompt"]   for r in chunk],   # chat lists
            }

    return _batches()


def apply_chat_template(tokenizer, chat_prompts: list[list[dict]]) -> list[str]:
    """
    Convert a list of chat message lists into formatted strings
    using the tokenizer's chat template.

    Args:
        tokenizer:    HuggingFace tokenizer with apply_chat_template
        chat_prompts: list of [{'role':..., 'content':...}, ...]

    Returns:
        list of formatted prompt strings
    """
    return [
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        for messages in chat_prompts
    ]


def normalize_number(value: str) -> str:
    """
    Normalize a number string by stripping whitespace and commas.
    """
    value = value.strip().replace(",", "")
    try:
        parsed = float(value)
        if parsed == int(parsed):
            return str(int(parsed))
        return str(parsed)
    except ValueError:
        return value



"""Evaluation metric primitives."""
from __future__ import annotations

from collections import Counter

from src.utils.text import extract_last_number


def compute_accuracy(predictions: list[str], references: list[str]) -> float:
    """
    Exact-match accuracy after extracting the last number from each prediction.

    Args:
        predictions: list of raw model output strings
        references:  list of gold answer strings (already extracted numbers)

    Returns:
        Fraction in [0, 1]
    """
    if not predictions:
        return 0.0
    correct = sum(
        extract_last_number(pred) == ref
        for pred, ref in zip(predictions, references)
    )
    return correct / len(predictions)


def compute_avg_length(texts: list[str]) -> float:
    """Average whitespace-tokenized length of generated texts."""
    if not texts:
        return 0.0
    return sum(len(t.split()) for t in texts) / len(texts)


def majority_vote(answers: list[str | None]) -> str | None:
    """Return the most common answer (None entries excluded)."""
    valid = [a for a in answers if a is not None]
    if not valid:
        return None
    return Counter(valid).most_common(1)[0][0]

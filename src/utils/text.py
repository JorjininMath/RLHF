"""Text processing helpers for GSM8K answer extraction."""
from __future__ import annotations

import re


def extract_last_number(text: str) -> str | None:
    """Return the last number (int or float) found in *text*, or None."""
    # Match numbers with optional commas (1,000) and decimals
    matches = re.findall(r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?|-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def extract_gsm8k_answer(solution: str) -> str | None:
    """
    Extract the gold answer from a GSM8K solution string.
    GSM8K uses the convention: '#### <answer>' at the end.
    """
    match = re.search(r"####\s*(-?\d[\d,]*(?:\.\d+)?)", solution)
    if match:
        return match.group(1).replace(",", "")
    return extract_last_number(solution)

"""Prompt templates for all pipeline stages."""
from __future__ import annotations

SYSTEM_PROMPT = (
    "Solve the math problem step by step, then output the final numeric answer "
    "on the last line in the format: Answer: <number>"
)


def format_sft_input(question: str) -> str:
    """Plain text prompt used for SFT tokenization and RM scoring."""
    return f"Problem: {question}\nAnswer:"


def format_chat_prompt(question: str, tokenizer) -> str:
    """
    Use the model's built-in chat template if available.
    Falls back to plain format_sft_input for models without a template.
    """
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        return format_sft_input(question)

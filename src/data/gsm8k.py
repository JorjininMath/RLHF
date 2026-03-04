"""GSM8K dataset loading with caching support."""
from __future__ import annotations

from typing import Optional

from datasets import Dataset, load_dataset

from src.utils.logging import get_logger

logger = get_logger(__name__)


def load_gsm8k(
    split: str,
    n: Optional[int] = None,
    cache_dir: str = "data/cache",
) -> Dataset:
    """
    Load GSM8K 'main' split from HuggingFace Hub with local caching.

    Args:
        split:     'train' or 'test'
        n:         Truncate to first *n* examples (None = full split).
        cache_dir: Local directory for dataset cache.

    Returns:
        HuggingFace Dataset with columns: question, answer
    """
    logger.info(f"Loading gsm8k split='{split}' n={n} cache_dir='{cache_dir}'")
    ds = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))
    logger.info(f"Loaded {len(ds)} examples")
    return ds

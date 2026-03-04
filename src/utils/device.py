"""Device resolution helper shared across all trainers."""
from __future__ import annotations

import torch


def resolve_device(cfg_device: str | None) -> tuple[str, torch.dtype]:
    """
    Return (device_str, dtype) based on config and what's available.

    Priority: explicit 'cpu' > CUDA > MPS > CPU fallback
    MPS uses float32 (bfloat16 not fully supported on MPS).
    """
    if cfg_device == "cpu":
        return "cpu", torch.float32

    if torch.cuda.is_available():
        return "cuda", torch.bfloat16

    if torch.backends.mps.is_available():
        return "mps", torch.float32   # MPS doesn't support bfloat16

    return "cpu", torch.float32

"""Structured logging utilities (console + JSONL metrics)."""
from __future__ import annotations

import json
import logging as _stdlib_logging
from datetime import datetime, timezone
from pathlib import Path


def get_logger(name: str) -> _stdlib_logging.Logger:
    logger = _stdlib_logging.getLogger(name)
    if not logger.handlers:
        handler = _stdlib_logging.StreamHandler()
        handler.setFormatter(
            _stdlib_logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(_stdlib_logging.INFO)
    return logger


class JsonlMetricsLogger:
    """Appends one JSON line per metric event to logs/<stage>_<run_id>.jsonl."""

    def __init__(self, log_dir: str, stage: str, run_id: str) -> None:
        self.path = Path(log_dir) / f"{stage}_{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, metrics: dict) -> None:
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **metrics}
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

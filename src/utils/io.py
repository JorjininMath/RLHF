"""Config loading, run-id generation, and JSONL helpers."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def generate_run_id() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


def load_config(base_path: str, stage_path: str, overrides: dict[str, Any] | None = None) -> dict:
    """
    Merge base.yaml + stage YAML + CLI overrides.
    Stage values override base; CLI overrides win over everything.
    """
    with open(base_path) as f:
        cfg: dict = yaml.safe_load(f)

    with open(stage_path) as f:
        stage_cfg: dict = yaml.safe_load(f)

    # Deep-merge lora sub-dict; shallow-merge everything else
    for k, v in stage_cfg.items():
        if k == "lora" and isinstance(v, dict) and isinstance(cfg.get("lora"), dict):
            cfg["lora"].update(v)
        else:
            cfg[k] = v

    if overrides:
        for k, v in overrides.items():
            cfg[k] = v

    cfg.setdefault("run_id", generate_run_id())
    return cfg


def save_jsonl(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

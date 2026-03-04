"""
Mini-RLHF-PPO command-line interface.

Usage:
    python -m src.cli sft  --config configs/sft.yaml
    python -m src.cli rm   --config configs/rm.yaml  --override sft_checkpoint=outputs/sft/xxx
    python -m src.cli ppo  --config configs/ppo.yaml --override sft_checkpoint=... rm_checkpoint=...
    python -m src.cli eval --config configs/eval.yaml --override sft_checkpoint=... ppo_checkpoint=...
    python -m src.cli report --config configs/eval.yaml
"""
from __future__ import annotations

import argparse
import sys

from src.utils.io import load_config


# ── Type coercion for CLI overrides ──────────────────────────────────────────

def _cast(value: str) -> int | float | bool | str:
    """Try to parse a string as int, float, bool, or leave as str."""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_overrides(override_list: list[str]) -> dict:
    result = {}
    for item in override_list:
        if "=" not in item:
            print(f"Warning: ignoring malformed override '{item}' (expected key=value)", file=sys.stderr)
            continue
        k, v = item.split("=", 1)
        result[k.strip()] = _cast(v.strip())
    return result


# ── Sub-command runners ───────────────────────────────────────────────────────

def _run_sft(cfg: dict) -> None:
    from src.trainers.sft import run_sft
    ckpt = run_sft(cfg)
    print(f"SFT checkpoint: {ckpt}")


def _run_rm(cfg: dict) -> None:
    from src.trainers.reward_model import run_reward_model
    ckpt = run_reward_model(cfg)
    print(f"RM checkpoint: {ckpt}")


def _run_ppo(cfg: dict) -> None:
    from src.trainers.ppo import run_ppo
    ckpt = run_ppo(cfg)
    print(f"PPO checkpoint: {ckpt}")


def _run_eval(cfg: dict) -> None:
    from src.eval.evaluate import run_evaluation
    results = run_evaluation(cfg)
    print("\n=== Evaluation Results ===")
    for model_name, metrics in results.items():
        print(f"  [{model_name}] accuracy={metrics['accuracy']:.3f}  avg_len={metrics['avg_length']:.1f}")


def _run_report(cfg: dict) -> None:
    from src.report.make_report import generate_report
    generate_report(cfg)
    print("Report → reports/weekly_report.md")


_RUNNERS = {
    "sft":    _run_sft,
    "rm":     _run_rm,
    "ppo":    _run_ppo,
    "eval":   _run_eval,
    "report": _run_report,
}

_DEFAULT_CONFIGS = {
    "sft":    "configs/sft.yaml",
    "rm":     "configs/rm.yaml",
    "ppo":    "configs/ppo.yaml",
    "eval":   "configs/eval.yaml",
    "report": "configs/eval.yaml",
}


# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.cli",
        description="Mini-RLHF-PPO pipeline runner",
    )
    parser.add_argument(
        "stage",
        choices=list(_RUNNERS),
        help="Pipeline stage to run",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to stage YAML config (default: configs/<stage>.yaml)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="key=value",
        help="Override a config key (can be repeated)",
    )

    args = parser.parse_args(argv)

    stage_config = args.config or _DEFAULT_CONFIGS[args.stage]
    overrides = _parse_overrides(args.override)

    cfg = load_config("configs/base.yaml", stage_config, overrides)

    import os
    if cfg.get("wandb_disabled", True):
        os.environ.setdefault("WANDB_DISABLED", "true")

    _RUNNERS[args.stage](cfg)


if __name__ == "__main__":
    main()

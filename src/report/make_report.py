"""Auto-generate reports/weekly_report.md from logs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from src.utils.io import load_jsonl
from src.utils.logging import get_logger

logger = get_logger(__name__)

_TEMPLATE_DIR = Path(__file__).parent
_TEMPLATE_FILE = "template.md.j2"


def _last_metric(log_dir: str, stage: str, event: str, key: str) -> float | None:
    """Scan all JSONL files for the last matching metric."""
    log_path = Path(log_dir)
    for jsonl_file in sorted(log_path.glob(f"{stage}_*.jsonl"), reverse=True):
        records = load_jsonl(jsonl_file)
        for rec in reversed(records):
            if rec.get("event") == event and key in rec:
                return rec[key]
    return None


def _last_metrics_avg(log_dir: str, stage: str, event: str, keys: list[str], n: int = 20) -> dict:
    """Average the last *n* step records for each key."""
    log_path = Path(log_dir)
    results: dict[str, float | None] = {k: None for k in keys}

    for jsonl_file in sorted(log_path.glob(f"{stage}_*.jsonl"), reverse=True):
        records = [r for r in load_jsonl(jsonl_file) if r.get("event") == event]
        if not records:
            continue
        tail = records[-n:]
        for k in keys:
            vals = [r[k] for r in tail if k in r and isinstance(r[k], (int, float))]
            if vals:
                results[k] = sum(vals) / len(vals)
        break

    return results


def _build_commentary(eval_rows: list[dict]) -> str:
    if not eval_rows:
        return "No evaluation results available."

    acc_map = {r["model"]: r["accuracy"] for r in eval_rows}
    base_acc = acc_map.get("base", 0.0)
    sft_acc  = acc_map.get("sft",  None)
    ppo_acc  = acc_map.get("ppo",  None)

    lines = []
    if sft_acc is not None:
        delta = (sft_acc - base_acc) * 100
        direction = "improved" if delta >= 0 else "dropped"
        lines.append(
            f"SFT {direction} accuracy by **{abs(delta):.1f}pp** "
            f"(base: {base_acc*100:.1f}% → SFT: {sft_acc*100:.1f}%)."
        )
    if ppo_acc is not None and sft_acc is not None:
        delta = (ppo_acc - sft_acc) * 100
        direction = "further improved" if delta >= 0 else "regressed"
        lines.append(
            f"PPO {direction} over SFT by **{abs(delta):.1f}pp** "
            f"(SFT: {sft_acc*100:.1f}% → PPO: {ppo_acc*100:.1f}%)."
        )
    if not lines:
        return "Evaluation complete. See table above for results."

    lines.append(
        "Note: small dataset size means these numbers have high variance. "
        "Scaled to the full GSM8K train set, results would be more stable."
    )
    return " ".join(lines)


def generate_report(cfg: dict) -> None:
    """Read logs and render reports/weekly_report.md via Jinja2 template."""
    log_dir    = cfg.get("log_dir", "logs/")
    output_dir = cfg.get("output_dir", "outputs/")
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    # ── Collect metrics from logs ─────────────────────────────────────────────
    sft_loss   = _last_metric(log_dir, "sft", "complete", "train_loss")
    rm_loss    = _last_metric(log_dir, "rm",  "epoch",    "train_loss")
    rm_val_acc = _last_metric(log_dir, "rm",  "epoch",    "val_acc") or 0.0

    ppo_avgs   = _last_metrics_avg(log_dir, "ppo", "step", ["mean_reward", "mean_kl"])
    ppo_mean_reward = ppo_avgs.get("mean_reward")
    ppo_mean_kl     = ppo_avgs.get("mean_kl")

    # ── Collect eval results ──────────────────────────────────────────────────
    eval_rows: list[dict] = []
    for result_file in sorted(Path(output_dir).glob("eval/*/results.json"), reverse=True):
        with open(result_file) as f:
            data: dict = json.load(f)
        for model_name, metrics in data.items():
            eval_rows.append({
                "model":      model_name,
                "accuracy":   metrics.get("accuracy", 0.0),
                "avg_length": metrics.get("avg_length", 0.0),
            })
        break   # use only the most recent eval run

    commentary = _build_commentary(eval_rows)

    # ── Render ────────────────────────────────────────────────────────────────
    env = Environment(loader=FileSystemLoader(str(_TEMPLATE_DIR)))
    template = env.get_template(_TEMPLATE_FILE)

    rendered = template.render(
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        config=cfg,
        sft_loss=sft_loss,
        rm_loss=rm_loss,
        rm_val_acc=rm_val_acc,
        ppo_mean_reward=ppo_mean_reward,
        ppo_mean_kl=ppo_mean_kl,
        eval_rows=eval_rows,
        commentary=commentary,
    )

    out_file = report_dir / "weekly_report.md"
    out_file.write_text(rendered)
    logger.info(f"Report written to {out_file}")

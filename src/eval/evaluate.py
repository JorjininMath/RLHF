"""Unified evaluation script: base vs SFT vs PPO."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.gsm8k import load_gsm8k
from src.data.prompts import format_chat_prompt
from src.eval.metrics import compute_accuracy, compute_avg_length
from src.utils.device import resolve_device
from src.utils.io import generate_run_id
from src.utils.logging import JsonlMetricsLogger, get_logger
from src.utils.seed import set_seed
from src.utils.text import extract_gsm8k_answer

logger = get_logger(__name__)


def _load_model_and_tokenizer(
    model_path: str,
    base_name: str,
    is_peft: bool,
    device: str,
    dtype: torch.dtype,
) -> tuple:
    """
    Load a model for evaluation.
    For PEFT checkpoints, loads base model + adapter then merges.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_peft:
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(
            base_name, torch_dtype=dtype, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload()   # merge LoRA into base for clean inference
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, trust_remote_code=True
        )

    model.eval()
    model.to(device)
    return model, tokenizer


@torch.no_grad()
def _evaluate_single_model(
    model,
    tokenizer,
    dataset,
    cfg: dict,
    device: str,
) -> dict:
    """
    Run greedy-decode evaluation on *dataset*.

    Returns:
        dict with keys: accuracy, avg_length, predictions (list)
    """
    max_new_tokens: int = cfg.get("max_new_tokens", 128)
    predictions: list[str] = []
    references: list[str] = []

    for example in tqdm(dataset, desc="Evaluating", leave=False):
        prompt = format_chat_prompt(example["question"], tokenizer)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,          # greedy for determinism
            pad_token_id=tokenizer.pad_token_id,
        )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(new_ids, skip_special_tokens=True)

        predictions.append(response)
        references.append(extract_gsm8k_answer(example["answer"]) or "")

    return {
        "accuracy":    compute_accuracy(predictions, references),
        "avg_length":  compute_avg_length(predictions),
        "predictions": predictions,
    }


def run_evaluation(cfg: dict) -> dict:
    """
    Evaluate base, SFT, and (optionally) PPO models on GSM8K.

    Returns:
        dict mapping model name → metrics
    """
    set_seed(cfg["seed"])
    run_id: str = cfg["run_id"]
    metrics_logger = JsonlMetricsLogger(cfg["log_dir"], "eval", run_id)

    device, dtype = resolve_device(cfg.get("device"))

    dataset = load_gsm8k("test", n=cfg.get("n_eval"), cache_dir=cfg.get("cache_dir", "data/cache"))
    base_name: str = cfg["model_name"]

    results: dict[str, dict] = {}

    # ── Base model ────────────────────────────────────────────────────────────
    logger.info("Evaluating base model…")
    base_model, base_tok = _load_model_and_tokenizer(
        base_name, base_name, is_peft=False, device=device, dtype=dtype
    )
    results["base"] = _evaluate_single_model(base_model, base_tok, dataset, cfg, device)
    del base_model
    if device == "cuda":
        torch.cuda.empty_cache()

    # ── SFT model ─────────────────────────────────────────────────────────────
    sft_ckpt: Optional[str] = cfg.get("sft_checkpoint")
    if sft_ckpt and Path(sft_ckpt).exists():
        logger.info(f"Evaluating SFT model from {sft_ckpt}…")
        sft_model, sft_tok = _load_model_and_tokenizer(
            sft_ckpt, base_name, is_peft=True, device=device, dtype=dtype
        )
        results["sft"] = _evaluate_single_model(sft_model, sft_tok, dataset, cfg, device)
        del sft_model
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        logger.warning("SFT checkpoint not found; skipping SFT evaluation.")

    # ── PPO model ─────────────────────────────────────────────────────────────
    ppo_ckpt: Optional[str] = cfg.get("ppo_checkpoint")
    if ppo_ckpt and Path(ppo_ckpt).exists():
        logger.info(f"Evaluating PPO model from {ppo_ckpt}…")
        ppo_model, ppo_tok = _load_model_and_tokenizer(
            ppo_ckpt, base_name, is_peft=True, device=device, dtype=dtype
        )
        results["ppo"] = _evaluate_single_model(ppo_model, ppo_tok, dataset, cfg, device)
        del ppo_model
        if device == "cuda":
            torch.cuda.empty_cache()
    else:
        logger.warning("PPO checkpoint not found; skipping PPO evaluation.")

    # ── Log and return ────────────────────────────────────────────────────────
    for model_name, metrics in results.items():
        safe = {k: v for k, v in metrics.items() if k != "predictions"}
        metrics_logger.log({"event": "result", "model": model_name, **safe})
        logger.info(
            f"[{model_name}] accuracy={metrics['accuracy']:.3f} "
            f"avg_len={metrics['avg_length']:.1f}"
        )

    # Save full results to JSON
    out_file = Path(cfg.get("output_dir", "outputs")) / "eval" / run_id / "results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(
            {k: {m: v for m, v in v.items() if m != "predictions"} for k, v in results.items()},
            f, indent=2,
        )

    return results

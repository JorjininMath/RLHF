"""PPO training with KL regularization using TRL."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from packaging import version

from src.data.gsm8k import load_gsm8k
from src.data.prompts import format_chat_prompt
from src.trainers.reward_model import load_reward_model, score_response
from src.utils.io import generate_run_id
from src.utils.logging import JsonlMetricsLogger, get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def _check_trl() -> tuple[Any, Any, Any, str]:
    """Import TRL components; return (PPOConfig, PPOTrainer, ValueHeadModel, trl_version)."""
    try:
        import trl
        from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer
    except ImportError as e:
        raise ImportError(
            "TRL is required for PPO training. "
            "Install with: pip install 'trl>=0.9,<1.0'"
        ) from e

    trl_ver = trl.__version__
    logger.info(f"TRL version: {trl_ver}")

    if version.parse(trl_ver) < version.parse("0.9.0"):
        raise RuntimeError(
            f"TRL >= 0.9.0 required, found {trl_ver}. "
            "Upgrade with: pip install 'trl>=0.9,<1.0'"
        )

    return PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead, trl_ver


def run_ppo(cfg: dict) -> str:
    """
    Run PPO fine-tuning with KL regularization.

    Requires:
        cfg['sft_checkpoint'] — path to the SFT PEFT adapter directory
        cfg['rm_checkpoint']  — path to the trained RM directory

    Returns:
        Path to the saved PPO adapter directory.
    """
    PPOConfig, PPOTrainer, ValueHeadModel, trl_ver = _check_trl()
    from peft import LoraConfig

    set_seed(cfg["seed"])
    run_id: str = cfg["run_id"]
    output_path = Path(cfg["output_dir"]) / "ppo" / run_id
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_logger = JsonlMetricsLogger(cfg["log_dir"], "ppo", run_id)
    metrics_logger.log({"event": "start", "trl_version": trl_ver})
    logger.info(f"PPO | run_id={run_id} | trl={trl_ver}")

    sft_checkpoint: str = cfg.get("sft_checkpoint") or ""
    rm_checkpoint:  str = cfg.get("rm_checkpoint")  or ""
    if not sft_checkpoint or not Path(sft_checkpoint).exists():
        raise FileNotFoundError(f"SFT checkpoint not found: '{sft_checkpoint}'")
    if not rm_checkpoint or not Path(rm_checkpoint).exists():
        raise FileNotFoundError(f"RM checkpoint not found: '{rm_checkpoint}'")

    # ── Resolve device ────────────────────────────────────────────────────────
    if cfg.get("device") == "cpu" or not torch.cuda.is_available():
        device = "cpu"
        dtype = torch.float32
    else:
        device = "cuda"
        dtype = torch.bfloat16

    # ── Load tokenizer from SFT checkpoint ───────────────────────────────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Load policy (SFT PEFT model + value head + new LoRA for PPO) ─────────
    #   AutoModelForCausalLMWithValueHead auto-detects the PEFT adapter
    #   when the checkpoint directory contains adapter_config.json.
    lora_cfg = cfg.get("lora", {})
    ppo_lora = LoraConfig(
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias="none",
    )
    policy = ValueHeadModel.from_pretrained(
        sft_checkpoint,
        peft_config=ppo_lora,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    policy.to(device)

    # ── Load reference model (frozen SFT weights) ─────────────────────────────
    ref_model = ValueHeadModel.from_pretrained(
        sft_checkpoint,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    ref_model.to(device)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── Load reward model ─────────────────────────────────────────────────────
    rm_model, rm_tokenizer = load_reward_model(rm_checkpoint)
    rm_model = rm_model.to(device)
    rm_model.eval()

    # ── Build query dataset ───────────────────────────────────────────────────
    raw_ds = load_gsm8k(
        "train",
        n=cfg.get("n_ppo_steps", 200) * cfg.get("batch_size", 16),
        cache_dir=cfg.get("cache_dir", "data/cache"),
    )

    def tokenize_query(example: dict) -> dict:
        prompt = format_chat_prompt(example["question"], tokenizer)
        ids = tokenizer(prompt, truncation=True, max_length=256, return_tensors=None)["input_ids"]
        return {"input_ids": ids, "query": prompt}

    query_ds = raw_ds.map(tokenize_query, remove_columns=raw_ds.column_names)

    # ── PPO config ────────────────────────────────────────────────────────────
    ppo_config = PPOConfig(
        batch_size=cfg.get("batch_size", 16),
        mini_batch_size=cfg.get("mini_batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        ppo_epochs=cfg.get("ppo_epochs", 4),
        learning_rate=cfg.get("learning_rate", 1e-5),
        init_kl_coef=cfg.get("kl_coef", 0.1),
        kl_penalty="kl",
        seed=cfg["seed"],
        log_with=None,
    )

    def collate(data: list[dict]) -> dict:
        return {k: [d[k] for d in data] for k in data[0]}

    trainer = PPOTrainer(
        config=ppo_config,
        model=policy,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=query_ds,
        data_collator=collate,
    )

    # ── PPO training loop ─────────────────────────────────────────────────────
    gen_kwargs = dict(
        max_new_tokens=cfg.get("max_new_tokens", 128),
        do_sample=True,
        temperature=0.9,
        pad_token_id=tokenizer.pad_token_id,
    )

    n_steps = cfg.get("n_ppo_steps", 200)
    for step, batch in enumerate(trainer.dataloader):
        if step >= n_steps:
            break

        query_tensors: list[torch.Tensor] = [
            torch.tensor(ids, device=device) for ids in batch["input_ids"]
        ]

        # Generate responses
        response_tensors = trainer.generate(
            query_tensors,
            return_prompt=False,
            **gen_kwargs,
        )

        # Decode for RM scoring
        queries_text   = tokenizer.batch_decode(query_tensors,   skip_special_tokens=True)
        responses_text = tokenizer.batch_decode(response_tensors, skip_special_tokens=True)

        rewards = [
            torch.tensor(
                score_response(q, r, rm_model, rm_tokenizer, device=device),
                dtype=torch.float32,
                device=device,
            )
            for q, r in zip(queries_text, responses_text)
        ]

        # PPO step
        stats = trainer.step(query_tensors, response_tensors, rewards)

        mean_reward = torch.stack(rewards).mean().item()
        mean_kl     = stats.get("objective/kl", float("nan"))

        metrics_logger.log({
            "event": "step",
            "step": step,
            "mean_reward": mean_reward,
            "mean_kl": mean_kl,
        })

        if step % 20 == 0:
            logger.info(f"PPO step {step}/{n_steps} | reward={mean_reward:.4f} kl={mean_kl:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    trainer.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    metrics_logger.log({"event": "complete"})
    logger.info(f"PPO complete → {output_path}")
    return str(output_path)

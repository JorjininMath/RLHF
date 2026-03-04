"""
PPO-lite: REINFORCE with KL regularization.

Does NOT use TRL's PPOTrainer (avoids version compatibility issues across
trl 0.9/0.10/0.11/0.12 which all have breaking API changes).

Full RLHF objective implemented explicitly:

    r_total(x, y) = r_RM(x, y) − β · Σ_t log[π_θ(t|x,y<t) / π_ref(t|x,y<t)]

    L = −r_RM · Σ_t log π_θ(t|x,y<t)         ← REINFORCE signal
      + β · Σ_t [log π_θ(t|x,y<t) − log π_ref(t|x,y<t)]   ← KL penalty

This makes every term in the loss function visible and directly matches
the theory described in docs/rlhf_blog.md §5.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data.gsm8k import load_gsm8k
from src.data.prompts import format_chat_prompt
from src.trainers.reward_model import load_reward_model, score_response
from src.utils.device import resolve_device
from src.utils.logging import JsonlMetricsLogger, get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _token_log_probs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    response_start: int,
) -> torch.Tensor:
    """
    Compute per-token log probabilities for the *response* portion of input_ids.

    Args:
        model:          causal LM (policy or ref)
        input_ids:      shape (1, T) — full prompt + response
        response_start: index where response tokens begin

    Returns:
        Tensor of shape (n_response_tokens,) — log P(token_t | context)
    """
    with torch.set_grad_enabled(model.training):
        logits = model(input_ids).logits               # (1, T, V)

    # Shift: logits[t] predicts token[t+1]
    shift_logits  = logits[:, response_start - 1 : -1, :]   # (1, R, V)
    response_ids  = input_ids[:, response_start:]            # (1, R)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_lp  = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1).squeeze(0)  # (R,)
    return token_lp


# ── Main training function ────────────────────────────────────────────────────

def run_ppo(cfg: dict) -> str:
    """
    Train the policy with REINFORCE + KL regularization.

    Returns:
        Path to the saved PEFT checkpoint directory.
    """
    set_seed(cfg["seed"])
    run_id: str = cfg["run_id"]
    output_path = Path(cfg["output_dir"]) / "ppo" / run_id
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_logger = JsonlMetricsLogger(cfg["log_dir"], "ppo", run_id)
    metrics_logger.log({"event": "start"})

    sft_checkpoint: str = cfg.get("sft_checkpoint") or ""
    rm_checkpoint:  str = cfg.get("rm_checkpoint")  or ""
    if not sft_checkpoint or not Path(sft_checkpoint).exists():
        raise FileNotFoundError(f"SFT checkpoint not found: '{sft_checkpoint}'")
    if not rm_checkpoint or not Path(rm_checkpoint).exists():
        raise FileNotFoundError(f"RM checkpoint not found: '{rm_checkpoint}'")

    # ── Device / dtype ────────────────────────────────────────────────────────
    device, dtype = resolve_device(cfg.get("device"))
    logger.info(f"PPO | run_id={run_id} | device={device}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(sft_checkpoint, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Policy (SFT PEFT checkpoint, trainable LoRA weights only) ────────────
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=dtype, trust_remote_code=True
    )
    policy = PeftModel.from_pretrained(base, sft_checkpoint)
    policy.to(device)
    for name, param in policy.named_parameters():
        param.requires_grad_("lora" in name.lower())
    policy.train()
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    logger.info(f"Policy trainable params: {n_trainable:,}")

    # ── Reference model (frozen SFT) ──────────────────────────────────────────
    base_ref = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=dtype, trust_remote_code=True
    )
    ref_model = PeftModel.from_pretrained(base_ref, sft_checkpoint)
    ref_model.to(device).eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # ── Reward model ──────────────────────────────────────────────────────────
    rm_model, rm_tokenizer = load_reward_model(rm_checkpoint)
    rm_model.to(device).eval()

    # ── Dataset ───────────────────────────────────────────────────────────────
    n_steps    = cfg.get("n_ppo_steps", 200)
    batch_size = cfg.get("batch_size", 4)
    dataset    = load_gsm8k(
        "train",
        n=min(n_steps * batch_size, 7473),
        cache_dir=cfg.get("cache_dir", "data/cache"),
    )
    n_examples = len(dataset)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=cfg.get("learning_rate", 1e-5),
    )

    beta: float  = cfg.get("kl_coef", 0.1)
    max_new: int = cfg.get("max_new_tokens", 128)

    # ── Training loop ─────────────────────────────────────────────────────────
    for step in range(n_steps):
        optimizer.zero_grad()

        step_reward = 0.0
        step_kl     = 0.0
        step_loss   = torch.zeros(1, device=device)

        indices = [(step * batch_size + i) % n_examples for i in range(batch_size)]

        for idx in indices:
            example = dataset[idx]
            prompt  = format_chat_prompt(example["question"], tokenizer)
            input_ids = tokenizer(
                prompt, return_tensors="pt", truncation=True, max_length=256,
            ).input_ids.to(device)                    # (1, L_prompt)
            response_start = input_ids.shape[1]

            # Generate response — sampling, no grad
            with torch.no_grad():
                output_ids = policy.generate(
                    input_ids,
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )                                     # (1, L_prompt + L_resp)

            if output_ids.shape[1] <= response_start:
                continue   # empty response, skip

            response_text = tokenizer.decode(
                output_ids[0, response_start:], skip_special_tokens=True
            )

            # ── RM reward (scalar, no grad) ───────────────────────────────────
            r_rm = score_response(
                prompt, response_text, rm_model, rm_tokenizer,
                max_length=256, device=device,
            )

            # ── Token log-probs ───────────────────────────────────────────────
            # Policy: gradients flow
            policy_lp = _token_log_probs(policy, output_ids, response_start)

            # Reference: no gradient
            with torch.no_grad():
                ref_lp = _token_log_probs(ref_model, output_ids, response_start)

            # ── Loss = REINFORCE + KL ─────────────────────────────────────────
            #   −r_RM · Σ log π_θ  →  push toward high-reward responses
            #   + β · Σ(log π_θ − log π_ref)  →  stay close to SFT
            kl        = (policy_lp - ref_lp.detach()).sum()
            reinforce = -torch.tensor(r_rm, device=device, dtype=torch.float32) * policy_lp.sum()
            loss_i    = (reinforce + beta * kl) / batch_size

            step_loss   = step_loss + loss_i
            step_reward += r_rm
            step_kl     += kl.detach().item()

        step_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in policy.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()

        mean_reward = step_reward / batch_size
        mean_kl     = step_kl     / batch_size
        metrics_logger.log({
            "event":       "step",
            "step":        step,
            "mean_reward": mean_reward,
            "mean_kl":     mean_kl,
            "loss":        step_loss.item(),
        })
        if step % max(1, n_steps // 10) == 0:
            logger.info(
                f"PPO step {step}/{n_steps} | "
                f"reward={mean_reward:.4f}  kl={mean_kl:.4f}  loss={step_loss.item():.4f}"
            )

    # ── Save ──────────────────────────────────────────────────────────────────
    policy.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    metrics_logger.log({"event": "complete"})
    logger.info(f"PPO complete → {output_path}")
    return str(output_path)

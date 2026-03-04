"""Supervised Fine-Tuning (SFT) with LoRA/PEFT."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.utils.rnn import pad_sequence
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.data.gsm8k import load_gsm8k
from src.data.prompts import format_sft_input
from src.utils.io import generate_run_id
from src.utils.logging import JsonlMetricsLogger, get_logger
from src.utils.seed import set_seed
from src.utils.text import extract_gsm8k_answer

logger = get_logger(__name__)

# ── LoRA target modules per model family ──────────────────────────────────────

_LORA_TARGETS: dict[str, list[str]] = {
    "qwen":    ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "llama":   ["q_proj", "v_proj", "k_proj", "o_proj"],
    "mistral": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "gpt2":    ["c_attn", "c_proj"],
}
_DEFAULT_TARGETS = ["q_proj", "v_proj"]


def _lora_targets(model_name: str) -> list[str]:
    name = model_name.lower()
    for key, modules in _LORA_TARGETS.items():
        if key in name:
            return modules
    return _DEFAULT_TARGETS


# ── Dataset preparation ───────────────────────────────────────────────────────

def _build_sft_dataset(raw_ds, tokenizer, cfg: dict):
    """
    Tokenize each example as:  <prompt> <answer> <eos>
    Labels mask the prompt tokens with -100 so loss is computed only on the answer.
    """
    max_length: int = cfg.get("max_length", 512)

    def preprocess(example: dict) -> dict:
        prompt = format_sft_input(example["question"])
        answer = extract_gsm8k_answer(example["answer"]) or ""
        full_text = f"{prompt} {answer}{tokenizer.eos_token}"

        tokenized = tokenizer(
            full_text,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        # Mask prompt portion in labels
        n_prompt = len(
            tokenizer(prompt, add_special_tokens=False, return_tensors=None)["input_ids"]
        )
        labels = list(tokenized["input_ids"])
        for i in range(min(n_prompt, len(labels))):
            labels[i] = -100
        tokenized["labels"] = labels
        return tokenized

    return raw_ds.map(preprocess, remove_columns=raw_ds.column_names)


class _SFTCollator:
    """Pads input_ids, attention_mask, and labels in a batch."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_ids  = [torch.tensor(f["input_ids"])      for f in features]
        attn_mask  = [torch.tensor(f["attention_mask"])  for f in features]
        labels     = [torch.tensor(f["labels"])          for f in features]

        return {
            "input_ids":      pad_sequence(input_ids, batch_first=True, padding_value=self.pad_id),
            "attention_mask": pad_sequence(attn_mask, batch_first=True, padding_value=0),
            "labels":         pad_sequence(labels,    batch_first=True, padding_value=-100),
        }


# ── Main training function ────────────────────────────────────────────────────

def run_sft(cfg: dict) -> str:
    """
    Train the policy with supervised fine-tuning + LoRA.

    Returns:
        Path to the saved PEFT checkpoint directory.
    """
    set_seed(cfg["seed"])
    run_id: str = cfg["run_id"]
    output_path = Path(cfg["output_dir"]) / "sft" / run_id
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_logger = JsonlMetricsLogger(cfg["log_dir"], "sft", run_id)
    metrics_logger.log({"event": "start", "config": cfg})
    logger.info(f"SFT | run_id={run_id} | model={cfg['model_name']}")

    # Resolve device
    if cfg.get("device") == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["model_name"],
        cache_dir=cfg.get("cache_dir"),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        cache_dir=cfg.get("cache_dir"),
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_cfg = cfg.get("lora", {})
    target_modules = lora_cfg.get("target_modules") or _lora_targets(cfg["model_name"])
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg.get("r", 8),
        lora_alpha=lora_cfg.get("alpha", 16),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Datasets
    train_ds = load_gsm8k("train", n=cfg.get("n_train"), cache_dir=cfg.get("cache_dir", "data/cache"))
    eval_ds  = load_gsm8k("test",  n=cfg.get("n_eval"),  cache_dir=cfg.get("cache_dir", "data/cache"))
    train_ds = _build_sft_dataset(train_ds, tokenizer, cfg)
    eval_ds  = _build_sft_dataset(eval_ds,  tokenizer, cfg)

    use_bf16 = (dtype == torch.bfloat16)
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=cfg.get("num_epochs", 3),
        max_steps=cfg.get("max_steps", -1),          # -1 = use num_epochs
        per_device_train_batch_size=cfg.get("batch_size", 4),
        per_device_eval_batch_size=cfg.get("batch_size", 4),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
        learning_rate=cfg.get("learning_rate", 2e-4),
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        fp16=False,
        bf16=use_bf16,
        report_to="none",
        seed=cfg["seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=_SFTCollator(tokenizer.pad_token_id),
    )

    train_result = trainer.train()

    # Save PEFT adapter + tokenizer
    model.save_pretrained(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    metrics_logger.log({
        "event": "complete",
        "train_loss": train_result.training_loss,
        "steps": train_result.global_step,
    })
    logger.info(f"SFT complete → {output_path}")
    return str(output_path)

"""Reward Model training with Bradley-Terry pairwise ranking loss."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from src.data.gsm8k import load_gsm8k
from src.data.prompts import format_sft_input
from src.utils.io import load_jsonl
from src.utils.logging import JsonlMetricsLogger, get_logger
from src.utils.seed import set_seed
from src.utils.text import extract_gsm8k_answer, extract_last_number

logger = get_logger(__name__)


# ── Model ─────────────────────────────────────────────────────────────────────

class RewardModel(nn.Module):
    """
    DistilBERT backbone + scalar regression head.
    Outputs a single reward score r ∈ ℝ for a (prompt, response) pair.
    """

    def __init__(self, backbone_name: str = "distilbert-base-uncased") -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        self.head = nn.Linear(self.backbone.config.hidden_size, 1)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Returns shape (batch,) reward scores."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]   # [CLS] token representation
        return self.head(cls).squeeze(-1)


# ── Dataset ───────────────────────────────────────────────────────────────────

class PairwiseDataset(Dataset):
    """Each item is (chosen, rejected) encoded for the RM."""

    def __init__(self, pairs: list[dict], tokenizer, max_length: int = 256) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        pair = self.pairs[idx]
        prompt, chosen, rejected = pair["prompt"], pair["chosen"], pair["rejected"]

        def enc(text: str) -> dict[str, torch.Tensor]:
            return self.tokenizer(
                prompt, text,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )

        c = enc(chosen)
        r = enc(rejected)
        return {
            "chosen_input_ids":       c["input_ids"].squeeze(0),
            "chosen_attention_mask":  c["attention_mask"].squeeze(0),
            "rejected_input_ids":     r["input_ids"].squeeze(0),
            "rejected_attention_mask": r["attention_mask"].squeeze(0),
        }


# ── Preference data generation ────────────────────────────────────────────────

def _generate_rm_pairs(sft_checkpoint: str, cfg: dict) -> list[dict]:
    """
    Sample two responses per question from the SFT policy.
    Label: correct > incorrect (skip ties).
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer as _Tok

    logger.info("Generating RM preference pairs from SFT model (this may take a while)…")

    tok = _Tok.from_pretrained(sft_checkpoint, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.float32, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, sft_checkpoint)
    model.eval()
    device = "cpu"
    model.to(device)

    dataset = load_gsm8k(
        "train",
        n=cfg.get("n_rm_pairs", 400),
        cache_dir=cfg.get("cache_dir", "data/cache"),
    )

    pairs: list[dict] = []
    gen_kwargs = dict(
        max_new_tokens=cfg.get("max_new_tokens", 64),
        do_sample=True,
        temperature=cfg.get("temperature", 0.8),
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )

    for example in dataset:
        question = example["question"]
        gold = extract_gsm8k_answer(example["answer"])
        prompt = format_sft_input(question)
        input_ids = tok(prompt, return_tensors="pt").input_ids.to(device)

        responses = []
        for _ in range(2):
            with torch.no_grad():
                out = model.generate(input_ids, **gen_kwargs)
            resp = tok.decode(out[0][input_ids.shape[1]:], skip_special_tokens=True)
            responses.append(resp)

        correct = [int(extract_last_number(r) == gold) for r in responses]
        if correct[0] == correct[1]:
            continue   # skip ties

        chosen, rejected = (
            (responses[0], responses[1]) if correct[0] > correct[1]
            else (responses[1], responses[0])
        )
        pairs.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    logger.info(f"Generated {len(pairs)} preference pairs")
    return pairs


# ── Training ──────────────────────────────────────────────────────────────────

def run_reward_model(cfg: dict) -> str:
    """
    Train the reward model.

    Returns:
        Path to the saved RM checkpoint directory.
    """
    set_seed(cfg["seed"])
    run_id: str = cfg["run_id"]
    output_path = Path(cfg["output_dir"]) / "rm" / run_id
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_logger = JsonlMetricsLogger(cfg["log_dir"], "rm", run_id)
    metrics_logger.log({"event": "start"})

    # ── Load or generate pairs ──────────────────────────────────────────────
    rm_data_path = cfg.get("rm_data_path", "data/rm_pairs.jsonl")
    if Path(rm_data_path).exists():
        pairs = load_jsonl(rm_data_path)
        logger.info(f"Loaded {len(pairs)} RM pairs from {rm_data_path}")
    elif cfg.get("generate_rm_data") and cfg.get("sft_checkpoint"):
        pairs = _generate_rm_pairs(cfg["sft_checkpoint"], cfg)
        Path(rm_data_path).parent.mkdir(parents=True, exist_ok=True)
        with open(rm_data_path, "w") as f:
            for p in pairs:
                f.write(json.dumps(p) + "\n")
        logger.info(f"Saved {len(pairs)} pairs to {rm_data_path}")
    else:
        raise FileNotFoundError(
            f"No RM data at '{rm_data_path}'. "
            "Either provide data/rm_pairs.jsonl or set generate_rm_data: true "
            "with a valid sft_checkpoint."
        )

    if len(pairs) < 2:
        raise ValueError(f"Need at least 2 pairs for RM training, got {len(pairs)}.")

    # ── Setup ───────────────────────────────────────────────────────────────
    rm_backbone = cfg.get("rm_model_name", "distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(rm_backbone)
    model = RewardModel(rm_backbone)

    device = (
        "cuda" if torch.cuda.is_available() and cfg.get("device") != "cpu" else "cpu"
    )
    model = model.to(device)

    # Train/val split (90/10)
    n_val = max(1, len(pairs) // 10)
    train_pairs, val_pairs = pairs[n_val:], pairs[:n_val]

    max_length: int = cfg.get("max_length", 256)
    train_loader = DataLoader(
        PairwiseDataset(train_pairs, tokenizer, max_length),
        batch_size=cfg.get("batch_size", 16), shuffle=True,
    )
    val_loader = DataLoader(
        PairwiseDataset(val_pairs, tokenizer, max_length),
        batch_size=cfg.get("batch_size", 16),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("learning_rate", 2e-5))

    # ── Training loop ───────────────────────────────────────────────────────
    for epoch in range(cfg.get("num_epochs", 3)):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            sc = model(
                batch["chosen_input_ids"].to(device),
                batch["chosen_attention_mask"].to(device),
            )
            sr = model(
                batch["rejected_input_ids"].to(device),
                batch["rejected_attention_mask"].to(device),
            )
            # Bradley-Terry pairwise loss: -log σ(s_chosen - s_rejected)
            loss = -torch.log(torch.sigmoid(sc - sr) + 1e-8).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train = total_loss / max(len(train_loader), 1)

        # Validation accuracy (chosen should score higher than rejected)
        model.eval()
        correct = total_v = 0
        with torch.no_grad():
            for batch in val_loader:
                sc = model(batch["chosen_input_ids"].to(device), batch["chosen_attention_mask"].to(device))
                sr = model(batch["rejected_input_ids"].to(device), batch["rejected_attention_mask"].to(device))
                correct += (sc > sr).sum().item()
                total_v += sc.shape[0]
        acc = correct / max(total_v, 1)

        metrics_logger.log({"event": "epoch", "epoch": epoch, "train_loss": avg_train, "val_acc": acc})
        logger.info(f"RM epoch {epoch}: loss={avg_train:.4f} val_acc={acc:.3f}")

    # ── Save ────────────────────────────────────────────────────────────────
    torch.save(model.state_dict(), output_path / "rm_model.pt")
    tokenizer.save_pretrained(str(output_path))
    with open(output_path / "rm_config.json", "w") as f:
        json.dump({"rm_model_name": rm_backbone}, f)

    metrics_logger.log({"event": "complete"})
    logger.info(f"RM complete → {output_path}")
    return str(output_path)


# ── Inference helper ──────────────────────────────────────────────────────────

def load_reward_model(rm_checkpoint: str) -> tuple[RewardModel, Any]:
    """Load a saved RM for inference. Returns (model, tokenizer)."""
    import json
    cfg_path = Path(rm_checkpoint) / "rm_config.json"
    with open(cfg_path) as f:
        rm_cfg = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(rm_checkpoint)
    model = RewardModel(rm_cfg["rm_model_name"])
    model.load_state_dict(torch.load(Path(rm_checkpoint) / "rm_model.pt", map_location="cpu", weights_only=True))
    model.eval()
    return model, tokenizer


def score_response(
    prompt: str,
    response: str,
    rm_model: RewardModel,
    rm_tokenizer,
    max_length: int = 256,
    device: str = "cpu",
) -> float:
    """Return scalar reward score for a single (prompt, response) pair."""
    enc = rm_tokenizer(
        prompt, response,
        max_length=max_length, truncation=True,
        padding="max_length", return_tensors="pt",
    )
    with torch.no_grad():
        score = rm_model(
            enc["input_ids"].to(device),
            enc["attention_mask"].to(device),
        )
    return score.item()

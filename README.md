# mini-rlhf-ppo

A professional, reproducible implementation of the full RLHF post-training pipeline:
**SFT → Reward Model → PPO with KL Regularization → Evaluation**

Built for single-GPU (or CPU) training on GSM8K using small open-source LLMs.

---

## Resume Bullet

> Implemented end-to-end RLHF post-training pipeline (SFT with LoRA, Bradley-Terry reward model, PPO with KL-regularization) on GSM8K using Qwen2.5-0.5B; config-driven with YAML, run-id-based checkpointing, and JSONL metric logging. Achieved measurable accuracy improvement over base model with under 1K training examples.

---

## Architecture

```
Qwen2.5-0.5B ──► SFT (LoRA) ──► RM Training ──► PPO (KL-reg) ──► Eval
                  │                │               │                │
                  ▼                ▼               ▼                ▼
              outputs/sft/    outputs/rm/     outputs/ppo/    outputs/eval/
              logs/sft_*.jsonl logs/rm_*.jsonl logs/ppo_*.jsonl
```

Each run creates a `run_id` (timestamp + hash), so all artifacts are versioned automatically.

---

## Environment Setup

```bash
git clone <this-repo>
cd mini-rlhf-ppo

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

**Requirements:** Python ≥ 3.10, ~4 GB RAM for CPU smoke test, ~16 GB VRAM for full run.

---

## Quick Start

### Smoke test (CPU, ~5 min)

```bash
bash scripts/smoke_test.sh
```

Runs all 4 stages with tiny data limits (`n_train=20`, `max_steps=2`) on CPU.
Creates `data/rm_pairs.jsonl` automatically if absent.

### Full pipeline (single GPU)

```bash
bash scripts/run_all.sh
```

Runs all stages with the defaults in `configs/`. Produces:
- `outputs/<stage>/<run_id>/` — model checkpoints
- `logs/<stage>_<run_id>.jsonl` — metrics per step
- `reports/weekly_report.md` — auto-generated summary

---

## Running Individual Stages

```bash
# SFT
python -m src.cli sft --config configs/sft.yaml

# Reward Model (requires sft_checkpoint)
python -m src.cli rm --config configs/rm.yaml \
  --override sft_checkpoint=outputs/sft/<run_id> \
  --override generate_rm_data=true

# PPO
python -m src.cli ppo --config configs/ppo.yaml \
  --override sft_checkpoint=outputs/sft/<run_id> \
  --override rm_checkpoint=outputs/rm/<run_id>

# Evaluation
python -m src.cli eval --config configs/eval.yaml \
  --override sft_checkpoint=outputs/sft/<run_id> \
  --override ppo_checkpoint=outputs/ppo/<run_id>

# Report
python -m src.cli report
```

Any config key can be overridden via `--override key=value` (repeatable).

---

## Project Structure

```
configs/          YAML configs for each stage + shared base
src/
  utils/          seed, io (config/JSONL), logging, text helpers
  data/           GSM8K loader, prompt templates
  trainers/       sft.py, reward_model.py, ppo.py
  eval/           metrics.py, evaluate.py
  report/         make_report.py + Jinja2 template
  cli.py          unified entry point
scripts/
  run_all.sh      one-command full pipeline
  smoke_test.sh   CPU smoke test
```

---

## Key Design Choices

| Choice | Reason |
|--------|--------|
| LoRA (r=8) for SFT & PPO | Trains <1% of parameters; fits on a single GPU |
| DistilBERT as RM | Fast, small; sufficient for demo-scale preference data |
| Bradley-Terry loss | Principled pairwise ranking: `-log σ(r_w − r_l)` |
| KL penalty in PPO | Prevents the policy from deviating too far from SFT |
| TRL PPOTrainer | Industry-standard; handles value head, advantage estimation |
| JSONL logging | Append-only, grep-friendly, easy to plot with pandas |

---

## Configuration

All parameters live in `configs/`. Override anything at runtime:

```bash
python -m src.cli sft --config configs/sft.yaml \
  --override model_name=gpt2 \
  --override n_train=500 \
  --override learning_rate=1e-4
```

See [configs/base.yaml](configs/base.yaml) for all shared defaults.

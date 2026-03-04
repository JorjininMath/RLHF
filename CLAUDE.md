# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`mini-rlhf-ppo` — end-to-end RLHF post-training pipeline:
**SFT (LoRA) → Reward Model (Bradley-Terry) → PPO (KL-regularized) → Evaluation**

Dataset: GSM8K math reasoning. Base model: `Qwen/Qwen2.5-0.5B-Instruct` (fallback: `gpt2`).

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Smoke test (CPU, ~5 min)
bash scripts/smoke_test.sh

# Full pipeline
bash scripts/run_all.sh

# Individual stages
python -m src.cli sft  --config configs/sft.yaml
python -m src.cli rm   --config configs/rm.yaml   --override sft_checkpoint=<path>
python -m src.cli ppo  --config configs/ppo.yaml  --override sft_checkpoint=<path> --override rm_checkpoint=<path>
python -m src.cli eval --config configs/eval.yaml --override sft_checkpoint=<path> --override ppo_checkpoint=<path>
python -m src.cli report
```

All config keys can be overridden: `--override key=value` (repeatable).

## Architecture

```
src/
  cli.py              unified entry point → dispatches to trainers/eval/report
  utils/              seed, io (load_config / JSONL), logging, text (number extraction)
  data/               gsm8k.py (HF dataset loader), prompts.py (chat templates)
  trainers/
    sft.py            HF Trainer + PEFT LoRA; saves adapter checkpoint
    reward_model.py   DistilBERT + pairwise loss; can generate pairs from SFT
    ppo.py            TRL PPOTrainer (requires trl>=0.9); KL vs frozen SFT ref
  eval/               metrics.py (accuracy, avg_length), evaluate.py (loads all 3 models)
  report/             Jinja2 template + make_report.py (reads logs/ → reports/)
configs/              base.yaml + per-stage YAML; base is always merged first
scripts/              run_all.sh, smoke_test.sh (both chmod +x)
```

## Key Conventions

- **run_id** = `YYYYMMDD_HHMMSS_xxxxxx`; every run writes to `outputs/<stage>/<run_id>/` and `logs/<stage>_<run_id>.jsonl`
- Config loading: `load_config("configs/base.yaml", stage_yaml, overrides)` — base → stage → CLI wins
- PEFT checkpoints contain `adapter_config.json`; load with `PeftModel.from_pretrained(base, ckpt_path)`
- TRL PPO requires `trl>=0.9`; version is checked at import in `ppo.py`
- Smoke test creates `data/rm_pairs.jsonl` with 5 synthetic pairs if absent

## Expected Layout

```
outputs/   model checkpoints per stage and run_id
logs/      JSONL metrics per stage
data/      cache/ (HF datasets), rm_pairs.jsonl
reports/   weekly_report.md (auto-generated)
docs/      rlhf_blog.md, prompt_design.md (gitignored)
```

## Language Preferences

- **Chinese** for explanations, summaries, document reading, and conceptual answers.
- **English** for all code, comments, docstrings, commit messages, file names, and shell commands.
- When both are needed: Chinese explanation first, then English code/commands in separate fenced blocks.
- All identifiers (variables, functions, classes, directories) must be in English.

## Workflow

- Prefer small, reviewable diffs. Do not change unrelated files.
- Do not delete or rename files unless explicitly requested.

## Defaults

- This repository is **public**: never introduce secrets, credentials, tokens, or personal paths into tracked files.
- `docs/` is gitignored (personal learning notes).

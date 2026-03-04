#!/usr/bin/env bash
# Full RLHF pipeline: SFT → RM → PPO → Eval → Report
set -euo pipefail

PYTHON="${PYTHON:-python}"
BASE_OVERRIDE=""

echo "============================================================"
echo " Mini-RLHF-PPO  |  Full Pipeline"
echo "============================================================"

# ── Stage 1: SFT ─────────────────────────────────────────────────────────────
echo ""
echo ">>> Stage 1/5: Supervised Fine-Tuning (SFT)"
SFT_OUT=$(
  $PYTHON -m src.cli sft --config configs/sft.yaml \
  | tee /dev/stderr \
  | grep "^SFT checkpoint:" \
  | awk '{print $NF}'
)
echo "SFT checkpoint: $SFT_OUT"

# ── Stage 2: Reward Model ─────────────────────────────────────────────────────
echo ""
echo ">>> Stage 2/5: Reward Model Training"
RM_OUT=$(
  $PYTHON -m src.cli rm --config configs/rm.yaml \
    --override sft_checkpoint="$SFT_OUT" \
    --override generate_rm_data=true \
  | tee /dev/stderr \
  | grep "^RM checkpoint:" \
  | awk '{print $NF}'
)
echo "RM checkpoint: $RM_OUT"

# ── Stage 3: PPO ──────────────────────────────────────────────────────────────
echo ""
echo ">>> Stage 3/5: PPO Fine-Tuning"
PPO_OUT=$(
  $PYTHON -m src.cli ppo --config configs/ppo.yaml \
    --override sft_checkpoint="$SFT_OUT" \
    --override rm_checkpoint="$RM_OUT" \
  | tee /dev/stderr \
  | grep "^PPO checkpoint:" \
  | awk '{print $NF}'
)
echo "PPO checkpoint: $PPO_OUT"

# ── Stage 4: Evaluation ───────────────────────────────────────────────────────
echo ""
echo ">>> Stage 4/5: Evaluation (base vs SFT vs PPO)"
$PYTHON -m src.cli eval --config configs/eval.yaml \
  --override sft_checkpoint="$SFT_OUT" \
  --override ppo_checkpoint="$PPO_OUT"

# ── Stage 5: Report ───────────────────────────────────────────────────────────
echo ""
echo ">>> Stage 5/5: Generating Report"
$PYTHON -m src.cli report --config configs/eval.yaml \
  --override sft_checkpoint="$SFT_OUT" \
  --override ppo_checkpoint="$PPO_OUT"

echo ""
echo "============================================================"
echo " Done!  See reports/weekly_report.md"
echo "============================================================"

#!/usr/bin/env bash
# Smoke test: end-to-end on CPU with tiny data limits (<5 min).
set -euo pipefail

PYTHON="${PYTHON:-python}"

# Tiny overrides for speed
TINY="--override device=cpu
      --override n_train=20
      --override n_eval=5
      --override n_rm_pairs=10
      --override n_ppo_steps=4
      --override max_steps=2
      --override num_epochs=1
      --override batch_size=2
      --override mini_batch_size=2
      --override gradient_accumulation_steps=1
      --override max_new_tokens=16
      --override max_length=128"

echo "============================================================"
echo " Mini-RLHF-PPO  |  Smoke Test (CPU, tiny data)"
echo "============================================================"

# ── Create minimal RM pairs if not present ────────────────────────────────────
if [ ! -f data/rm_pairs.jsonl ]; then
  echo ""
  echo ">>> Creating minimal rm_pairs.jsonl for smoke test…"
  mkdir -p data
  $PYTHON - <<'PYEOF'
import json, pathlib
pairs = [
    {"prompt": "Problem: What is 2+2?\nAnswer:", "chosen": "4",  "rejected": "5"},
    {"prompt": "Problem: What is 3*3?\nAnswer:", "chosen": "9",  "rejected": "10"},
    {"prompt": "Problem: What is 10-4?\nAnswer:", "chosen": "6", "rejected": "7"},
    {"prompt": "Problem: What is 5+5?\nAnswer:", "chosen": "10", "rejected": "11"},
    {"prompt": "Problem: What is 6*2?\nAnswer:", "chosen": "12", "rejected": "13"},
]
pathlib.Path("data/rm_pairs.jsonl").write_text(
    "\n".join(json.dumps(p) for p in pairs) + "\n"
)
print("Created data/rm_pairs.jsonl")
PYEOF
fi

# ── Stage 1: SFT ─────────────────────────────────────────────────────────────
echo ""
echo ">>> [1/4] SFT"
SFT_OUT=$(
  $PYTHON -m src.cli sft --config configs/sft.yaml $TINY \
  | tee /dev/stderr \
  | grep "^SFT checkpoint:" \
  | awk '{print $NF}'
)
echo "SFT checkpoint: $SFT_OUT"

# ── Stage 2: RM ───────────────────────────────────────────────────────────────
echo ""
echo ">>> [2/4] Reward Model (using pre-built rm_pairs.jsonl)"
RM_OUT=$(
  $PYTHON -m src.cli rm --config configs/rm.yaml $TINY \
    --override sft_checkpoint="$SFT_OUT" \
    --override generate_rm_data=false \
  | tee /dev/stderr \
  | grep "^RM checkpoint:" \
  | awk '{print $NF}'
)
echo "RM checkpoint: $RM_OUT"

# ── Stage 3: PPO ──────────────────────────────────────────────────────────────
echo ""
echo ">>> [3/4] PPO"
PPO_OUT=$(
  $PYTHON -m src.cli ppo --config configs/ppo.yaml $TINY \
    --override sft_checkpoint="$SFT_OUT" \
    --override rm_checkpoint="$RM_OUT" \
  | tee /dev/stderr \
  | grep "^PPO checkpoint:" \
  | awk '{print $NF}'
)
echo "PPO checkpoint: $PPO_OUT"

# ── Stage 4: Eval ─────────────────────────────────────────────────────────────
echo ""
echo ">>> [4/4] Evaluation"
$PYTHON -m src.cli eval --config configs/eval.yaml $TINY \
  --override sft_checkpoint="$SFT_OUT" \
  --override ppo_checkpoint="$PPO_OUT"

echo ""
echo "============================================================"
echo " Smoke test PASSED"
echo "============================================================"

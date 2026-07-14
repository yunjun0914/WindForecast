#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/yunjun0914/.venvs/WindForecast/bin/python}"
STEM="issue24_group_tcn_oof_v1"

"$PYTHON" -m py_compile \
  models/issue_block_tcn.py \
  utils/issue_block_dataset.py \
  experiments/evaluate_issue24_group_tcn_oof.py

"$PYTHON" -m unittest discover -s tests -p 'test_*.py' -v

"$PYTHON" experiments/evaluate_issue24_group_tcn_oof.py \
  --variants "independent_causal,independent_full,shared_causal,shared_full" \
  --years "2022,2023,2024" \
  --epochs 120 \
  --patience 18 \
  --batch-size 64 \
  --eval-batch-size 256 \
  --hidden-size 128 \
  --num-layers 3 \
  --kernel-size 3 \
  --dropout 0.10 \
  --weight-policy actual_sqrt \
  --stem "$STEM" \
  --verbose \
  2>&1 | tee "results/${STEM}.log"

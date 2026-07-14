#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/yunjun0914/.venvs/WindForecast/bin/python}"
STEM="per_turbine_issue24_tcn_oof_v1"

"$PYTHON" -m py_compile \
  utils/per_turbine_optimal_grid_builder.py \
  utils/issue_block_dataset.py \
  models/issue_block_tcn.py \
  experiments/evaluate_per_turbine_issue24_tcn_oof.py

"$PYTHON" -m unittest \
  tests.test_issue_block_tcn \
  tests.test_per_turbine_optimal_grid_builder \
  -v

"$PYTHON" experiments/evaluate_per_turbine_issue24_tcn_oof.py \
  --epochs 120 \
  --patience 18 \
  --batch-size 64 \
  --eval-batch-size 256 \
  --hidden-size 128 \
  --num-layers 3 \
  --kernel-size 3 \
  --dropout 0.10 \
  --target-share-alpha 0.50 \
  --stem "$STEM" \
  2>&1 | tee "results/${STEM}.log"

printf '\n=== per-turbine issue24 pooled summary ===\n'
cat "results/${STEM}_summary.csv"

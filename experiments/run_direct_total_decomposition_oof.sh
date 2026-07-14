#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/yunjun0914/.venvs/WindForecast/bin/python}"
STEM="direct_total_decomposition_oof_v1"

"$PYTHON" -m py_compile \
  experiments/evaluate_direct_total_decomposition_oof.py \
  tests/test_direct_total_decomposition.py

"$PYTHON" -m unittest tests.test_direct_total_decomposition -v

"$PYTHON" experiments/evaluate_direct_total_decomposition_oof.py \
  --epochs 120 \
  --patience 18 \
  --hidden-size 128 \
  --num-layers 3 \
  --kernel-size 3 \
  --dropout 0.10 \
  --tree-estimators 1200 \
  --alphas 0,0.25,0.5,0.75,1 \
  --stem "$STEM" \
  2>&1 | tee "results/${STEM}.log"

printf '\n=== direct-total summary ===\n'
cat "results/${STEM}_summary.csv"

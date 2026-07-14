#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/yunjun0914/.venvs/WindForecast/bin/python}"
STEM="per_turbine_bin_moe_tcn_w72_oof_v1"

"$PYTHON" -m py_compile \
  models/seqnn.py \
  utils/bin_moe.py \
  experiments/evaluate_per_turbine_bin_moe_tcn_oof.py

"$PYTHON" -m unittest \
  tests.test_bin_moe \
  tests.test_metrics \
  tests.test_per_turbine_scada \
  -v

"$PYTHON" experiments/evaluate_per_turbine_bin_moe_tcn_oof.py \
  --window 72 \
  --checkpoints 10,20,40 \
  --batch-size 512 \
  --eval-batch-size 4096 \
  --hidden-size 64 \
  --num-layers 1 \
  --kernel-size 3 \
  --dropout 0.10 \
  --feature-variant optimal_grid_replace_local16 \
  --target-share-alpha 1.0 \
  --stem "$STEM" \
  2>&1 | tee "results/${STEM}.log"

printf '\n=== bin MoE pooled summary ===\n'
cat "results/${STEM}_summary.csv"

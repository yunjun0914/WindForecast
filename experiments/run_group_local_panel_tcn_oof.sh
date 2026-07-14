#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/yunjun0914/.venvs/WindForecast/bin/python}"
STEM="group_local_panel_tcn_w29_oof_v1"

"$PYTHON" -m py_compile \
  utils/group_local_panel.py \
  experiments/evaluate_group_local_panel_tcn_oof.py

"$PYTHON" -m unittest \
  tests.test_group_local_panel \
  tests.test_metrics \
  tests.test_per_turbine_scada \
  -v

"$PYTHON" experiments/evaluate_group_local_panel_tcn_oof.py \
  --window 29 \
  --checkpoints 5,10,20,40 \
  --fallback-epoch 10 \
  --batch-size 512 \
  --eval-batch-size 4096 \
  --hidden-size 128 \
  --num-layers 3 \
  --kernel-size 3 \
  --dropout 0.10 \
  --feature-variant optimal_grid_replace_local16 \
  --stem "$STEM" \
  2>&1 | tee "results/${STEM}.log"

printf '\n=== group local-panel TCN pooled summary ===\n'
cat "results/${STEM}_summary.csv"

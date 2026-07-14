#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/yunjun0914/.venvs/WindForecast/bin/python}"
BASE_STEM="per_turbine_tcn_mean_grid_context_h128_l3_oof_v1"
SWEEP_STEM="static_share_target_h128_l3_blend_v1"

"$PYTHON" -m py_compile \
  utils/per_turbine_scada.py \
  utils/per_turbine_optimal_grid.py \
  experiments/evaluate_per_turbine_tcn_interval_head_oof.py \
  experiments/evaluate_tcn_capacity_sweep_oof.py

"$PYTHON" -m unittest discover -s tests -p test_per_turbine_scada.py -v

for spec in "050 0.50" "025 0.25" "000 0.00"; do
  read -r tag alpha <<< "$spec"
  stem="per_turbine_tcn_share_a${tag}_h128_l3_oof_v1"
  "$PYTHON" experiments/evaluate_per_turbine_tcn_interval_head_oof.py \
    --window 72 \
    --epochs 120 \
    --patience 18 \
    --hidden-size 128 \
    --num-layers 3 \
    --kernel-size 3 \
    --dropout 0.10 \
    --feature-variant optimal_grid_issue_context \
    --weather-source mixed \
    --input-ablation weather_only \
    --skip-interval-head \
    --target-share-alpha "$alpha" \
    --stem "$stem" \
    2>&1 | tee "results/${stem}.log"
done

"$PYTHON" experiments/evaluate_tcn_capacity_sweep_oof.py \
  --tcn "dynamic=results/${BASE_STEM}_turbine_predictions.csv" \
  --tcn "share_a050=results/per_turbine_tcn_share_a050_h128_l3_oof_v1_turbine_predictions.csv" \
  --tcn "share_a025=results/per_turbine_tcn_share_a025_h128_l3_oof_v1_turbine_predictions.csv" \
  --tcn "share_a000=results/per_turbine_tcn_share_a000_h128_l3_oof_v1_turbine_predictions.csv" \
  --stem "$SWEEP_STEM"

printf '\n=== static-share target comparison ===\n'
cat "results/${SWEEP_STEM}_summary.csv"

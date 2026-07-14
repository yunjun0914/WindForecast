#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/yunjun0914/.venvs/WindForecast/bin/python}"
BLEND_STEM="pinn_static_share_with_tcn_a050_blend_v1"

"$PYTHON" -m py_compile \
  utils/per_turbine_scada.py \
  experiments/evaluate_per_turbine_pinn_oof.py \
  experiments/evaluate_pinn_backbone_replacement_oof.py

"$PYTHON" -m unittest discover -s tests -p test_per_turbine_scada.py -v

"$PYTHON" experiments/evaluate_per_turbine_pinn_oof.py \
  --groups kpx_group_1 \
  --epochs 1 \
  --patience 1 \
  --feature-variant optimal_grid_replace_local16 \
  --target-share-alpha 0.50 \
  --smoke-test \
  --stem per_turbine_pinn_share_smoke_v1

for spec in "075 0.75" "050 0.50" "025 0.25"; do
  read -r tag alpha <<< "$spec"
  stem="per_turbine_pinn_share_a${tag}_oof_v1"
  "$PYTHON" experiments/evaluate_per_turbine_pinn_oof.py \
    --epochs 300 \
    --patience 30 \
    --hidden-size 32 \
    --backbone mlp \
    --residual-amplitude 0.15 \
    --feature-variant optimal_grid_replace_local16 \
    --target-share-alpha "$alpha" \
    --stem "$stem" \
    2>&1 | tee "results/${stem}.log"
done

"$PYTHON" experiments/evaluate_pinn_backbone_replacement_oof.py \
  --pinn "dynamic=results/per_turbine_pinn_optimal_grid_replace_v1_turbine_predictions.csv" \
  --pinn "share_a075=results/per_turbine_pinn_share_a075_oof_v1_turbine_predictions.csv" \
  --pinn "share_a050=results/per_turbine_pinn_share_a050_oof_v1_turbine_predictions.csv" \
  --pinn "share_a025=results/per_turbine_pinn_share_a025_oof_v1_turbine_predictions.csv" \
  --tcn results/per_turbine_tcn_share_a050_h128_l3_oof_v1_turbine_predictions.csv \
  --stem "$BLEND_STEM"

printf '\n=== PINN static-share + TCN a050 comparison ===\n'
cat "results/${BLEND_STEM}_summary.csv"

#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-/home/yunjun0914/.venvs/WindForecast/bin/python}"
TCN_STEM="per_turbine_tcn_share50_full_v1"
TCN_SUBMISSION="results/submission_tcn_share50_v1.csv"
FINAL_SUBMISSION="results/submission_share50_v1.csv"

"$PYTHON" -m py_compile \
  utils/per_turbine_scada.py \
  utils/per_turbine_optimal_grid.py \
  experiments/predict_per_turbine_weather_only_tcn_submission.py \
  experiments/replace_tcn_branch_submission.py

"$PYTHON" -m unittest discover -s tests -p test_per_turbine_scada.py -v
"$PYTHON" -m unittest discover -s tests -p test_weather_time_features.py -v

"$PYTHON" experiments/predict_per_turbine_weather_only_tcn_submission.py \
  --tcn-oof-training results/per_turbine_tcn_share_a050_h128_l3_oof_v1_training.csv \
  --tcn-window 72 \
  --tcn-hidden-size 128 \
  --tcn-num-layers 3 \
  --tcn-kernel-size 3 \
  --tcn-dropout 0.10 \
  --target-share-alpha 0.50 \
  --include-issue-context \
  --turbine-output "results/${TCN_STEM}_turbine_predictions.csv" \
  --training-output "results/${TCN_STEM}_training_epochs.csv" \
  --submission-output "$TCN_SUBMISSION" \
  2>&1 | tee "results/${TCN_STEM}.log"

"$PYTHON" experiments/replace_tcn_branch_submission.py \
  --base results/submission_per_turbine_optgrid_pinn50_groupquota65tunedtree15_weatheronlytcn35_pinnfloor20_finalfloor10_v1.csv \
  --old-tcn results/submission_per_turbine_optgrid_weather_only_tcn_w72_full_v1.csv \
  --new-tcn "$TCN_SUBMISSION" \
  --tcn-weight 0.35 \
  --final-floor 0.10 \
  --output "$FINAL_SUBMISSION" \
  --diagnostics results/submission_share50_v1_diagnostics.csv

"$PYTHON" -c "import hashlib,pandas as pd; p='results/submission_share50_v1.csv'; d=pd.read_csv(p,encoding='utf-8-sig'); print('rows=',len(d),'sha256=',hashlib.sha256(open(p,'rb').read()).hexdigest())"

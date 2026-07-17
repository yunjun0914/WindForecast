#!/usr/bin/env bash
set -euo pipefail

RUN=/home/yunjun0914/windforecast_runs/temporal_representation_experts_20260717_v1
REPO=/home/yunjun0914/WindForecast-source-expert
PY=/home/yunjun0914/.venvs/WindForecast/bin/python
CACHE=/home/yunjun0914/windforecast_runs/canonical_recovery/cache
BASE=/home/yunjun0914/windforecast_runs/repro_tcn_h128_l3_29h_20260717_v1/results/repro_tcn_share50_h128_l3_w72_v1_predictions.csv

mkdir -p "$RUN"/{work,smoke,min,median,max,analysis}
cd "$RUN/work"
if [[ ! -e data ]]; then
    ln -s /home/yunjun0914/windforecast_runs/canonical_recovery/data data
fi
test -f "$BASE"

COMMON_ARGS=(
    --window 72
    --hidden-size 128
    --num-layers 3
    --kernel-size 3
    --dropout 0.10
    --feature-variant optimal_grid_issue_context
    --weather-source mixed
    --input-ablation weather_only
    --skip-interval-head
    --target-share-alpha 0.50
    --temporal-representation-window 7
    --cache-root "$CACHE"
)

"$PY" "$REPO/experiments/evaluate_per_turbine_tcn_interval_head_oof.py" \
    "${COMMON_ARGS[@]}" \
    --groups kpx_group_1 \
    --epochs 1 \
    --patience 1 \
    --temporal-representation min \
    --results-dir "$RUN/smoke" \
    --stem smoke_temporal_min \
    --smoke-test

for representation in min median max; do
    "$PY" "$REPO/experiments/evaluate_per_turbine_tcn_interval_head_oof.py" \
        "${COMMON_ARGS[@]}" \
        --epochs 120 \
        --patience 18 \
        --temporal-representation "$representation" \
        --results-dir "$RUN/$representation" \
        --stem "temporal_${representation}_h128_l3_w72_v1" \
        2>&1 | tee "$RUN/${representation}.log"
done

"$PY" "$REPO/experiments/analyze_temporal_representation_experts.py" \
    --min-prediction "$RUN/min/temporal_min_h128_l3_w72_v1_predictions.csv" \
    --median-prediction "$RUN/median/temporal_median_h128_l3_w72_v1_predictions.csv" \
    --max-prediction "$RUN/max/temporal_max_h128_l3_w72_v1_predictions.csv" \
    --baseline-prediction "$BASE" \
    --results-dir "$RUN/analysis" \
    --stem temporal_representation_experts_v1 \
    2>&1 | tee "$RUN/analysis.log"

tar --exclude='*turbine_predictions.csv' -czf \
    /tmp/temporal_representation_experts_v1_results.tar.gz \
    -C "$RUN" \
    min median max analysis min.log median.log max.log analysis.log

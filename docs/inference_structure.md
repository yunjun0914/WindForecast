# Inference Structure

작성일: 2026-07-08 00:46:07 +09:00

현재 inference 구조는 세 블록만 유지한다.

```text
PINN
TREE
PINN50:TREE50
```

## 1. PINN

```powershell
conda run -n WindForecast python predict_pinn_effective_grid_g1_year_bagging.py
```

Output:

```text
results/submission_pinn_effective_grid_g1_year_bagging.csv
```

## 2. TREE

```powershell
conda run -n WindForecast python predict_tree_compact_v2_metric_valid_lgbm_mean.py
```

Output:

```text
results/submission_tree_compact_v2_metric_valid_lgbm_mean.csv
```

## 3. PINN50:TREE50

```powershell
conda run -n WindForecast python blend_submission_files.py --base results/submission_pinn_effective_grid_g1_year_bagging.csv --extra results/submission_tree_compact_v2_metric_valid_lgbm_mean.csv --extra-weight 0.5 --out results/submission.csv
```

Output:

```text
results/submission.csv
```

## Validation

```powershell
conda run -n WindForecast python experiments/evaluate_pinn_effective_grid_g1_year_bagging_oof.py
conda run -n WindForecast python experiments/evaluate_tree_compact_v2_multi_year_models.py --models lgbm_tuned,lgbm_regularized --train-policies metric_valid --stem tree_compact_v2_multi_year_lgbm_policy
conda run -n WindForecast python experiments/evaluate_pinn_tree_compact_v2_metric_valid_blend.py --tree-predictions results/tree_compact_v2_multi_year_lgbm_policy_predictions.csv --pinn-oof results/pinn_effective_grid_g1_year_bagging_oof_predictions.csv --tree-weights 0,0.25,0.4,0.5,0.6,0.75,1.0
```

## Policy

- Do not add broad experimental entrypoints to the repository root.
- Keep diagnostic scripts under `experiments/`, and only keep them if they are part of the current validation loop.
- Do not create test submissions unless validation improvement is meaningful or the user explicitly asks.


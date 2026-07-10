# WindForecast Agent Context

Updated: 2026-07-11 KST

작업 시작 전에 `AGENTS.md`, 이 문서, `docs/best_model_usage.md`, `docs/rules.md`, `docs/exp_logs.md` 최근 항목을 읽는다.

## Current Public Best

| Item | Value |
|---|---|
| File | `results/submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_finalfloor10_v1.csv` |
| Public score | `0.6386205415` |
| Public 1-nMAE | `0.8682636645` |
| Public FiCR | `0.4089774184` |
| Submit id | `1484025` |
| Submitted at | `2026-07-11 01:21:21 KST` |

`results/submission.csv`는 작업용 임시 파일이다. 최고 모델로 간주하지 않는다.

## Current Best Formula

```text
TCN_family =
    0.30 * submission_seqnn_short_tcn_w24_v1
  + 0.40 * submission_seqnn_mid_tcn_w72_v1
  + 0.30 * submission_seqnn_long_tcn_w168_v1

PINN_floor = clip(submission_pinn_lgbm_teacher_year_bagging, 0.35 * capacity, capacity)

final_raw =
    0.25 * PINN_floor
  + 0.20 * submission_tree_lgbm_best_v2_l1
  + 0.55 * TCN_family

final = clip(final_raw, 0.10 * capacity, capacity)
```

Branch files:

| Branch | File |
|---|---|
| PINN | `results/submission_pinn_lgbm_teacher_year_bagging.csv` |
| TREE | `results/submission_tree_lgbm_best_v2_l1.csv` |
| TCN W24 | `results/submission_seqnn_short_tcn_w24_v1.csv` |
| TCN W72 | `results/submission_seqnn_mid_tcn_w72_v1.csv` |
| TCN W168 | `results/submission_seqnn_long_tcn_w168_v1.csv` |

Important correction:

- The current best TREE branch is plain `submission_tree_lgbm_best_v2_l1.csv`.
- It is not the group3 pseudo2022 TREE branch.
- The previous best `submission_pinn25_tree40_tcn35_tree_g3_vestas_pseudo2022_w010.csv` is now stale.

## Fast Rebuild

Full commands are in `docs/best_model_usage.md`. Short version:

1. Apply 35% floor to the PINN branch with `experiments/apply_metric_floor_submission.py`.
2. Blend floored PINN / TREE / TCN family with `experiments/blend_three_branch_submission.py` using weights `0.25 / 0.20 / 0.55`.
3. Apply final 10% floor with `experiments/apply_metric_floor_submission.py`.

The reconstruction was numerically checked against the current best CSV; maximum absolute difference was `3.64e-12`.

## From-Scratch Regeneration

Detailed branch commands and hyperparameters are in `docs/best_model_usage.md`.

Branch entry points:

```text
PINN: predict_pinn_effective_grid_g1_year_bagging.py
TREE: predict_power_lgbm_best.py
TCN : experiments/predict_seqnn_submission.py
```

TREE best hyperparameter file:

```text
results/power_lgbm_hyperparams_v2_l1_20_best.csv
```

TCN family uses weighted-L1 loss, not FiCR-only loss, for the current best.

## Public Timeline

The date/time best timeline is maintained in `docs/best_model_usage.md`.

Key transition:

```text
2026-07-09 16:50:38
  previous best: submission.csv, memo pinn25_tree40_tcn35_tree_g3_vestas_pseudo2022
  public score: 0.6370788926

2026-07-11 01:21:21
  current best: submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_finalfloor10_v1.csv
  public score: 0.6386205415
```

## Operating Rules

1. Before experiments or large code changes, explain purpose, pipeline, changed files, validation, expected runtime, and result file names.
2. Do not create test submissions unless the user explicitly asks.
3. Do not change current-best weights or floors without user approval.
4. Do not use LGBM teacher training for teacher-style features unless the user explicitly re-approves it.
5. Report OOF and public results separately and always include exact file names.
6. Use `docs/rules.md` as the source of truth for competition rule decisions.

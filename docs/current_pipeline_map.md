# Current Pipeline Map

Updated: 2026-07-11 KST

이 문서는 현재 public best 기준의 파일 흐름만 빠르게 보여준다. 상세 재현 명령과 하이퍼파라미터는 `docs/best_model_usage.md`를 기준으로 한다.

## Current Best Path

```text
LDAPS/GFS + SCADA + labels
  -> PINN branch
  -> TREE branch
  -> TCN W24/W72/W168 branches
  -> PINN 35% floor
  -> PINN25 / TREE20 / TCN55 blend
  -> final 10% floor
  -> results/submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_finalfloor10_v1.csv
```

## Branch Entry Points

| Branch | Script | Current best output |
|---|---|---|
| PINN | `predict_pinn_effective_grid_g1_year_bagging.py` | `results/submission_pinn_lgbm_teacher_year_bagging.csv` |
| TREE | `predict_power_lgbm_best.py` | `results/submission_tree_lgbm_best_v2_l1.csv` |
| TCN W24 | `experiments/predict_seqnn_submission.py --window 24` | `results/submission_seqnn_short_tcn_w24_v1.csv` |
| TCN W72 | `experiments/predict_seqnn_submission.py --window 72` | `results/submission_seqnn_mid_tcn_w72_v1.csv` |
| TCN W168 | `experiments/predict_seqnn_submission.py --window 168` | `results/submission_seqnn_long_tcn_w168_v1.csv` |
| PINN floor / final floor | `experiments/apply_metric_floor_submission.py` | intermediate and final floored CSVs |
| Three-branch blend | `experiments/blend_three_branch_submission.py` | pre-final-floor blend |

## Blend Formula

```text
TCN_family =
    0.30 * TCN_W24
  + 0.40 * TCN_W72
  + 0.30 * TCN_W168

PINN_floor = clip(PINN, 0.35 * capacity, capacity)

final_raw =
    0.25 * PINN_floor
  + 0.20 * TREE
  + 0.55 * TCN_family

final = clip(final_raw, 0.10 * capacity, capacity)
```

## Notes

- The current best TREE is `submission_tree_lgbm_best_v2_l1.csv`.
- The group3 pseudo2022 TREE branch is not part of the current best.
- `results/submission.csv` is a scratch output and should not be used as a source of truth.
- Do not change weights or floors without explicit user approval.

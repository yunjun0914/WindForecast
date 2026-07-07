# Experiment Logs

작성일: 2026-07-08 00:46:07 +09:00

실험명: repository cleanup and main pipeline reset

## Current Main Pipeline

현재 모델은 세 덩어리만 메인으로 본다.

| Block | Role | Main files | Status |
|---|---|---|---|
| PINN | physics-heavy prediction, peak/FICR support | `predict_pinn_effective_grid_g1_year_bagging.py`, `train_pinn.py` | keep |
| TREE | compact weather + meteo + physics LGBM mean | `predict_tree_compact_v2_metric_valid_lgbm_mean.py`, `predict_tree_compact_physics_v2.py` | keep |
| PINN50:TREE50 | final stable blend | `blend_submission_files.py` | keep |

Main validation files:

| Purpose | File |
|---|---|
| PINN OOF validation | `experiments/evaluate_pinn_effective_grid_g1_year_bagging_oof.py` |
| TREE year-fold validation | `experiments/evaluate_tree_compact_v2_multi_year_models.py` |
| PINN/TREE blend validation | `experiments/evaluate_pinn_tree_compact_v2_metric_valid_blend.py` |

Important result files:

| Result | File | Note |
|---|---|---|
| Best public candidate | `results/submission_pinn50_tree_all_meteo_compact_v2_50.csv` | public around `0.62423` |
| Current final candidate | `results/submission.csv` | LGBM-teacher PINN 50% + tuned LGBM TREE 50% |
| Current aggressive candidate | `results/submission_pinn40_tree60_lgbmteacher_powerlgbm_v2_l1.csv` | validation best weight, tree=0.60 |
| PINN only candidate | `results/submission_pinn_effective_grid_g1_year_bagging.csv` | PINN-only reference |
| TREE candidate | `results/submission_tree_compact_v2_metric_valid_lgbm_mean.csv` | tree-only reference |
| PINN OOF | `results/pinn_effective_grid_g1_year_bagging_oof_predictions.csv` | blend validation input |
| PINN OOF, LGBM teacher | `results/pinn_effective_grid_g1_year_bagging_lgbm_time_oof_oof_predictions.csv` | current PINN validation input |
| TREE OOF | `results/tree_compact_v2_multi_year_lgbm_policy_predictions.csv` | blend validation input |
| Tuned TREE OOF | `results/power_lgbm_best_v2_l1_predictions.csv` | current TREE validation input |

## Current Scores

Latest update: 2026-07-08 02:43:51 +09:00

Validation 기준:

| Model | Mean score | Mean nMAE | Mean FICR | Note |
|---|---:|---:|---:|---|
| PINN, corrected RF-OOB teacher | `0.60838` | `0.14124` | `0.35800` | honest teacher baseline |
| PINN, LGBM time-OOF teacher | `0.61259` | `0.14268` | `0.36786` | better FICR, group3 slightly worse |
| TREE only, metric-valid LGBM mean | `0.61157` | `0.13045` | `0.35358` | current tree baseline |
| TREE only, tuned group LGBM v2 | `0.62361` | `0.12851` | `0.37573` | group-specific hparams, OOF power-curve |
| PINN/TREE blend, LGBM teacher + tuned TREE, tree=0.50 | `0.62679` | `0.13001` | `0.38359` | current stable final candidate |
| PINN/TREE blend, LGBM teacher + tuned TREE, tree=0.60 | `0.62749` | `0.12879` | `0.38378` | validation best, slightly less conservative |
| PINN/TREE blend, best validation weight near tree=0.4 | `0.62481` | `0.12908` | `0.37870` | validation best |
| PINN50:TREE50 public submission | `0.62423` | - | - | best confirmed public candidate |

## Kept Ideas

These are useful but not main pipeline code.

| Idea | Current judgment |
|---|---|
| Tree hyperparameter search | useful; v2 focused L1 search improved TREE OOF by about `+0.012` |
| Group-specific tree tuning | confirmed useful |
| Sample-weight policy search | useful; `actual_sqrt` often won in v2 |
| Weather calibration using SCADA | conceptually strong but hard; keep as later track |
| SCADA inventory/data quality audit | useful reference only |

## Archived Or Rejected Experiments

These were removed from code/results and kept only as conclusions.

| Experiment | Result | Decision |
|---|---|---|
| PINN residual tree | public/validation gain too small or unstable | reject for now |
| SCADA operational teacher feature | feature importance high, validation worse | reject direct feature use |
| SCADA quality sample weights | tiny improvement at best | defer |
| Site/turbine weather reconstruction as extra features | `+0.001~0.002` only | defer |
| Group3 VESTAS residual transfer | worse | reject |
| GRU/RNN quick sequence model | no clear gain | defer |
| Statewise/metric calibration variants | small changes only | defer |
| Kneedle/hour-bias residual filtering | not robust enough | defer |

## Cleanup Decision

The repository was reduced to the current three-block pipeline plus small validation scripts.

Removed categories:

- Old root `evaluate_*`, `predict_*`, `diagnose_*`, `calibrate_*`, `tune*`, `sweep*` scripts not needed for the three-block pipeline.
- Old teacher/residual/transfer/statewise/site-reconstruction experiment scripts.
- Old result CSV/PNG/PT artifacts not needed for current validation or final submission.
- Previous competition reference folder.
- Unused model wrappers except `models/pinn.py`.

## Next Work

Do not restart broad feature-chasing yet.

Next recommended sequence:

1. Submit or externally check `results/submission.csv`.
2. If public score confirms direction, commit the LGBM-teacher/tuned-tree pipeline.
3. Next tuning: XGB/ExtraTrees diversity for TREE ensemble.
4. Optional: group3-specific teacher backend mix, because RF teacher was better for group3 than LGBM.
5. Keep `PINN50:TREE50` as stable default unless user explicitly chooses validation-best tree=0.60.

Submission rule:

- Do not create new test submission unless validation improves by roughly `+0.01` or more, or the user explicitly asks for a diagnostic submission.

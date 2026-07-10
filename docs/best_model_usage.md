# Best Model Usage And Reproduction

Updated: 2026-07-11 KST

This document is the source of truth for the current public-best submission, how to rebuild it from existing branch files, and how to regenerate the branch files from scratch.

## Current Public Best

| Item | Value |
|---|---|
| Submission file | `results/submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_finalfloor10_v1.csv` |
| Public submit id | `1484025` |
| Submitted at | `2026-07-11 01:21:21` |
| Public score | `0.6386205415` |
| Public 1-nMAE | `0.8682636645` |
| Public FiCR | `0.4089774184` |
| Rows | `8760` |

Important:

- `results/submission.csv` is a working scratch file. Do not treat it as the best model.
- The current public best is not the older `PINN25 + TREE40 + TCN35 + group3 pseudo2022` submission.
- The current public best uses the plain TREE branch `results/submission_tree_lgbm_best_v2_l1.csv`, not the group3 pseudo2022 TREE branch.
- The file name says `weightedl1` because the TCN family is the weighted-L1 TCN family.

## Formula

The final prediction is:

```text
TCN_family =
    0.30 * TCN_W24
  + 0.40 * TCN_W72
  + 0.30 * TCN_W168

PINN_floor =
  clip(PINN, 0.35 * capacity, capacity)

blend =
    0.25 * PINN_floor
  + 0.20 * TREE
  + 0.55 * TCN_family

final =
  clip(blend, 0.10 * capacity, capacity)
```

Group capacities:

| Group | Capacity |
|---|---:|
| `kpx_group_1` | `21600` |
| `kpx_group_2` | `21600` |
| `kpx_group_3` | `21000` |

## Branch Files

The current best is exactly reproduced from these branch files:

| Branch | File | Role |
|---|---|---|
| PINN | `results/submission_pinn_lgbm_teacher_year_bagging.csv` | Physics/effective-wind PINN branch. Legacy filename; keep the file name as-is for exact reproduction. |
| TREE | `results/submission_tree_lgbm_best_v2_l1.csv` | Tuned group-wise LightGBM TREE branch. |
| TCN W24 | `results/submission_seqnn_short_tcn_w24_v1.csv` | Short-window TCN. |
| TCN W72 | `results/submission_seqnn_mid_tcn_w72_v1.csv` | Mid-window TCN. |
| TCN W168 | `results/submission_seqnn_long_tcn_w168_v1.csv` | Long-window TCN. |

Verification note:

The above branch combination was checked against `results/submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_finalfloor10_v1.csv`; maximum absolute difference was `3.64e-12`, so the reconstruction is numerically identical.

## Fast Rebuild From Existing Branch Files

Use the normal project environment:

```powershell
conda run -n WindForecast python --version
```

Step 1: floor only the PINN branch at 35% capacity.

```powershell
conda run -n WindForecast python experiments\apply_metric_floor_submission.py `
  results\submission_pinn_lgbm_teacher_year_bagging.csv `
  --floor-ratio 0.35 `
  --output results\submission_pinn_lgbm_teacher_year_bagging_pinnfloor35.csv `
  --diagnostics-output results\submission_pinn_lgbm_teacher_year_bagging_pinnfloor35_diagnostics.csv
```

Step 2: blend the floored PINN, TREE, and weighted-L1 TCN family.

```powershell
conda run -n WindForecast python experiments\blend_three_branch_submission.py `
  --pinn results\submission_pinn_lgbm_teacher_year_bagging_pinnfloor35.csv `
  --tree results\submission_tree_lgbm_best_v2_l1.csv `
  --tcn24 results\submission_seqnn_short_tcn_w24_v1.csv `
  --tcn72 results\submission_seqnn_mid_tcn_w72_v1.csv `
  --tcn168 results\submission_seqnn_long_tcn_w168_v1.csv `
  --pinn-weight 0.25 `
  --tree-weight 0.20 `
  --tcn-family-weight 0.55 `
  --tcn24-weight 0.30 `
  --tcn72-weight 0.40 `
  --tcn168-weight 0.30 `
  --output results\submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_prefinalfloor_v1.csv
```

Step 3: apply the final 10% floor.

```powershell
conda run -n WindForecast python experiments\apply_metric_floor_submission.py `
  results\submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_prefinalfloor_v1.csv `
  --floor-ratio 0.10 `
  --output results\submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_finalfloor10_v1.csv `
  --diagnostics-output results\submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_finalfloor10_v1_diagnostics.csv
```

Do not add `--also-update-submission-csv` unless the user explicitly asks to overwrite `results/submission.csv`.

## Current Best Diagnostics

Diagnostics file:

```text
results/submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_finalfloor10_v1_diagnostics.csv
```

Current values:

| Group | PINN floor ratio | PINN floor value | PINN rows raised | Final floor value | Final rows raised after blend | Final mean |
|---|---:|---:|---:|---:|---:|---:|
| `kpx_group_1` | `0.35` | `7560` | `4620` | `2160` | `0` | `8822.7941` |
| `kpx_group_2` | `0.35` | `7560` | `3858` | `2160` | `0` | `9671.8902` |
| `kpx_group_3` | `0.35` | `7350` | `4672` | `2100` | `0` | `7954.2285` |

The final 10% floor did not raise any row after the blend in this specific submission. It is still part of the official reconstruction recipe.

## From-Scratch Branch Regeneration

Use this section only when the branch files are missing or stale. These commands retrain/regenerate branch files and can take a long time.

### 1. PINN Branch

Command:

```powershell
conda run -n WindForecast python predict_pinn_effective_grid_g1_year_bagging.py `
  --output results\submission_pinn_lgbm_teacher_year_bagging.csv `
  --fold-stats-output results\pinn_lgbm_teacher_year_bagging_fold_stats.csv
```

Current code notes:

- `TEACHER_BACKEND = rf_oob` in `utils/pinn_effective_pipeline.py`.
- The output file name is legacy and contains `lgbm_teacher`, but new regeneration uses RF-OOB teacher cache logic.
- `utils/pinn_scada_teacher_config.py` applies the tuned PINN hyperparameters:

```text
LAMBDA:
  betz = 0.29310056
  bc = 0.00035899
  flat = 0.06738267
  smooth = 0.00323965
  hod = 0.00004575
  moy = 0.001
  hour = 0.01
  hour_l1 = 0.0
  hour_prox_start_epoch = 0
  year = 0.01
GAMMA = 0.00682273
BIAS_LR = 0.00201426
LR = 0.00119518
```

PINN training structure:

| Item | Value |
|---|---|
| Year bagging | train `2022,2023`, `2022,2024`, `2023,2024`; average test predictions |
| Group1 teacher mode | `cubic` effective grid |
| Group2 teacher mode | `p90` canonical |
| Group3 teacher mode | Unison canonical + Vestas canonical mix, Vestas weight `0.30` |
| Data loss default | `metric_soft` |
| Stage1 epochs | inherited from `train_pinn.py`, default `500` unless overridden |
| Stage2 epochs | inherited from `train_pinn.py`, default `2000` unless overridden |
| Early stopping | enabled by default in the current script |

### 2. TREE Branch

Command:

```powershell
conda run -n WindForecast python predict_power_lgbm_best.py `
  --best-csv results\power_lgbm_hyperparams_v2_l1_20_best.csv `
  --feature-profile full_v2 `
  --output results\submission_tree_lgbm_best_v2_l1.csv
```

The current public best uses `full_v2` through the script default. Do not substitute `group_family_quota65_v1` if the goal is to reproduce the current public best.

TREE hyperparameters are loaded from:

```text
results/power_lgbm_hyperparams_v2_l1_20_best.csv
```

Best rows:

| Group | objective | n_estimators | lr | leaves | depth | min_child | subsample | colsample | alpha | lambda | min_split | min_output_ratio | weight |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `kpx_group_1` | `regression_l1` | `1646` | `0.0203991594` | `128` | `6` | `80` | `0.8184083164` | `0.7571670024` | `0.0081840132` | `1.8139900710` | `0.0265533784` | `0.10` | `actual_sqrt` |
| `kpx_group_2` | `regression_l1` | `1147` | `0.0144991591` | `48` | `6` | `160` | `0.9438219366` | `0.7670068382` | `0.0333872971` | `3.1545764871` | `0.0527314458` | `0.10` | `actual_sqrt` |
| `kpx_group_3` | `regression_l1` | `1309` | `0.0259778246` | `64` | `4` | `80` | `0.7483047258` | `0.6741920876` | `0.0011754614` | `2.8931055157` | `0.0406495774` | `0.10` | `actual_sqrt` |

TREE OOF reference for this branch:

| Metric | Value |
|---|---:|
| OOF score | `0.6236110317` |
| OOF nMAE | `0.1285047060` |
| OOF FiCR | `0.3757267694` |
| Worst fold | `0.6066260194` |

### 3. TCN Branches

Weighted-L1 TCN family commands:

```powershell
conda run -n WindForecast python experiments\predict_seqnn_submission.py `
  --model tcn `
  --window 24 `
  --stem submission_seqnn_short_tcn_w24_v1

conda run -n WindForecast python experiments\predict_seqnn_submission.py `
  --model tcn `
  --window 72 `
  --stem submission_seqnn_mid_tcn_w72_v1

conda run -n WindForecast python experiments\predict_seqnn_submission.py `
  --model tcn `
  --window 168 `
  --stem submission_seqnn_long_tcn_w168_v1
```

Default TCN training settings:

| Setting | Value |
|---|---:|
| Loss | `weighted_l1` |
| Weight policy | `actual_sqrt` |
| Epochs | `120` |
| Patience | `18` |
| Batch size | `512` |
| Eval batch size | `4096` |
| Hidden size | `64` |
| Num layers | `1` |
| Kernel size | `3` |
| Dropout | `0.10` |
| LR | `1e-3` |
| Weight decay | `1e-4` |
| Gradient clip | `1.0` |
| Seed | `42` |

Do not use the FiCR-only TCN family to reproduce the current best. FiCR-only raised public FiCR but hurt total score.

## Public Best Timeline

This table tracks public submissions seen in the scoreboard screenshots. `Best-so-far` tells which model was strongest after that timestamp.

| Time KST | Submit id | File / memo | Public score | 1-nMAE | FiCR | Best-so-far after this |
|---|---:|---|---:|---:|---:|---|
| `2026-07-09 16:50:38` | `1480546` | `submission.csv`, memo `pinn25_tree40_tcn35_tree_g3_vestas_pseudo2022` | `0.6370788926` | `0.8701764551` | `0.4039813302` | This model |
| `2026-07-09 17:11:18` | `1480587` | `submission.csv` | `0.6362267968` | `0.8699141079` | `0.4025394858` | `1480546` |
| `2026-07-09 18:57:39` | `1480736` | `submission.csv` | `0.6357518532` | `0.8692285391` | `0.4022751673` | `1480546` |
| `2026-07-10 21:28:01` | `1483296` | `submission_pinn45_tcn55_g13_delta0375.csv` | `0.6275136665` | `0.8582049759` | `0.3968223571` | `1480546` |
| `2026-07-10 21:39:57` | `1483310` | `submission_pinn45_tcn55_tree_stack_g13_clip025.csv` | `0.6335692297` | `0.8667639269` | `0.4003745325` | `1480546` |
| `2026-07-10 23:12:44` | `1483459` | `submission_pinn10_tree15_tcn75_tcn_ficr_only_family_v1.csv` | `0.6361412210` | `0.8615048788` | `0.4107775632` | `1480546` |
| `2026-07-10 23:22:07` | `1483482` | `submission_pinn25_tree40pseudo_tcn35_ficr_only_family_v1.csv` | `0.6339040222` | `0.8669600505` | `0.4008479938` | `1480546` |
| `2026-07-11 00:06:52` | `1483681` | `submission_current_best_floor10.csv` | `0.6364805561` | `0.8702088685` | `0.4027522436` | `1480546` |
| `2026-07-11 00:13:35` | `1483725` | `submission_pinn25_tree40_tcn35_tree_g3_vestas_pseudo2022_w010_rebuilt_floor10.csv` | `0.6354556595` | `0.8696982200` | `0.4012130991` | `1480546` |
| `2026-07-11 01:21:21` | `1484025` | `submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_finalfloor10_v1.csv` | `0.6386205415` | `0.8682636645` | `0.4089774184` | This model |

Notes:

- The FiCR-only TCN submission at `2026-07-10 23:12:44` had the highest public FiCR in this list (`0.4107775632`) but lower total score.
- The `2026-07-11 01:21:21` model won by raising FiCR while keeping nMAE damage manageable.

## OOF Context For The Current Best

The OOF grid that motivated the current best is:

```text
results/pinn_floor_three_branch_grid_v1_summary.csv
```

Top OOF row:

| PINN floor | PINN weight | TREE weight | TCN weight | OOF score | nMAE | FiCR | Worst fold |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.35` | `0.25` | `0.20` | `0.55` | `0.6336379975` | `0.1283582410` | `0.3956342361` | `0.6224051731` |

Generated submission candidates:

```text
results/submission_pinn_floor_three_branch_candidates_v1_summary.csv
```

## Operational Rules

- Before running a new experiment, explain the goal, changed files, validation plan, expected runtime, and result file names to the user.
- Do not create a submission unless the user explicitly asks.
- Do not change the current-best weights without user approval.
- Do not use LGBM teacher training for teacher-style features unless the user explicitly re-approves it.
- For current-best reproduction, use the branch files and formula in this document.

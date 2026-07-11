# Current Best Detailed Structure

Updated: 2026-07-11 KST

This document records the current public-best structure in detail. It separates the
reproducible submission recipe from the unresolved OOF provenance gap.

## Scope

Current public best:

```text
results/submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_finalfloor10_v1.csv
```

Public scoreboard:

| Item | Value |
|---|---:|
| Submit id | `1484025` |
| Submitted at | `2026-07-11 01:21:21 KST` |
| Public score | `0.6386205415` |
| Public 1-nMAE | `0.8682636645` |
| Public FiCR | `0.4089774184` |
| Rows | `8760` |

Not current best:

- `results/submission.csv` is a scratch file.
- The older `PINN25 + TREE40 + TCN35 + group3 pseudo2022` submission is stale.
- The current public best TREE branch is plain `submission_tree_lgbm_best_v2_l1.csv`, not the group3 pseudo2022 TREE branch.
- `group_family_quota65_v1` is a TREE candidate, not part of the current public-best submission.

## Final Recipe

Capacity:

| Group | Capacity |
|---|---:|
| `kpx_group_1` | `21600` |
| `kpx_group_2` | `21600` |
| `kpx_group_3` | `21000` |

Branch files:

| Branch | File | Role |
|---|---|---|
| PINN | `results/submission_pinn_lgbm_teacher_year_bagging.csv` | Physics/effective-wind PINN branch. Filename is legacy. |
| TREE | `results/submission_tree_lgbm_best_v2_l1.csv` | group-wise LightGBM branch, `full_v2` profile. |
| TCN W24 | `results/submission_seqnn_short_tcn_w24_v1.csv` | Short-window TCN. |
| TCN W72 | `results/submission_seqnn_mid_tcn_w72_v1.csv` | Mid-window TCN. |
| TCN W168 | `results/submission_seqnn_long_tcn_w168_v1.csv` | Long-window TCN. |

Formula:

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

Numerical rebuild status:

- On duck, the branch files above reproduced the current public-best CSV.
- Maximum absolute difference against the saved public-best CSV: `3.64e-12`.
- This confirms the submission recipe, not the original OOF component provenance.

## Floor Logic

### PINN floor35

`floor35` is not a PINN training hyperparameter. It is a post-processing value selected from an OOF blend sweep.

```text
PINN_floor = clip(PINN, 0.35 * capacity, capacity)
```

Role:

- Prevents the PINN branch from dragging the blend too low.
- Targets the observed low-output underprediction issue in the PINN branch.
- Helps the final blend stay closer to FiCR 6 percent and 8 percent bands when actual generation is not low.
- Does not force the final prediction to be at least 35 percent capacity, because PINN contributes only 25 percent to the blend.

OOF sweep source:

```text
results/pinn_floor_three_branch_grid_v1_summary.csv
```

Top summary row:

| PINN floor | PINN weight | TREE weight | TCN weight | OOF score | nMAE | FiCR | Worst fold |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `0.35` | `0.25` | `0.20` | `0.55` | `0.6336379975` | `0.1283582410` | `0.3956342361` | `0.6224051731` |

Important provenance note:

- The summary file remains available.
- The exact PINN/TREE/TCN OOF component filenames that produced this row are not currently recorded.
- Full OOF reconstruction of this exact row was attempted on duck and abandoned because exact component provenance was missing.
- Do not treat a near match or TREE-only score as proof of this OOF row.

### Final floor10

```text
final = clip(blend, 0.10 * capacity, capacity)
```

Role:

- Aligns the prediction floor with the official metric's `actual >= 10% capacity` valid region.
- In this specific public-best submission, diagnostics show final floor10 raised `0` rows after blending.
- It remains part of the exact rebuild recipe and should not be removed without a new OOF check.

## PINN Branch

### Files And Entry Points

Submission branch:

```text
results/submission_pinn_lgbm_teacher_year_bagging.csv
```

Main regeneration script:

```text
predict_pinn_effective_grid_g1_year_bagging.py
```

OOF script:

```text
experiments/evaluate_pinn_effective_grid_g1_year_bagging_oof.py
```

Legacy naming note:

- The output filename contains `lgbm_teacher`.
- Current code path uses RF-OOB teacher cache logic, not LGBM teacher training.

### Data Used

Training and prediction inputs:

| Data | Use |
|---|---|
| `data/train/ldaps_train.csv` | Train weather |
| `data/train/gfs_train.csv` | Train weather |
| `data/train/train_labels.csv` | Supervised power targets |
| `data/train/scada_vestas_train.csv` | Vestas SCADA teacher source |
| `data/train/scada_unison_train.csv` | Unison SCADA teacher source |
| `data/test/ldaps_test.csv` | Test weather |
| `data/test/gfs_test.csv` | Test weather |
| `data/sample_submission.csv` | Forecast ids and output alignment |

No test labels are used.

### Year Bagging

The submission branch is generated with leave-one-year-out style year bagging over train years:

```text
years = 2022, 2023, 2024
folds:
  train 2022,2023 -> held-out valid 2024 -> predict test
  train 2022,2024 -> held-out valid 2023 -> predict test
  train 2023,2024 -> held-out valid 2022 -> predict test

final PINN branch = mean(test predictions from the 3 folds)
```

The held-out year is used for early stopping when validation data is available.

### Model Split

| Model | Groups |
|---|---|
| Vestas PINN | `kpx_group_1`, `kpx_group_2` |
| Unison PINN | `kpx_group_3` |

Each group gets its own `TurbineGroupBias`, but group1/group2 share the Vestas physical PINN model.

### Weather And Teacher Construction

Base weather:

- `build_extended_pinn_weather(ldaps, gfs)`
- `build_meteo_features(ldaps, gfs)`
- `add_meteo_block(..., "all_meteo")`
- Optional effective wind features via `add_effective_wind_features`

Group teacher recipes in the current code path:

| Group | Weather base | SCADA source | Teacher mode |
|---|---|---|---|
| `kpx_group_1` | effective-grid weather | Vestas SCADA | `cubic` |
| `kpx_group_2` | canonical weather | Vestas SCADA | `p90` |
| `kpx_group_3` | canonical weather mix | Unison SCADA + Vestas SCADA | `p90` mix |

Current code detail for group3:

```text
blend_weather("effective_g1_group3_canonical_mix", g3_unison, g3_vestas, 0.30)
```

`blend_weather` computes:

```text
0.30 * g3_unison + 0.70 * g3_vestas
```

This is the current code behavior. If older notes describe this as "Vestas weight 0.30", treat that older wording as stale unless the generating commit proves otherwise.

Teacher backend:

```text
teacher_backend = rf_oob
```

Teacher cache behavior:

- Uses `apply_extended_teacher_crossfit_cached`.
- Default is cache-first.
- `--refresh-teacher-cache` is required to rebuild teacher cache.
- Do not reintroduce LGBM teacher training without explicit approval.

### PINN Training Loss

Default data loss:

```text
data_loss = metric_soft
```

Implementation shape:

```text
valid = y >= 0.10 * capacity
error = (pred - y) / capacity
error_rate = abs(error)
l_nmae = mean(error_rate)

soft_price =
    4.0
  - sigmoid((error_rate - 0.06) / gamma)
  - 3.0 * sigmoid((error_rate - 0.08) / gamma)

ficr_soft = sum(y * soft_price) / sum(y * 4.0)
l_ficr = 1 - ficr_soft

metric_soft loss = 0.5 * l_nmae + 0.5 * l_ficr
```

Current tuned values from `utils/pinn_scada_teacher_config.py`:

```text
GAMMA = 0.00682273
BIAS_LR = 0.00201426
LR = 0.00119518

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
```

Physics losses used in stage1:

- Betz ceiling loss
- cut-in/cut-out boundary condition loss
- rated-to-cutout flatness loss
- C curve smoothness loss
- data loss per group

Training stages:

| Stage | Trained parameters | Loss |
|---|---|---|
| Stage1 | physical PINN model, plus optional auxiliary bias modules if flags are on | physics losses + data loss |
| Stage2 | HOD calendar bias by default; MOY/DOW only if flags are on | data loss over base prediction + calendar bias |

Default flags for current regeneration command:

| Flag | Default | Current public-best regeneration meaning |
|---|---:|---|
| `--use-dow-bias` | off | DOW bias off |
| `--use-moy-bias` | off | MOY bias off |
| `--use-train-hour-bias` | off | train-only row/hour absorption off |
| `--use-train-year-bias` | off | train-year absorption off |
| `--use-scada-wd-correction` | off | SCADA wind-direction correction off |
| `--use-teacher-residual-bias` | off in OOF script | teacher residual NN bias off unless explicitly enabled |

Important distinction:

- `TurbineGroupBias` always has HOD/DOW/MOY modules internally.
- With default flags, stage2 trains HOD bias only.
- DOW/MOY modules exist but are not used unless their flags are enabled.

## TREE Branch

### Files And Entry Points

Submission branch:

```text
results/submission_tree_lgbm_best_v2_l1.csv
```

Regeneration script:

```text
predict_power_lgbm_best.py
```

Current public-best command shape:

```text
python predict_power_lgbm_best.py \
  --best-csv results/power_lgbm_hyperparams_v2_l1_20_best.csv \
  --feature-profile full_v2 \
  --output results/submission_tree_lgbm_best_v2_l1.csv
```

### Data Used

| Data | Use |
|---|---|
| `data/train/ldaps_train.csv` | Train weather |
| `data/train/gfs_train.csv` | Train weather |
| `data/train/train_labels.csv` | Supervised power targets |
| `data/train/scada_vestas_train.csv` | power-curve feature source for group1/group2 |
| `data/train/scada_unison_train.csv` | power-curve feature source for group3 |
| `data/test/ldaps_test.csv` | Test weather |
| `data/test/gfs_test.csv` | Test weather |
| `data/sample_submission.csv` | Forecast ids and output alignment |

Current public-best TREE does not use the optional SCADA wind-direction teacher:

```text
--use-scada-wd-teacher = off
```

### Feature Profile

Current public-best profile:

```text
full_v2
```

`full_v2` construction:

```text
build_weather_features(ldaps, gfs)
build_meteo_features(ldaps, gfs)
add_meteo_block(..., "all_meteo")
add_compact_physics_features(..., include_advanced=True)
add_power_curve_feature_oof(...)
```

The resulting feature count is large. This is known and is part of the current public-best reproduction, even though later feature-pruning discussion favors quota profiles for new experiments.

### Training Rows

For each group:

```text
x_train, y_train = build_group_dataset(train_weather, labels, group)
mask = y_train >= capacity * min_output_ratio
```

Current best hyperparameter file uses:

```text
min_output_ratio = 0.10
```

So TREE is trained only on target-valid rows:

```text
actual >= 0.10 * capacity
```

### Model And Loss

Model:

```text
LightGBM LGBMRegressor
```

Objective in current best:

```text
regression_l1
```

Sample weight:

```text
actual_sqrt = 0.5 + sqrt(clip(y / capacity, 0, 1))
```

Output clip:

```text
pred = clip(pred, 0, capacity)
```

TREE standalone OOF reference:

| Branch | OOF score | nMAE | FiCR | Worst fold |
|---|---:|---:|---:|---:|
| `full_v2` current TREE | `0.6236110317` | `0.1285047060` | `0.3757267694` | `0.6066260194` |
| `group_family_quota65_v1`, old params | `0.6242589458` | `0.1283303295` | `0.3768482211` | `0.6070145518` |
| `group_family_quota65_v1`, retune | `0.6240879064` | `0.1286602057` | `0.3768360185` | `0.6051383406` |

This TREE standalone OOF comparison is not the same as current-best full blend OOF.

## TCN Branches

### Files And Entry Points

Submission branches:

| Branch | File |
|---|---|
| W24 | `results/submission_seqnn_short_tcn_w24_v1.csv` |
| W72 | `results/submission_seqnn_mid_tcn_w72_v1.csv` |
| W168 | `results/submission_seqnn_long_tcn_w168_v1.csv` |

Regeneration script:

```text
experiments/predict_seqnn_submission.py
```

Command shape:

```text
python experiments/predict_seqnn_submission.py --model tcn --window 24  --stem submission_seqnn_short_tcn_w24_v1
python experiments/predict_seqnn_submission.py --model tcn --window 72  --stem submission_seqnn_mid_tcn_w72_v1
python experiments/predict_seqnn_submission.py --model tcn --window 168 --stem submission_seqnn_long_tcn_w168_v1
```

### Data Used

| Data | Use |
|---|---|
| `data/train/ldaps_train.csv` | Train weather |
| `data/train/gfs_train.csv` | Train weather |
| `data/train/train_labels.csv` | Supervised power targets |
| `data/test/ldaps_test.csv` | Test weather |
| `data/test/gfs_test.csv` | Test weather |
| `data/sample_submission.csv` | Forecast ids and output alignment |

SCADA is not used in the TCN branch.

### Feature Construction

Weather feature profile:

```text
FEATURE_PROFILE_AGGRESSIVE_MINIMAL_CONTEXT_V1
```

Sequence feature list includes seasonal/time, wind speed, gust, cubic/physics, shear, grid summary, lead-time, available-time, and wind-direction sin/cos features.

Examples:

- `sin_doy`, `cos_doy`, `sin_hod`, `cos_hod`
- `gfs_ws100_speed`, `gfs_ws850_speed`, `gfs_ws10_speed`
- `ldaps_ws50_max_speed`, `ldaps_ws10_speed`
- `gfs_surface_0_gust`
- `phys_gfs_air_density_x_gfs_ws850_speed_cube`
- `phys_gfs_air_density_x_gfs_ws100_speed_cube`
- `phys_ldaps_air_density_x_ldaps_ws50_max_speed_cube`
- `phys_shear_*`
- `phys_*_grid_max`, `phys_*_grid_p90`
- `forecast_lead_hours`, `forecast_lead_mod24_sin`, `forecast_lead_mod24_cos`
- `data_available_hod_sin`, `data_available_hod_cos`
- wind-direction sin/cos features

### Year Bagging

For each group and each TCN window:

```text
years = 2022, 2023, 2024
folds:
  train 2022,2023 -> valid 2024 -> predict test
  train 2022,2024 -> valid 2023 -> predict test
  train 2023,2024 -> valid 2022 -> predict test

final branch prediction = mean(test predictions from the 3 folds)
```

Sequence construction:

- Rows are sorted by `forecast_kst_dtm`.
- Each sample uses the previous `window` rows within the same year.
- Early rows are padded by repeating the first available row.
- Features are standardized by `SequenceStandardScaler` fitted on the fold's train sequences.

### Model And Loss

Model:

```text
TCNPowerRegressor
```

Current best default settings:

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
| Grad clip | `1.0` |
| Seed | `42` |

Weighted-L1 loss:

```text
loss = mean(abs(pred_norm - target_norm) * weight)
weight = 0.5 + sqrt(clip(target_norm, 0, 1))
target_norm = target / capacity
```

Defaults that matter:

| Flag | Default | Current best meaning |
|---|---:|---|
| `--target-zero-ffill` | off | zero target ffill not used |
| `--target-valid-only` | off | target-valid-only filtering not used |
| `--loss` | `weighted_l1` | FiCR-only TCN is not current best |

Validation during training:

- After each epoch, predictions on the held-out year are scored with `group_nmae_ficr`.
- Best checkpoint is selected by official-like score.
- Early stopping uses patience `18`.

### TCN Family

The three TCN branches are not interchangeable in the current recipe:

```text
TCN_family =
    0.30 * W24
  + 0.40 * W72
  + 0.30 * W168
```

Interpretation:

- W24: shorter-term/local dynamics.
- W72: main middle-window component.
- W168: longer weekly/context pattern.

## Official-Like Metric Used Internally

Internal scoring helper:

```text
utils.metrics.group_nmae_ficr
```

Important behavior:

```text
valid = actual >= 0.10 * capacity
```

Rows below 10 percent actual are excluded from metric calculation. They are not lifted to 10 percent inside `group_nmae_ficr`.

For valid rows:

```text
error_rate = abs(pred - actual) / capacity
nmae = mean(error_rate)

unit_price =
  4.0 if error_rate <= 0.06
  3.0 if error_rate <= 0.08
  0.0 otherwise

ficr = sum(actual * unit_price) / sum(actual * 4.0)
score = 0.5 * (1 - nmae) + 0.5 * ficr
```

This explains why underprediction in valid actual regions can heavily hurt FiCR even when NMAE looks moderate.

## Current Best Rebuild Steps

These steps rebuild the final submission from existing branch files. They do not retrain models.

1. Apply PINN floor35:

```text
python experiments/apply_metric_floor_submission.py \
  results/submission_pinn_lgbm_teacher_year_bagging.csv \
  --floor-ratio 0.35 \
  --output results/submission_pinn_lgbm_teacher_year_bagging_pinnfloor35.csv \
  --diagnostics-output results/submission_pinn_lgbm_teacher_year_bagging_pinnfloor35_diagnostics.csv
```

2. Blend branches:

```text
python experiments/blend_three_branch_submission.py \
  --pinn results/submission_pinn_lgbm_teacher_year_bagging_pinnfloor35.csv \
  --tree results/submission_tree_lgbm_best_v2_l1.csv \
  --tcn24 results/submission_seqnn_short_tcn_w24_v1.csv \
  --tcn72 results/submission_seqnn_mid_tcn_w72_v1.csv \
  --tcn168 results/submission_seqnn_long_tcn_w168_v1.csv \
  --pinn-weight 0.25 \
  --tree-weight 0.20 \
  --tcn-family-weight 0.55 \
  --tcn24-weight 0.30 \
  --tcn72-weight 0.40 \
  --tcn168-weight 0.30 \
  --output results/submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_prefinalfloor_v1.csv
```

3. Apply final floor10:

```text
python experiments/apply_metric_floor_submission.py \
  results/submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_prefinalfloor_v1.csv \
  --floor-ratio 0.10 \
  --output results/submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_finalfloor10_v1.csv \
  --diagnostics-output results/submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_finalfloor10_v1_diagnostics.csv
```

Do not overwrite `results/submission.csv` unless explicitly requested.

## Known Open Issues

1. Exact OOF component provenance for `pinn_floor_three_branch_grid_v1_summary.csv` is missing.
2. Current public-best submission reconstruction is confirmed, but exact OOF summary reconstruction is not.
3. TREE quota65 has better standalone TREE OOF than full_v2, but current-best full blend impact is not proven.
4. Future quota/proxy/feature-pruning experiments must recreate branch OOF predictions first, then run the same floor and weight sweep method. TREE-only score is not enough.
5. Long experiment details should stay in dedicated docs like this file. `docs/exp_logs.md` should remain a short index.

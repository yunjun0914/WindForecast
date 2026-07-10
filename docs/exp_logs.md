# Experiment Logs

작성일: 2026-07-08 00:46:07 +09:00

> 2026-07-10 note: SCADA LGBM teacher retraining has been removed from executable paths. Historical `lgbm_time_oof` mentions below are legacy result names or old logs, not commands to rerun.

## Log Entries

### 2026-07-10 KST - TREE target-zero ffill validation

Purpose: validate the target-zero ffill hypothesis on the TREE/LGBM branch, where target labels directly affect the fitted regressor.

Important code reality:

- Current best LGBM rows use `min_output_ratio = 0.10` for all groups.
- Therefore raw zero targets are normally excluded from TREE training before fitting.
- This experiment applies ffill to training targets before the existing `min_output_ratio` filter, while validation actuals remain original.

Variants:

- `full`: all zero target runs ffilled within each year.
- `gap2`: only zero target runs of length <= 2 ffilled within each year.
- `min0`: override `min_output_ratio=0.0` to test the actual "LGBM sees raw zero targets and gets suppressed" hypothesis.

Results:

| Feature profile / variant | OOF score | nMAE | FiCR | Worst fold | Judgment |
|---|---:|---:|---:|---:|---|
| aggressive baseline | `0.623952` | `0.128100` | `0.376003` | `0.606055` | reference |
| aggressive full zero ffill | `0.623557` | `0.128230` | `0.375344` | `0.606995` | worse |
| aggressive gap2 zero ffill | `0.623418` | `0.128139` | `0.374974` | `0.606189` | worse |
| aggressive min0 raw zeros | `0.614424` | `0.132481` | `0.361330` | `0.595101` | much worse |
| aggressive min0 full zero ffill | `0.613938` | `0.132461` | `0.360337` | `0.594240` | much worse |
| group_family_quota65 baseline | `0.624259` | `0.128330` | `0.376848` | `0.607015` | reference |
| group_family_quota65 full zero ffill | `0.623747` | `0.128361` | `0.375856` | `0.606755` | worse |
| group_family_quota65 gap2 zero ffill | `0.623653` | `0.128459` | `0.375765` | `0.606694` | worse |

Judgment: reject target-zero ffill for TREE. The strong cutoff check (`min0`) confirms raw zeros would suppress the model, but the current best TREE already avoids that with `min_output_ratio=0.10`. Reintroducing ffilled zero rows adds noise and lowers FiCR/score.

Outputs:

- `results/tree_lgbm_best_v2_l1_aggressive_minimal_rolling_v1_target0ffill_full_v1_*`
- `results/tree_lgbm_best_v2_l1_aggressive_minimal_rolling_v1_target0ffill_gap2_v1_*`
- `results/tree_lgbm_best_v2_l1_aggressive_minimal_rolling_v1_min0_raw_v1_*`
- `results/tree_lgbm_best_v2_l1_aggressive_minimal_rolling_v1_min0_target0ffill_full_v1_*`
- `results/tree_lgbm_best_v2_l1_group_family_quota65_v1_target0ffill_full_v1_*`
- `results/tree_lgbm_best_v2_l1_group_family_quota65_v1_target0ffill_gap2_v1_*`

### 2026-07-10 KST - TCN target-zero ffill validation

Purpose: check whether zero targets in `train_labels` are suppressing TCN predictions and hurting public nMAE.

Audit:

- Zero target rate: group1 `3001/26200 = 11.45%`, group2 `2992/26201 = 11.42%`, group3 `2846/17538 = 16.23%`.
- There are both short isolated zero runs and very long runs over 100 hours, so full zero->ffill is a strong assumption.

Setup:

- Branch: TCN FiCR-only only
- Validation actuals: original labels, not cleaned
- Training target cleaning:
  - `full`: all zero target runs ffilled within each year
  - `gap2`: only zero runs of length <= 2 ffilled within each year

Results:

| Variant | OOF score | nMAE | FiCR | Worst fold | Judgment |
|---|---:|---:|---:|---:|---|
| baseline TCN W24 FiCR-only | `0.627956` | `0.139290` | `0.395203` | `0.618440` | reference |
| W24 full zero ffill | about `0.6257` | - | - | - | worse; full ffill overcorrects |
| W24 gap2 zero ffill | `0.629361` | `0.142029` | `0.400750` | `0.619122` | score up, nMAE worse |
| W72 gap2 zero ffill | `0.629262` | `0.140858` | `0.399382` | `0.618153` | tiny score up, nMAE worse |
| W168 gap2 zero ffill | `0.627289` | `0.142164` | `0.396741` | `0.617913` | worse |
| TCN family gap2 0.30/0.40/0.30 | `0.632660` | `0.139278` | `0.404598` | `0.622509` | below baseline family |
| baseline TCN family FiCR-only | `0.633524` | `0.136600` | `0.403648` | `0.624694` | reference |

Judgment: do not submit this target-cleaning variant. Short zero-run ffill slightly boosts FiCR but worsens nMAE, which is the same failure mode as the public submission. Full ffill is too aggressive because many zero runs look like real long low-output/offline periods.

Outputs:

- `results/oof_oof_seqnn_tcn_w24_ficr_only_target0ffill_gap2_v1.csv`
- `results/oof_oof_seqnn_tcn_w72_ficr_only_target0ffill_gap2_v1.csv`
- `results/oof_oof_seqnn_tcn_w168_ficr_only_target0ffill_gap2_v1.csv`
- `results/oof_oof_tcn_family_ficr_only_target0ffill_gap2_v1.csv`

### 2026-07-10 KST - TREE pseudo2022 + TCN FiCR-only submission v1

Purpose: recover nMAE stability by restoring the TREE group3 pseudo2022 component while keeping the TCN FiCR-only family signal that improved public FiCR.

Setup:

- Server: duck
- TREE component rebuilt:
  - Base: `results/submission_tree_lgbm_best_v2_l1_aggressive_minimal_rolling_v1.csv`
  - group3 only replaced by VESTAS-transfer pseudo2022 TREE
  - pseudo source: VESTAS group1/group2 teachers predict group3 2022 rows
  - pseudo row weight: `0.10`
  - output: `results/submission_tree_lgbm_best_v2_l1_aggressive_minimal_rolling_v1_g3_vestas_pseudo2022_w010_rebuilt.csv`
- TCN family: FiCR-only W24/W72/W168, `0.30/0.40/0.30`
- Final blend: `0.25 * PINN + 0.40 * TREE_pseudo2022 + 0.35 * TCN_FiCR_only_family`
- `results/submission.csv` was not overwritten.

Pseudo TREE stats:

- VESTAS group1 teacher -> group3 2022 mean `7825.04`
- VESTAS group2 teacher -> group3 2022 mean `8513.30`
- averaged pseudo2022: rows `8759`, min `1833.23`, max `20634.53`, mean `8169.17`
- final group3 TREE pseudo model: true rows `9414`, pseudo rows `8745`, features `161`, pred mean `7923.61`

Output:

- `results/submission_pinn25_tree40pseudo_tcn35_ficr_only_family_v1.csv`

Validation:

- Final submission: 8760 rows, 5 columns, 0 missing values.
- Prediction range: min `1689.6573`, max `20894.5204`.

### 2026-07-10 KST - TCN FiCR-only family submission v1

Purpose: create a test submission from the TCN FiCR-only family after the OOF check showed this was the best TCN loss setting.

Setup:

- Server: duck
- TCN family: `0.30 * W24 + 0.40 * W72 + 0.30 * W168`
- SeqNN features: family-pruned profile, 34 features
- TCN loss: `ficr_only`
- Final blends generated:
  - OOF-grid blend: `0.10 * PINN + 0.15 * TREE + 0.75 * TCN_family`
  - Fixed comparison blend: `0.25 * PINN + 0.40 * TREE + 0.35 * TCN_family`
- Component limitation: generated with the PINN/TREE submission files available on duck at run time, not the missing g3/pseudo TREE best file.
- `results/submission.csv` was not overwritten.

OOF reference:

| Variant | OOF score | nMAE | FiCR | Worst fold |
|---|---:|---:|---:|---:|
| TCN FiCR-only family 0.30/0.40/0.30 | `0.633524` | `0.136600` | `0.403648` | `0.624694` |
| 3-branch OOF-grid blend `PINN10/TREE15/TCN75` | `0.635223` | `0.132392` | `0.402837` | `0.624631` |

Outputs:

- `results/submission_seqnn_tcn_w24_ficr_only_v1.csv`
- `results/submission_seqnn_tcn_w72_ficr_only_v1.csv`
- `results/submission_seqnn_tcn_w168_ficr_only_v1.csv`
- `results/submission_pinn10_tree15_tcn75_tcn_ficr_only_family_v1.csv`
- `results/submission_pinn25_tree40_tcn35_tcn_ficr_only_family_v1.csv`

Validation:

- Both final blend submissions: 8760 rows, 5 columns, 0 missing values.
- `submission_pinn10_tree15_tcn75_tcn_ficr_only_family_v1.csv`: min `1839.7857`, max `20958.2357`.
- `submission_pinn25_tree40_tcn35_tcn_ficr_only_family_v1.csv`: min `1633.8728`, max `20894.5204`.

### 2026-07-10 KST - rate_mse_ficr loss OOF v1

Purpose: test a loss that keeps reducing normalized error everywhere while giving stronger pressure near the 6%/8% FiCR settlement bands.

Loss:

```text
error_rate = abs(pred - actual) / capacity
soft_band_loss = 1 - soft_FiCR
loss = 8.0 * mean(error_rate^2) + soft_band_loss
```

Setup:

- PINN: `--data-loss rate_mse_ficr`, RF-OOB teacher cache
- TCN: `--loss rate_mse_ficr`, W24/W72/W168
- No test submission.

Results:

| Variant | OOF score | nMAE | FiCR | Worst fold | Judgment |
|---|---:|---:|---:|---:|---|
| PINN RF-OOB `rate_mse_ficr` | `0.601196` | `0.144639` | `0.347030` | `0.592048` | reject |
| TCN W24 `rate_mse_ficr` | `0.628479` | `0.134108` | `0.391066` | `0.611674` | below FiCR-only |
| TCN W72 `rate_mse_ficr` | `0.628256` | `0.132398` | `0.388910` | `0.616079` | below FiCR-only |
| TCN W168 `rate_mse_ficr` | `0.626360` | `0.133994` | `0.386715` | `0.614276` | reject |
| TCN family 0.30/0.40/0.30 | `0.632071` | `0.130717` | `0.394859` | `0.618804` | below FiCR-only family |
| 3-branch grid best | `0.633299` | `0.128788` | `0.395386` | `0.618794` | below prior TCN FiCR-only grid |

Judgment: formula direction is conceptually better than `ficr_only`, but this v1 weighting is not good. It over-pulls toward nMAE/rate reduction and loses the FiCR gain that pure TCN FiCR-only found. If revisiting, reduce `RATE_MSE_WEIGHT` strongly or use a robust/Huber rate term instead of `8 * rate_mse`.

Outputs:

- `results/pinn_rf_oob_rate_mse_ficr_v1_oof_scores.csv`
- `results/oof_seqnn_tcn_w24_rate_mse_ficr_v1.csv`
- `results/oof_seqnn_tcn_w72_rate_mse_ficr_v1.csv`
- `results/oof_seqnn_tcn_w168_rate_mse_ficr_v1.csv`
- `results/oof_tcn_family_rate_mse_ficr_w24_030_w72_040_w168_030.csv`
- `results/three_branch_oof_blend_rate_mse_ficr_v1_summary.csv`

### 2026-07-10 KST - PINN FiCR-only loss OOF with RF-OOB teacher cache

Purpose: test whether optimizing PINN data loss only for soft FiCR improves FiCR/score after removing the SCADA LGBM teacher path.

Setup:

- Branch: PINN only
- Command stem: `pinn_rf_oob_ficr_only_v1`
- Teacher backend: `rf_oob`
- Teacher cache: enabled, `results/cache/pinn_teacher_weather/`
- Data loss: `ficr_only`
- No test submission.

Result:

| Variant | OOF score | nMAE | FiCR | Worst fold |
|---|---:|---:|---:|---:|
| PINN RF-OOB, metric baseline | `0.612594` | `0.142675` | `0.367863` | `0.605352` |
| PINN RF-OOB, FiCR-only | `0.611915` | `0.142154` | `0.365983` | `0.602294` |

Judgment: rejected. FiCR-only does not improve PINN FiCR and weakens the 2023 fold.

Outputs:

- `results/pinn_rf_oob_ficr_only_v1_oof_scores.csv`
- `results/pinn_rf_oob_ficr_only_v1_oof_predictions.csv`

### 2026-07-10 KST - SCADA teacher LGBM backend removal / cache guard

Purpose: stop accidental repeated SCADA teacher retraining during PINN OOF/submission runs, especially old LGBM teacher paths.

Changes:

- Removed executable SCADA LGBM teacher backend from `utils/pinn_effective_pipeline.py`.
- Locked PINN teacher CLI choices to `rf_oob`.
- Added PINN teacher cache in `utils/pinn_teacher_cache.py`; repeated same train/pred/scada/profile calls reuse `results/cache/pinn_teacher_weather/`.
- Deleted teacher-LGBM experiment/tuning scripts and updated docs so `lgbm_time_oof` is no longer a rerun instruction.
- TREE LGBM branch is unchanged.

No new OOF or submission was created.

### 2026-07-10 20:02:48 +09:00 - PINN teacher family quota pruning OOF

Purpose: test whether SCADA wind teacher inputs can be pruned after TREE family quota pruning worked.

Setup:

- Branch: PINN only
- Teacher backend: `lgbm_time_oof`
- New teacher feature profile: `family_quota_v1`
- Important correction: the earlier `current_teacher_feature_importance_v1` was based on teacher tables without the OOF path's `all_meteo` block. Therefore teacher importance was recomputed with the actual OOF path before defining the quota.
- Actual full teacher feature counts:
  - group1 effective teacher: 129
  - canonical group2/group3 teacher: 95
- Pruned feature counts:
  - group1 effective teacher: 36
  - canonical group2/group3 teacher: 28
- No test submission created.

Actual teacher importance summary:

- Canonical group2/group3 teacher still concentrates on `forecast_wind` + `other`, but `raw_meteo` is not negligible at about `10%` gain.
- Effective group1 teacher is led by `effective_wind`, then `other`, `forecast_wind`, and `raw_meteo`.
- The first quota draft without raw meteo was not used for OOF; `family_quota_v1` includes raw-meteo quota.

Result:

| PINN teacher profile | Mean score | Mean nMAE | Mean FiCR |
|---|---:|---:|---:|
| full `lgbm_time_oof` baseline | `0.612593` | `0.142675` | `0.367861` |
| `family_quota_v1` | `0.611552` | `0.141795` | `0.364899` |

Fold deltas:

| pred year | baseline | quota | delta |
|---|---:|---:|---:|
| 2024 | `0.617821` | `0.625172` | `+0.007351` |
| 2023 | `0.605349` | `0.600754` | `-0.004595` |
| 2022 | `0.614610` | `0.608730` | `-0.005880` |

Group deltas:

- 2024 improved across all groups, especially group3 (`+0.0140`).
- 2023/2022 group1 degraded strongly (`-0.0118`, `-0.0075`), which erased the gain.
- group3 was nearly flat in 2023 and has no 2022 label fold, so the useful signal may be group3-specific rather than global.

Judgment: **global teacher pruning is rejected.** Do not replace the full PINN teacher profile globally. The result suggests a narrower follow-up candidate: apply teacher pruning only to the group3 teacher/proxy path or design a wider group1-effective quota. Do not make a test submission from this result.

Outputs:

- `results/current_teacher_feature_importance_actual_v1_teacher_aggregated.csv`
- `results/current_teacher_feature_importance_actual_v1_teacher_family.csv`
- `results/current_teacher_feature_importance_actual_v1_teacher_stat_ops.csv`
- `results/pinn_lgbm_time_oof_teacher_family_quota_v1_oof_scores.csv`
- `results/pinn_lgbm_time_oof_teacher_family_quota_v1_oof_predictions.csv`

### 2026-07-10 19:35:44 +09:00 - TREE group family quota65 feature OOF

Purpose: test family-balanced TREE pruning after `group_top40_v1` showed that hard top-k removes too much context.

Setup:

- Branch: TREE only
- New feature profile: `group_family_quota65_v1`
- Source importance: `aggressive_minimal_rollmean_v1` TREE gain importance
- Selection rule per group:
  - `spatial_wind`: 18
  - `forecast_wind`: 12, with mandatory `gfs_ws100_speed` for `power_curve_est`
  - `time_context`: 10
  - `physics`: 9
  - `calendar`: 7
  - `raw_meteo`: 4
  - `other`: 4
- Feature count: 64 weather features per group, then existing `power_curve_est` is added as the 65th model input
- Validation: leave-one-year-out OOF
- No test submission created

Result:

| TREE profile | Mean score | Mean nMAE | Mean FiCR | Worst fold |
|---|---:|---:|---:|---:|
| `aggressive_minimal_rollmean_v1` 98-input baseline | `0.623916` | `0.128277` | `0.376109` | `0.606363` |
| `group_top40_v1`, old best params | `0.622757` | `0.129022` | `0.374537` | `0.605590` |
| `group_family_quota65_v1`, old best params | `0.624259` | `0.128330` | `0.376848` | `0.607015` |
| `group_family_quota65_v1`, focused L1 12-trial retune | `0.624088` | `0.128660` | `0.376836` | `0.605138` |

Group mean comparison:

| Group | baseline 98 | family quota65 old params | family quota65 retune |
|---|---:|---:|---:|
| group1 | `0.622019` | `0.622053` | `0.621627` |
| group2 | `0.650142` | `0.650799` | `0.651992` |
| group3 | `0.580650` | `0.580932` | `0.577888` |

Judgment: **candidate, not submission reason by itself.** Family-balanced pruning finally beats the 98-input baseline slightly with old best params, and worst fold also improves. The short retune overfits/tilts toward group2 and hurts group3, so keep the old-param version as the candidate. This supports the diagnosis that TREE needed context preservation, not a hard top-k cut. The gain is small (`+0.00034` mean OOF), so do not make a test submission from this alone.

Outputs:

- `results/power_lgbm_best_v2_l1_group_family_quota65_v1_scores.csv`
- `results/power_lgbm_best_v2_l1_group_family_quota65_v1_summary.csv`
- `results/power_lgbm_hyperparams_group_family_quota65_v1_l1_12_best.csv`
- `results/power_lgbm_best_group_family_quota65_v1_l1_12_scores.csv`
- `results/power_lgbm_best_group_family_quota65_v1_l1_12_summary.csv`

### 2026-07-10 19:26:46 +09:00 - TREE group-specific top40 feature OOF

Purpose: test the user's suggestion to prune TREE features by group-specific importance top-k instead of one shared lean profile.

Setup:

- Branch: TREE only
- New feature profile: `group_top40_v1`
- Source importance: `aggressive_minimal_rollmean_v1` TREE gain importance
- Selection rule: per group top 39 by gain + mandatory `gfs_ws100_speed` for `power_curve_est`
- Feature count: 40 weather features per group, then existing `power_curve_est` is added as the 41st model input
- Validation: leave-one-year-out OOF
- No test submission created

Result:

| TREE profile | Mean score | Mean nMAE | Mean FiCR | Worst fold |
|---|---:|---:|---:|---:|
| `aggressive_minimal_rollmean_v1` 98-input baseline | `0.623916` | `0.128277` | `0.376109` | `0.606363` |
| `group_top40_v1`, old best params | `0.622757` | `0.129022` | `0.374537` | `0.605590` |
| `group_top40_v1`, focused L1 12-trial retune | `0.622230` | `0.128363` | `0.372822` | `0.604703` |

Group mean comparison:

| Group | baseline 98 | top40 old params | top40 retune |
|---|---:|---:|---:|
| group1 | `0.622019` | `0.621310` | `0.618777` |
| group2 | `0.650142` | `0.648327` | `0.650076` |
| group3 | `0.580650` | `0.579378` | `0.577969` |

Judgment: **rejected as a replacement.** Group-specific top-k is a better pruning concept than a shared 30-feature profile, but `k=40` still removes too much TREE context. Group2 nearly recovers after retune and gets slightly higher FiCR, but nMAE loss cancels it; group1 and group3 lose more clearly. Keep `aggressive_minimal_rollmean_v1` as the current TREE base. Next pruning attempt should use a larger group-specific budget such as top60/top70 or family quotas, not a hard 40.

Outputs:

- `results/power_lgbm_best_v2_l1_group_top40_v1_scores.csv`
- `results/power_lgbm_best_v2_l1_group_top40_v1_summary.csv`
- `results/power_lgbm_hyperparams_group_top40_v1_l1_12_best.csv`
- `results/power_lgbm_best_group_top40_v1_l1_12_scores.csv`
- `results/power_lgbm_best_group_top40_v1_l1_12_summary.csv`

### 2026-07-10 KST - TREE wind cubic minimal profile OOF

Purpose: test the hypothesis that TREE still has too many noisy features and should be reduced to about 20-30 wind/energy-focused inputs, with raw grid-cubic features aligned to the SCADA `scada_ws_cubic` target.

Setup:

- Branch: TREE only
- New feature profile: `wind_cubic_minimal_v1`
- Feature count: 29 weather features before SCADA power curve, 30 model inputs after `power_curve_est`
- Added raw grid-cubic features:
  - `phys_ldaps_ws50max_grid_cubic`
  - `phys_ldaps_ws50min_grid_cubic`
  - `phys_ldaps_ws10_grid_cubic`
  - `phys_gfs_ws100_grid_cubic`
  - `phys_gfs_ws850_grid_cubic`
  - `phys_gfs_ws10_grid_cubic`
- Dropped broad radiation/cloud/temp/raw-pressure/raw-humidity and most grid/time statistics
- Validation: leave-one-year-out OOF
- No test submission created

Feature set:

```text
sin_doy, cos_doy, sin_hod, cos_hod, lead_hour,
ldaps_ws10_speed, ldaps_ws50_max_speed, ldaps_ws50_min_speed,
gfs_ws100_speed, gfs_ws850_speed, gfs_surface_0_gust,
phys_ldaps_ws50max_grid_mean, phys_ldaps_ws50max_grid_max, phys_ldaps_ws50max_grid_p90,
phys_ldaps_ws50max_grid_cubic, phys_ldaps_ws50min_grid_cubic, phys_ldaps_ws10_grid_cubic,
phys_gfs_ws100_grid_cubic, phys_gfs_ws850_grid_cubic, phys_gfs_ws10_grid_cubic,
phys_gfs_air_density_x_gfs_ws100_speed_cube,
phys_gfs_air_density_x_gfs_ws850_speed_cube,
phys_ldaps_air_density_x_ldaps_ws50_max_speed_cube,
phys_shear_gfs_100_10, phys_shear_ldaps_50max_10, phys_gfs_gust_factor,
ldaps_ws50_max_speed_lead1, ldaps_ws50_max_speed_lead3, ldaps_ws50_max_speed_roll3_mean
```

Result:

| Run | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---:|---:|---:|---:|
| old tuned params on new profile | `0.607812` | `0.135956` | `0.351580` | `0.590043` |
| short retune, focused L1 12 trials | `0.606170` | `0.135454` | `0.347793` | `0.587401` |

Judgment: this exact 30-input profile is rejected. The philosophy is still plausible, but this cut is too aggressive and/or missing important non-cubic signals. Retuning did not recover the loss. The likely mistake is removing too much directional/vector/raw wind structure while keeping only energy/cubic summaries. A better next profile should stay lean but use importance-ranked survivors from the 98-input rollmean profile rather than hand-picking only cubic/energy features.

Outputs:

- `utils/compact_physics_features.py`
- `utils/tree_feature_profiles.py`
- `results/power_lgbm_best_v2_l1_wind_cubic_minimal_v1_summary.csv`
- `results/power_lgbm_hyperparams_wind_cubic_minimal_v1_l1_12_best.csv`
- `results/power_lgbm_best_wind_cubic_minimal_v1_l1_12_summary.csv`

### 2026-07-10 KST - TREE input test with lead/month calibrated wind

Purpose: test whether the raw wind reconstruction gain from lead/month calibration improves power prediction when added as a TREE input.

Setup:

- Data used: raw train weather, raw SCADA wind for calibration, train labels for OOF
- Previous OOF/submission result files were not used
- Calibration: no group conditioning, fit only on each fold's train years
- Feature profile: `aggressive_minimal_rollmean_v1`
- Model checks:
  - `lgbm_tuned`
  - `lgbm_regularized`
- Train policy: `metric_valid`
- Sample weight: `actual_sqrt`
- Variants:
  - `baseline`
  - `calibrated_ws`: add `lmcal_ldaps_ws50max` and related raw/global/offset/cube fields
  - `calibrated_ws_curve`: also add a group power-curve feature from `lmcal_ldaps_ws50max`
- No test submission created

Result:

| Model | Variant | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---|---:|---:|---:|---:|
| `lgbm_tuned` | `baseline` | `0.622516` | `0.128498` | `0.373529` | `0.605024` |
| `lgbm_tuned` | `calibrated_ws` | `0.621700` | `0.128401` | `0.371801` | `0.604273` |
| `lgbm_tuned` | `calibrated_ws_curve` | `0.621432` | `0.128510` | `0.371374` | `0.604736` |
| `lgbm_regularized` | `baseline` | `0.623737` | `0.128439` | `0.375914` | `0.607955` |
| `lgbm_regularized` | `calibrated_ws` | `0.623251` | `0.128346` | `0.374848` | `0.607272` |
| `lgbm_regularized` | `calibrated_ws_curve` | `0.623103` | `0.128555` | `0.374761` | `0.607082` |

Judgment: direct TREE feature injection is rejected for now. The calibrated wind signal improves raw SCADA wind reconstruction and slightly improves nMAE in TREE, but it reduces FiCR more than it helps nMAE. The effect is mixed by group: group2 often benefits, group1 is mostly hurt, and group3 is unstable. The signal is still useful, but likely needs to be used as a teacher/regime feature or group-specific power conversion rather than a plain extra TREE input.

Outputs:

- `experiments/evaluate_tree_lead_month_calibrated_ws.py`
- `results/tree_lead_month_calibrated_ws_v1_summary.csv`
- `results/tree_lead_month_calibrated_ws_v1_scores.csv`
- `results/tree_lead_month_calibrated_ws_v1_predictions.csv`
- `results/tree_lead_month_calibrated_ws_v1_calibrators.csv`
- `results/tree_lead_month_calibrated_ws_v1_regularized_summary.csv`
- `results/tree_lead_month_calibrated_ws_v1_regularized_scores.csv`
- `results/tree_lead_month_calibrated_ws_v1_regularized_predictions.csv`
- `results/tree_lead_month_calibrated_ws_v1_regularized_calibrators.csv`

### 2026-07-10 KST - Lead/month wind calibration raw diagnostic

Purpose: check a new hypothesis without using previous OOF/submission results: forecast wind may have systematic bias by forecast lead slot and month.

Setup:

- Data used: raw train LDAPS/GFS weather + raw SCADA wind only
- No existing model predictions, submissions, or previous score files used
- Target: hourly SCADA wind speed
  - `vestas`: all VESTAS turbines, full 2022-2024 coverage
  - `unison`: all UNISON turbines, 2023-2024 coverage
  - `all`: available VESTAS/UNISON mean, exploratory only
- Calibration variables: `lead_hour` and `month` only
- No group-conditioned calibration
- Validation: leave-one-year-out by raw SCADA year
- Main weather proxy: `ldaps_ws50max_grid_mean`
- Methods compared:
  - raw weather wind
  - global affine calibration
  - lead residual
  - month residual
  - lead-month residual
  - lead-month affine

Key result for `ldaps_ws50max_grid_mean`:

| Target | Global affine MAE | Best calibrated MAE | Delta | Delta % |
|---|---:|---:|---:|---:|
| `vestas` | `1.678479` | `1.492800` | `0.185678` | `11.06%` |
| `unison` | `1.655040` | `1.549839` lead-only (`1.557864` lead-month) | `0.105201` | `6.36%` |
| `all` | `1.676040` | `1.507254` | `0.168786` | `10.07%` |

Observed pattern on VESTAS full-train offsets:

- Early lead slots `12-17` are generally underpredicted after global affine and need roughly `+0.6 m/s` residual correction.
- Mid lead slots `23-27` are generally overpredicted and need roughly `-0.9 m/s` residual correction.
- Seasonal effect is strong: January/December positive offsets, July/August negative offsets.
- Lead-only already helps; lead-month is best for VESTAS/all, while UNISON has less data and lead-only is slightly more stable.

Judgment: accepted as a real high-leverage signal. This is not a final power-score result yet, but raw wind reconstruction improves by about `6-11%` MAE without group conditioning. Next useful step is to turn `lead_month_calibrated_ws` into a reusable feature and test it inside TREE/PINN with strict train-fold fitting.

Outputs:

- `results/lead_month_wind_calibration_v1_scores.csv`
- `results/lead_month_wind_calibration_v1_summary.csv`
- `results/lead_month_wind_calibration_v1_cell_counts.csv`
- `results/lead_month_wind_calibration_v1_full_offsets.csv`

### 2026-07-10 KST - PINN teacher residual bias OOF

Purpose: test the previously discussed `physical equation + learned bias + NN residual bias from teacher` path.

Setup:

- Branch: PINN OOF, then diagnostic 3-branch OOF blend
- Base PINN: effective-grid g1 year bagging with `lgbm_time_oof` teacher backend
- Residual structure: small NN residual bias added after physical prediction and existing bias
- Residual inputs: SCADA wind teacher summary features such as `scada_ws_mean`, cubic/quantile wind stats, wind std/ramp, and direction concentration/spread features
- Residual hyperparameters: hidden `16`, amplitude `0.05` capacity fraction, lr `5e-4`, L2 `0.01`
- Validation: leave-one-year-out OOF
- No test submission created

PINN-only result:

| Variant | Mean score | Mean nMAE | Mean FICR |
|---|---:|---:|---:|
| baseline `lgbm_time_oof` PINN | `0.612593` | `0.142675` | `0.367861` |
| PINN + teacher residual bias | `0.613323` | `0.140325` | `0.366970` |

3-branch OOF blend using residual PINN + current TREE rolling + TCN family:

| Blend | Weights | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---|---:|---:|---:|---:|
| fixed current | `PINN=0.25, TREE=0.40, TCN=0.35` | `0.630254` | `0.126700` | `0.387207` | `0.614173` |
| best grid | `PINN=0.20, TREE=0.40, TCN=0.40` | `0.630770` | `0.126735` | `0.388275` | `0.614484` |

Judgment: direction is valid but the gain is small. The residual teacher path improves PINN nMAE clearly, but slightly hurts FiCR. In the full ensemble it is a candidate component, especially with a little less PINN weight and more TCN weight, but it is not a standalone submission reason. It also does not beat the earlier group3-only SCADA teacher TREE blend by itself.

Outputs:

- `results/pinn_lgbm_time_oof_teacherres_h16_a0p05_oof_scores.csv`
- `results/pinn_lgbm_time_oof_teacherres_h16_a0p05_oof_predictions.csv`
- `results/pinn_lgbm_time_oof_teacherres_h16_a0p05_oof_long.csv`
- `results/pinn_teacherres_h16_a0p05_tree_rolling_tcnfamily_oof_summary.csv`
- `results/pinn_teacherres_h16_a0p05_tree_rolling_tcnfamily_oof_scores.csv`
- `results/pinn_teacherres_h16_a0p05_tree_rolling_tcnfamily_oof_best_oof.csv`

### 2026-07-10 KST - TREE multi-horizon target OOF (1h/3h/5h)

Purpose: test whether smoothed power targets improve low-output stability while keeping the hourly TREE model as the main branch.

Setup:

- Branch: TREE only
- Feature profile: `aggressive_minimal_rollmean_v1` (98 features)
- Hyperparameters: fixed current tuned LGBM v2_l1 best rows
- Targets:
  - `h1`: raw hourly target `y[t]`
  - `h3`: centered 3-hour mean target
  - `h5`: centered 5-hour mean target
- Centered targets are built inside each train fold and split by year, so validation labels do not enter train targets.
- Evaluation: original hourly labels, leave-one-year-out OOF
- Blend search: nonnegative `h1/h3/h5` weights, step `0.05`
- No test submission created

Individual horizon result:

| Variant | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---:|---:|---:|---:|
| `h1` | `0.623916` | `0.128277` | `0.376109` | `0.606363` |
| `h3` | `0.621904` | `0.128229` | `0.372036` | `0.603249` |
| `h5` | `0.619356` | `0.128101` | `0.366812` | `0.600707` |

Blend result:

| Blend | Weights | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---|---:|---:|---:|---:|
| best global | `h1=0.90, h3=0.00, h5=0.10` | `0.624080` | `0.128063` | `0.376223` | `0.606698` |
| group best | g1 `0.90/0.00/0.10`, g2 `0.55/0.25/0.20`, g3 `0.85/0.00/0.15` | `0.624113` | `0.127937` | `0.376163` | `0.606475` |

Low-output bin (`actual/capacity` 10-30%) MAE per capacity:

| Variant | g1 | g2 | g3 |
|---|---:|---:|---:|
| `h1` | `0.11490` | `0.13112` | `0.13221` |
| `h3` | `0.11368` | `0.13022` | `0.13022` |
| `h5` | `0.11232` | `0.12796` | `0.12777` |
| group best blend | `0.11448` | `0.12989` | `0.13122` |

Judgment: direction is valid but small. Smoothed targets improve low-output MAE and reduce positive low-output bias, but standalone `h3/h5` lose FICR due to peak smoothing. Keep multi-horizon as a small blend feature, not as a replacement for hourly target. Strongest signal is group2, where the best OOF weights use `45%` smoothed target contribution.

Outputs:

- `results/tree_multi_horizon_target_1_3_5_rollmean_v1_individual_summary.csv`
- `results/tree_multi_horizon_target_1_3_5_rollmean_v1_global_blend_summary.csv`
- `results/tree_multi_horizon_target_1_3_5_rollmean_v1_group_best_weights.csv`
- `results/tree_multi_horizon_target_1_3_5_rollmean_v1_group_blend_summary.csv`
- `results/tree_multi_horizon_target_1_3_5_rollmean_v1_bin_metrics.csv`
- `results/tree_multi_horizon_target_1_3_5_rollmean_v1_predictions_wide.csv`

### 2026-07-10 KST - TREE rollmean feature pruning OOF

Purpose: reduce noisy TREE time-context statistics while keeping the current best tuned LGBM hyperparameters fixed.

Setup:

- Base profile: `aggressive_minimal_rolling_v1` (161 features per group)
- New profile: `aggressive_minimal_rollmean_v1` (98 features per group)
- Kept time-context features: `lead1`, `lead3`, `roll3_mean`
- Dropped most weak time statistics: `lag1`, `lag3`, `roll3_std`, `roll6_mean`, `roll6_std`, `roll6_max`, `lead1_minus_lag1`
- Validation: leave-one-year-out OOF
- No test submission created

Result:

| TREE profile | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---:|---:|---:|---:|
| `aggressive_minimal_rolling_v1` repro | `0.623952` | `0.128100` | `0.376003` | `0.606055` |
| `aggressive_minimal_rollmean_v1` | `0.623916` | `0.128277` | `0.376109` | `0.606363` |

Judgment: pruning is accepted as a clean slim profile candidate. Mean score is essentially flat (`-0.000036`), FiCR and worst fold are slightly better, and feature count drops from 161 to 98 columns. This is useful as the next base for stronger wind/teacher features, but it is not a standalone submission reason.

Outputs:

- `results/power_lgbm_best_v2_l1_aggressive_minimal_rollmean_v1_scores.csv`
- `results/power_lgbm_best_v2_l1_aggressive_minimal_rollmean_v1_predictions.csv`
- `results/power_lgbm_best_v2_l1_aggressive_minimal_rollmean_v1_summary.csv`

### 2026-07-09 04:09:49 +09:00 - PINN + TREE + TCN family OOF blend

목적: PINN, TREE, TCN family를 같은 OOF 기준에서 3-branch로 섞어 상호보완이 있는지 확인한다.

고정:

- TCN family: W24 `0.30`, W72 `0.40`, W168 `0.30`
- branch weight grid: `0.05`
- validation: leave-one-year-out OOF
- test submission 생성 없음

결과:

| Variant | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---:|---:|---:|---:|
| TREE only | `0.62396` | `0.12809` | `0.37602` | `0.60563` |
| TCN family only | `0.62084` | `0.13797` | `0.37966` | `0.61141` |
| PINN only | `0.61293` | `0.14197` | `0.36783` | `0.60596` |
| PINN `0.25` + TREE `0.40` + TCN family `0.35` | `0.63088` | `0.12760` | `0.38937` | `0.61667` |

판단: 3-branch ensemble은 유효하다. TREE only 대비 `+0.00692`, 기존 TCN W72 best blend `0.63021` 대비 `+0.00068`. FICR 상승이 크지만 group3는 여전히 약점이라 추가 개선은 group3 데이터/teacher/feature 쪽이 필요하다.

문서:

- `docs/pinn_tree_tcn_family_oof_blend.md`

### 2026-07-09 03:58:29 +09:00 - TCN window comparison W24/W72/W168

목적: TCN branch가 window 길이에 대해 일관적으로 유효한지 확인한다.

고정:

- model: TCN, 4 layers, hidden 64, kernel 3
- input features: 34
- validation: leave-one-year-out OOF
- TREE base: `aggressive_minimal_rolling_v1`
- test submission 생성 없음

결과:

| Model | Single score | Best TREE+TCN score | Best weight |
|---|---:|---:|---:|
| TCN W24 | `0.61415` | `0.62775` | `0.30` |
| TCN W72 | `0.61521` | `0.63021` | `0.40` |
| TCN W168 | `0.61220` | `0.62876` | `0.30` |

판단: TCN family는 일관적으로 유효하다. W72가 단독/블렌드 기준 best. W168은 lower residual correlation이지만 품질이 낮아 W72를 넘지 못함. 다음 후보는 W72 TCN seed bagging/light tuning 또는 W72/W168 TCN ensemble.

문서:

- `docs/seqnn_tcn_window_comparison.md`

### 2026-07-09 03:48:27 +09:00 - SeqNN mid TCN W72 v1 OOF

목적: 최근 72시간 weather sequence를 TCN으로 보면서 short GRU보다 더 넓은 ramp/weather pattern 신호가 있는지 확인한다.

고정:

- 기존 PINN/TREE 제출 경로 변경 없음
- test submission 생성 없음
- validation: leave-one-year-out OOF
- target history/raw SCADA feature 금지
- early stopping 사용

결과:

| Variant | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---:|---:|---:|---:|
| TREE rolling baseline | `0.62396` | `0.12809` | `0.37602` | `0.60563` |
| SeqNN mid TCN W72 | `0.61521` | `0.14078` | `0.37120` | `0.60283` |
| TREE 60% + TCN 40% | `0.63021` | `0.12714` | `0.38756` | `0.61593` |

판단: 현재 SeqNN 후보 중 가장 유망하다. TREE blend에서 `+0.00624`, 모든 group 개선, FICR 개선이 크다. test submission은 아직 만들지 않고 TCN seed/tuning 또는 PINN/TREE/TCN OOF blend를 다음 후보로 둔다.

문서:

- `docs/seqnn_mid_tcn_w72_v1.md`

### 2026-07-09 03:35:28 +09:00 - SeqNN long DLinear W168 v1 OOF

목적: 최근 168시간 weather sequence를 보는 long-window branch가 TREE와 다른 long-term regime 신호를 주는지 확인한다.

고정:

- 기존 PINN/TREE 제출 경로 변경 없음
- test submission 생성 없음
- validation: leave-one-year-out OOF
- target history/raw SCADA feature 금지
- early stopping 사용

결과:

| Variant | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---:|---:|---:|---:|
| TREE rolling baseline | `0.62396` | `0.12809` | `0.37602` | `0.60563` |
| SeqNN long DLinear W168 | `0.54953` | `0.17689` | `0.27595` | `0.53806` |
| TREE 95% + DLinear 5% | `0.62387` | `0.12798` | `0.37572` | `0.60573` |

판단: residual correlation은 lower overall `0.68020`이지만 단독 품질이 너무 낮아 blend에 도움이 안 된다. DLinear W168 v1은 기각. long-window 아이디어는 GRU/TCN 또는 shared model로 재검토.

문서:

- `docs/seqnn_long_dlinear_w168_v1.md`

### 2026-07-09 03:28:12 +09:00 - SeqNN short GRU W24 v1 OOF

목적: 최근 24시간 weather sequence를 직접 보는 GRU branch가 TREE와 다른 예측 신호를 주는지 확인한다.

고정:

- 기존 PINN/TREE 제출 경로 변경 없음
- test submission 생성 없음
- validation: leave-one-year-out OOF
- target history/raw SCADA feature 금지
- early stopping 사용

결과:

| Variant | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---:|---:|---:|---:|
| TREE rolling baseline | `0.62396` | `0.12809` | `0.37602` | `0.60563` |
| SeqNN short GRU W24 | `0.61323` | `0.13700` | `0.36347` | `0.59955` |
| TREE 70% + SeqNN 30% | `0.62768` | `0.12571` | `0.38107` | `0.60973` |

판단: SeqNN 단독은 약하지만 TREE blend에서 `+0.00372`가 나왔다. 모든 group이 소폭 개선했다. residual correlation은 높으므로 바로 제출하지 말고 PINN/TREE/SeqNN 3-branch OOF blend와 long-window branch를 다음 후보로 본다.

문서:

- `docs/seqnn_short_gru_w24_v1.md`

### 2026-07-09 02:02:30 +09:00 - model branch comparison plan

목적: TREE만 강화하지 않고, short/long window SeqNN branch를 PINN/TREE와 같은 OOF 기준으로 비교하는 계획을 고정한다.

결정:

- 현재 위치: Phase 4B, model branch comparison
- TREE 기준: `aggressive_minimal_rolling_v1`
- SeqNN은 residual이 아니라 direct forecast sibling
- short window: 24h GRU/TCN, ramp/FICR 목적
- long window: 168h DLinear/GRU, NMAE/generalization 목적
- 입력 feature는 20-40개 small set으로 시작
- test submission은 만들지 않음

문서:

- `docs/model_branch_comparison_plan.md`

다음 권장 구현: `seqnn_short_gru_w24_v1` OOF.

### 2026-07-09 01:50:58 +09:00 - TREE context profile: lead/cycle + direction + spatial

목적: 대회/연구 코드에서 남은 후보 1/3/4를 rolling profile 위에 compact하게 추가한다.

고정:

- 모델: tuned group LGBM v2
- 검증: year OOF
- 제출: 생성하지 않음
- 기존 `aggressive_minimal_rolling_v1` 유지

변경:

- 새 profile: `aggressive_minimal_context_v1`
- feature count: `211`
- 추가 family: forecast lead/cycle, normalized direction/veer, compact spatial q25/q75/IQR/std

결과:

| Model | Feature count | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---:|---:|---:|---:|---:|
| aggressive_minimal_rolling_v1 TREE | 160 | `0.62396` | `0.12809` | `0.37602` | `0.60563` |
| aggressive_minimal_context_v1 TREE | 211 | `0.62343` | `0.12794` | `0.37481` | `0.60494` |

판단: NMAE는 좋아졌지만 FICR이 하락했다. 세 family를 한번에 붙이는 방식은 제출 후보가 아니며, 다음에는 lead/cycle, direction, spatial을 분리 ablation해야 한다.

### 2026-07-09 01:39:51 +09:00 - TREE rolling time-context profile

목적: 외부 대회/연구 코드에서 반복되는 weather lag/lead/rolling feature를 70개 minimal TREE profile 위에 추가한다.

고정:

- 모델: tuned group LGBM v2
- 검증: year OOF
- 제출: 생성하지 않음
- 기존 `full_v2`, `aggressive_minimal_v1` 유지

변경:

- 새 profile: `aggressive_minimal_rolling_v1`
- feature count: `160`
- 추가 family: 주요 wind/weather `lag1/lag3/lead1/lead3/roll3/roll6`

결과:

| Model | Feature count | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---:|---:|---:|---:|---:|
| aggressive_minimal_v1 TREE | 70 | `0.62085` | `0.12980` | `0.37150` | `0.60369` |
| aggressive_minimal_rolling_v1 TREE | 160 | `0.62396` | `0.12809` | `0.37602` | `0.60563` |
| full_v2 TREE | 511 | `0.62361` | `0.12851` | `0.37573` | - |

판단: rolling family는 유효하다. 큰 점프는 아니지만 minimal profile의 약점을 메우며 full feature control보다도 소폭 좋다. test submission은 만들지 않는다.

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
| PINN/TREE + inverse residual TREE | `0.63140` | `0.13194` | `0.39474` | 2026-07-08, OOF only; residual mean of XGB/Extra, alpha `-0.50`, clip `0.08`; promising but no test submission yet |
| TREE, group3 global LGBM blend | `0.62482` | `0.12795` | `0.37760` | 2026-07-08 09:21:30, group3-only global weight `0.75`; small gain over tuned TREE `0.62361` |
| TREE, power-curve proxy replacement | `0.62411` | `0.12851` | `0.37673` | 2026-07-08, group-best proxy; small gain over tuned TREE `0.62361`, no test submission |
| TREE family ensemble, LGBM/XGB/ExtraTrees | `0.62361` | `0.12851` | `0.37573` | 2026-07-08 10:05:43, coarse weight grid selected LGBM `1.0`, XGB/Extra `0.0` |
| TREE, family-level feature pruning | `0.62361` | `0.12851` | `0.37573` | 2026-07-08, full 511 features still best overall; group-specific pruning gives tiny gains only |
| PINN/TREE blend, best validation weight near tree=0.4 | `0.62481` | `0.12908` | `0.37870` | validation best |
| PINN50:TREE50 public submission | `0.62423` | - | - | best confirmed public candidate |

## Kept Ideas

These are useful but not main pipeline code.

| Idea | Current judgment |
|---|---|
| Tree hyperparameter search | useful; v2 focused L1 search improved TREE OOF by about `+0.012` |
| Group-specific tree tuning | confirmed useful |
| Sample-weight policy search | useful; `actual_sqrt` often won in v2 |
| Global stacked tree for group3 | small positive; group3 improved, but total TREE gain only about `+0.0012` |
| XGB/ExtraTrees diversity ensemble | no gain yet; tuned LGBM still dominates, revisit only after stronger non-LGBM models |
| Inverse residual TREE correction | promising OOF; likely acts as overcorrection/uncertainty correction, not literal residual addition |
| Alternative power-curve wind proxy | small positive; `gfs_ws100` power curve was weak, LDAPS/GFS proxy changes help slightly but not submission-worthy |
| Family-level feature pruning | good for interpretability; overall OOF did not beat full feature set yet |
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

---

작성일: 2026-07-10 16:20:42 +09:00

실험명: SCADA wind translator RF OOB TREE ablation

목적: forecast wind를 site/turbine wind로 복원하는 SCADA-supervised teacher가 TREE OOF와 3-branch OOF blend를 올리는지 확인. LGBM teacher는 기존 TREE teacher 계열과 다르고 너무 무거워 중단했으며, RF OOB teacher로 재실행.

설정:

- base feature: `aggressive_minimal_rolling_v1`
- model: 기존 LGBM L1 best hyperparams 고정
- variants:
  - `baseline`
  - `teacher:rf_oob:group_default`
  - `teacher_pc:rf_oob:group_default`
- `teacher_pc`는 predicted SCADA wind `v`로 empirical power curve feature를 추가.
- test submission 생성 없음.

결과:

| variant | mean score | mean nMAE | mean FICR | worst fold |
|---|---:|---:|---:|---:|
| baseline | `0.623952` | `0.128100` | `0.376003` | `0.606055` |
| teacher_pc all groups | `0.623195` | `0.128572` | `0.374961` | `0.605741` |
| teacher all groups | `0.622337` | `0.128854` | `0.373527` | `0.605374` |

group별:

| group | baseline | teacher_pc | delta | 해석 |
|---|---:|---:|---:|---|
| group1 | `0.623013` | `0.622431` | `-0.000582` | 하락 |
| group2 | `0.650090` | `0.647820` | `-0.002270` | 하락 |
| group3 | `0.578409` | `0.581771` | `+0.003362` | 개선. FiCR `0.300989 -> 0.308229` |

추가 blend 확인:

- group1/group2는 baseline 유지, group3만 `teacher_pc:rf_oob` 적용한 TREE OOF 생성:
  - `results/standard_oof_tree_lgbm_rf_scada_wind_teacherpc_group3_only_v1.csv`
- 기존 PINN/TCN family와 3-branch OOF:
  - fixed `pinn0.25_tree0.40_tcn0.35`: `0.630453` (기존 재현 fixed `0.630185` 대비 `+0.000268`)
  - grid best: `pinn0.15_tree0.60_tcn0.25` `0.631212` (기존 재현 grid best `0.630514` 대비 `+0.000698`)

판단:

- **전체 group에 SCADA wind teacher를 붙이는 것은 기각.**
- **group3-only SCADA wind teacher_pc는 후보 유지.** 크기는 작지만 방향이 명확하고 FiCR 개선 신호가 있음.
- 다음 실험은 임의로 확장하지 말고 사용자 확인 후 진행. 유망 후보는 group3 전용 teacher target/v_mode/backend 튜닝, group3 teacher + final blend weight 영향, 또는 group3 wind 복원 진단표.

출력:

- `results/scada_wind_translator_rf_oob_v1_*`
- `results/standard_oof_tree_lgbm_rf_scada_wind_teacherpc_group3_only_v1.csv`
- `results/repro_tree_rf_scada_wind_teacherpc_group3_only_v1_oof_pinn_tree_tcnfamily_*`

---

---

작성일: 2026-07-10 15:41:24 +09:00

실험명: L2/MSE 손실함수 재구성 ablation (기각)

목적: 사용자가 제안한 "학습 loss를 더 민감한 L2/MSE로 바꾸기"를 OOF로 검증. 평가는 기존 공식 metric OOF 그대로 유지하고, 학습 손실만 바꿈.

결과:

| branch | variant | mean score | mean nMAE | mean FICR | worst fold | 판단 |
|---|---:|---:|---:|---:|---:|---|
| TREE | 기존 L1 재현 | `0.62395` | `0.12810` | `0.37600` | `0.60605` | 기준 |
| TREE | L1 best params에서 objective만 L2 | `0.61553` | `0.13042` | `0.36148` | `0.59863` | 하락 |
| TCN W24 | weighted L2 | `0.61421` | `0.13565` | `0.36406` | `0.59600` | 기존 W24 `0.61580`보다 하락 |
| TCN W72 | weighted L2 | `0.61359` | `0.13495` | `0.36213` | `0.60147` | 기존 W72 `0.61409`보다 소폭 하락 |
| TCN W168 | weighted L2 | `0.61235` | `0.13507` | `0.35978` | `0.59640` | 기존 W168 `0.61620`보다 하락 |
| PINN | data loss MSE | `0.57995` | `0.14791` | `0.30781` | `0.56222` | 기존 PINN `0.61288` 대비 큰 하락 |

추가 확인:

- L2 TCN family를 기존 PINN/TREE와 섞어도 fixed weight `pinn0.25_tree0.40_tcn0.35`는 `0.62892`로 기존 재현 fixed `0.63019`보다 하락.
- PINN MSE를 기존 TREE/TCN과 섞으면 grid가 PINN weight `0`을 선택. ensemble diversity 관점에서도 도움 없음.
- LGBM focused L2 20-trial 튜닝을 짧게 확인했으나 branch 전환 신호는 약함. best rows: g1 `0.62135`, g2 `0.63758`, g3 `0.57478`.

판단: **손실함수 재구성은 기각.** 현재 score 구조에서는 L2/MSE가 nMAE를 의미 있게 줄이기보다 FICR/피크 포착을 잃는 방향으로 작동. 기존 metric-soft PINN, L1 TREE, weighted L1 TCN 유지.

출력:

- `results/repro_l2_power_lgbm_best_v2_l2_aggressive_minimal_rolling_v1_*`
- `results/oof_repro_l2_seqnn_short_tcn_w24_v1.csv`
- `results/oof_repro_l2_seqnn_mid_tcn_w72_v1.csv`
- `results/oof_repro_l2_seqnn_long_tcn_w168_v1.csv`
- `results/repro_mse_pinn_lgbm_time_oof_stage2_es_hod_v1_oof_*`
- `results/power_lgbm_hyperparams_v2_l2_20_*`

주의: test submission 생성 없음. 기존 best submission/weight 변경 없음.

다음 액션: loss 쪽은 닫고, 점수 상승 후보는 group3 teacher/backend mix, wind reconstruction/FiCR 피크 보정, residual correction 안정화 쪽으로 이동.

---

Next recommended sequence:

1. Submit or externally check `results/submission.csv`.
2. If public score confirms direction, commit the LGBM-teacher/tuned-tree pipeline.
3. Optional: group3-specific teacher backend mix, because RF teacher was better for group3 than LGBM.
4. Global stacked tree can be blended for group3, but current gain is too small for a submission.
5. XGB/ExtraTrees diversity is currently not useful; best validation weight is still LGBM-only.
6. Investigate inverse residual TREE correction stability before making a test submission.
7. Keep `PINN50:TREE50` as stable default unless user explicitly chooses validation-best tree=0.60.

Submission rule:

- Do not create new test submission unless validation improves by roughly `+0.01` or more, or the user explicitly asks for a diagnostic submission.

## Log Entries

작성일: 2026-07-09 01:19:50 +09:00

실험명: aggressive minimal TREE feature profile (`aggressive_minimal_v1`)

목적: 기존 TREE 511개 feature를 대회 코드 스타일에 맞춰 70개로 축소. 기존 tuned LGBM hyperparameter는 고정하고 feature profile만 변경.

결과:

| Model | Feature count | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---:|---:|---:|---:|---:|
| full_v2 TREE | 511 | `0.62361` | `0.12851` | `0.37573` | - |
| aggressive_minimal_v1 TREE | 70 | `0.62085` | `0.12980` | `0.37150` | `0.60369` |

판단: 하락폭 약 `-0.00276`으로 허용 기준 `-0.005` 이내. 제출 후보는 아니지만 관리 가능한 feature 기준점으로 유지. 다음 feature 실험은 70개 profile 위에서 direction-conditioned wind 또는 SCADA availability weighting처럼 큰 구조 후보를 붙이는 방향이 좋음.

출력: `results/power_lgbm_best_v2_l1_aggressive_minimal_v1_*`, `results/standard_oof_tree_lgbm_best_v2_l1_aggressive_minimal_v1.csv`

---

작성일: 2026-07-08 10:26:03 +09:00

실험명: test 2025 기상분포 유사도 진단 (`experiments/audit_test_weather_similarity.py`)

목적: test 2025 기상 입력분포가 train 어느 연도와 닮았는지 측정. 모델 입력 공간(wind+all_meteo 83 피처)에서 월-매칭 정규화 Wasserstein distance. 학습/제출 없음.

결과:

| train_year | dist (core 8) | dist (all 83) | rank |
|---|---:|---:|---:|
| 2022 | 0.2792 | 0.2178 | 1 |
| 2023 | 0.2923 | 0.2318 | 2 |
| 2024 | 0.3584 | 0.2783 | 3 |

- 2025는 2022와 가장 유사, 2024와 가장 다름. core/all 순위 일치.
- 2025-train 거리(0.218~0.278)는 train 연도끼리 거리(0.245~0.273)와 같은 범위 → 2025 기상 입력분포 자체는 이상 drift 아님.
- 2025년 2월만 예외적: 2022와만 유사(0.139), 2024와는 크게 다름(0.654).
- sanity check: 2024 pseudo-test의 최근접은 2023 — pred_2024 fold(2022+2023 학습)가 최고점이었던 것과 방향 일치.
- 교차분석: 2025 최근접 연도 2022를 예측한 fold 점수는 TREE 0.6358(최고 fold), PINN 0.6146 → public(0.6304) > OOF(0.6268) 역전과 정합.

판단: 연도 간 거리 차이가 작아 fold 가중(softmax 제안 0.79/0.13/0.08)을 적용할 근거 부족. year-bagging **균등가중 유지**. 실질 가치는 (1) 2025 입력분포 정상 확인, (2) 기울인다면 (2022,2023) fold 방향이라는 순서 정보, (3) PINN public 부진의 남은 의심 지점은 입력분포가 아니라 forecast→발전량 관계(라벨 쪽)라는 좁힘.

다음 액션: fold 가중 보류. 다음 데이터 실험은 라벨 쪽 — group3 라벨 용량 초과 38건 클리닝 또는 teacher 분포 보정.

출력: `results/test_weather_similarity_feature_distances.csv`, `_year_summary.csv`, `_month_summary.csv`

---

작성일: 2026-07-08 10:40:00 +09:00

실험명: group3 라벨 클리닝 후보 진단 (일회성 스크립트, 미보존)

목적: group3 라벨 용량 초과 38행이 오류인지, commissioning 초기 라벨 왜곡/시변 유효용량 문제가 있는지 확인. 클리닝으로 group3 개선 여지 판단.

결과:

| 확인 항목 | 결과 |
|---|---|
| 용량 초과 38행 | **오류 아님** — 초과분 중앙값 58kWh(용량의 0.3%), 38행 중 34행은 SCADA 시간합이 라벨과 일치(ratio≈1.00). 실제 정격 초과 발전 |
| 라벨 vs UNISON SCADA 시간합 | corr `0.9966`, 월별 ratio 중앙값 전 기간 `1.00` — **group3 라벨은 사실상 SCADA 합계** |
| commissioning 초기(2023 초) 왜곡 | 없음. 2023-01부터 ratio 1.00, active turbine 5.0 |
| 시변 유효용량 | 비문제. active turbine 평균 4.9~5.0으로 안정 |
| 예외 | 2024-02-12~13 4행: 터빈 1기 SCADA 미보고 상태에서 라벨은 만출력(ratio 1.25) — SCADA 부분 결측 시 SCADA 파생 피처가 과소평가될 수 있다는 소규모 신호 |

판단: **라벨 클리닝/시변 용량 방향은 기각.** group3 라벨 품질은 우수하고 용량 초과도 실제 발전. 예측이 21,000으로 clip되어 생기는 구조적 오차는 38행×최대 131kWh로 무시 가능. group3 부진의 원인은 라벨이 아니라 forecast→풍속 복원 단계(teacher)로 다시 좁혀짐.

다음 액션: group3 개선은 기존에 확인된 미적용 개선인 **group3 전용 RF teacher backend mix**(Next Work 3번)로 진행. 부수 발견인 "SCADA 부분 결측 시간대 downweight"는 teacher target 생성 시 참고.

---

작성일: 2026-07-08 10:55:00 +09:00

실험명: 데이터 이상 정밀 스캔 (일회성 스크립트, 미보존)

목적: audit 미커버 항목 점검 — 라벨 결측/zero-run 패턴, VESTAS power spike 해부, 라벨-SCADA 정합, grid/lead train-test 일치, UNISON 부분보고.

결과 (정상 확인):

| 항목 | 결과 |
|---|---|
| LDAPS/GFS grid 좌표 | train=test 완전 일치 (LDAPS 16, GFS 9, nearest grid 5 동일) |
| lead_hour 분포 | train/test 동일 (12~35h, 사분위 일치) |
| test 시간축 | 결측 0h, NaN 행 48개(0.03%)뿐 |
| VESTAS 음수 power 31.7만건 | 전부 [-50,0) 대기전력 — 정상 |
| ws=0 & power>50 모순 | 0건 — frozen 센서 없음 |
| 라벨 vs SCADA합 (clean) | g1 corr 0.9998 / g2 0.9998, 연도별 ratio 0.984~0.990 완전 안정 — 3그룹 모두 라벨≈SCADA합, 라벨 체계 변화 없음 |

결과 (이상 발견):

| 항목 | 내용 |
|---|---|
| VESTAS power spike | ±1e6 이상 값 868건(759행, 0.5%), +/- 정확히 쌍, 전 기간 분포 — 센서 누적값 리셋. 파워커브는 `clean=True`로 방어 중 |
| 3그룹 동시 zero-run | 2024-02-22(142~169h), 2024-01-18(85~120h), 2023-02-13(77~85h) 등 — 바람 무관 그리드/변전 정지. metric 제외 구간이지만 **파워커브 fit에는 (고풍속, power=0) 샘플로 유입** |
| UNISON 부분보고 | 271h, 그중 2024-02에 180h 집중(위 정지 이벤트와 겹침) |
| 라벨 결측 | g1/g2 각 104행 중 82h가 2022-10-24~27 집중 |

판단: 원천 데이터 골격(grid/lead/시간축/라벨체계)은 train-test 정합. 남은 실질 오염 경로는 (1) 파워커브 fit의 curtailment 샘플, (2) UNISON 부분보고 시간대의 group 통계 왜곡 두 개.

다음 액션: 이상치 처리 후보 — 동시 zero-run 시간 마스크를 파워커브/power계 teacher fit에서 제외, UNISON 보고 기수 정규화. 피처 후보 — 과거 풍속 EMA(2h/6h) 및 t+1/t+2 풍속(diff만 있고 level 스무딩 없음). t+1/t+2 미래 풍속 피처는 사용자 판단으로 보류.

---

작성일: 2026-07-08 11:30:00 +09:00

실험명: TREE 데이터 수정 2건 OOF 검증 — curtailment 마스크 + 과거 풍속 EMA (실패, 코드 미보존)

목적: (1) 파워커브 fit에서 3그룹 동시 zero-run(>=12h, 556시간 탐지) SCADA 샘플 제외, (2) PINN tau=2h를 이식한 과거 풍속 EMA(2h/6h)+rolling max 3h 피처 추가. tuned LGBM v2_l1 동일 조건 year-fold OOF 비교.

결과:

| variant | mean score | worst fold | vs baseline |
|---|---:|---:|---:|
| wind_ema | 0.62382 | 0.60491 | +0.0002 |
| baseline | 0.62361 | 0.60663 | - |
| mask_ema | 0.62357 | 0.60524 | -0.0000 |
| curtail_mask | 0.62254 | 0.60411 | -0.0011 |

- wind_ema는 group1만 +0.0009 (2024 fold +0.0026), group3는 악화. 평균 +0.0002는 fold std(0.015~0.017) 대비 잡음.
- curtail_mask는 오히려 소폭 악화. 라벨 자체가 정지 시간을 포함하므로, 정지 샘플이 섞인(살짝 눌린) 파워커브가 기대 라벨과 더 정합하는 것으로 해석.
- 두 variant 모두 worst fold 개선 없음.

판단: **둘 다 기각.** 승격 기준(mean +0.005 이상 & worst fold 비악화) 미달. curtailment 마스크는 tree 파워커브 경로에서는 무효 — 단, power-weighted SCADA teacher류 실험을 재개할 때는 재검토 가치 있음(라벨이 아니라 SCADA power를 직접 target으로 쓰는 경로는 오염 구조가 다름). 실험 코드/헬퍼(utils/curtailment.py, add_wind_history_features)는 workflow 원칙대로 제거. 동시 zero-run 탐지 로직은 "라벨 전 그룹 0 & 연속 >=12h"로 단순해 재구현 쉬움.

다음 액션: TREE 쪽 데이터 수정은 소진. 남은 우선 후보는 group3 전용 RF teacher backend mix(PINN 쪽, Next Work 3번)와 inverse residual TREE 안정성 검증(Next Work 6번).

---

작성일: 2026-07-10 19:17:40 +09:00

실험명: TREE importance 기반 lean 31입력 OOF 검증 (기각)

목적: `aggressive_minimal_rollmean_v1` 98입력에서 중요도 상위 축만 남겨 TREE 입력을 20~30개대로 줄일 수 있는지 확인. 이전 `wind_cubic_minimal_v1`은 hand-pick 성격이 강하고 중요 `lead/roll/context` 축을 과하게 제거했으므로, 실제 중요도표를 기준으로 `importance_lean_v1`을 새로 구성했다.

입력 구조:

- weather 30개 + `power_curve_est` 1개 = 모델 입력 31개.
- `gfs_ws100_speed`는 gain 자체는 낮지만 `power_curve_est` 계산용 `HUB_HEIGHT_PROXY_COL`이라 유지.
- 주요 축: `phys_ldaps_ws50max_grid_max`의 current/lead1/lead3/roll3_mean, LDAPS range/gradient, GFS 850/500 directional context, calendar, 일부 raw meteo, 신규 grid cubic 2개.

결과:

| variant | mean score | mean nMAE | mean FiCR | worst fold | 비교 |
|---|---:|---:|---:|---:|---:|
| rollmean_v1 98입력, 기존 best params | 0.623916 | 0.128277 | 0.376109 | 0.606363 | 기준 |
| importance_lean_v1 31입력, 기존 best params | 0.623217 | 0.129276 | 0.375709 | 0.604535 | -0.000699 |
| importance_lean_v1 31입력, focused_l1 12-trial retune | 0.622310 | 0.129187 | 0.373806 | 0.602617 | -0.001606 |

그룹별 특징:

- 기존 best params에서는 group2 2024가 `0.66939`로 강했지만, group1 2023과 group3가 약해 전체 개선으로 이어지지 않았다.
- retune은 group2 trial score는 좋아 보였으나 최종 OOF에서는 FiCR가 더 낮아졌고 worst fold도 악화.
- nMAE보다 FiCR 하락이 더 문제다. TREE는 풍속 크기뿐 아니라 lead/roll/공간 방향성 컨텍스트를 넓게 들고 있을 때 coverage 판정이 안정적인 것으로 보인다.

판단: **기각.** 31입력 lean은 노이즈 제거 방향 자체는 타당하지만, 현재 TREE replacement로 쓰기에는 정보 손실이 더 크다. 현 TREE 후보는 `aggressive_minimal_rollmean_v1` 98입력을 유지한다. 다음 feature pruning은 전체 30개 단일 profile보다 group별 top-k 또는 family별 quota 방식으로 가는 편이 낫다.

출력:

- `results/power_lgbm_best_v2_l1_importance_lean_v1_scores.csv`
- `results/power_lgbm_best_v2_l1_importance_lean_v1_summary.csv`
- `results/power_lgbm_hyperparams_importance_lean_v1_l1_12_best.csv`
- `results/power_lgbm_best_importance_lean_v1_l1_12_scores.csv`
- `results/power_lgbm_best_importance_lean_v1_l1_12_summary.csv`
---

작성일: 2026-07-10 KST

실험명: PINN+TCN -> TREE weather stacker submission candidate

목적: OOF에서 기각하지 않고 후보로 남긴 true stacker(`PINN+TCN baseline` 위에 `TREE weather direct-delta stacker`)를 test submission에 실제 적용한다. 이전 `submission_pinn45_tcn55_g13_delta0375.csv`는 stacker가 아니라 fixed-delta 후처리였고 public 기각됨. 이번 파일은 full-train TREE stacker를 test weather feature에 적용한 것이다.

설정:

- Script: `experiments/create_pinn_tcn_tree_stack_submission.py`
- Train stacker meta input: PINN OOF + TCN family OOF
- Test stacker meta input: PINN submission + TCN family submission
- Weather feature profile: `group_family_quota65_v1`
- Baseline: `0.45 * PINN + 0.55 * TCN_family`
- Stacker:
  - target mode: `direct_delta`
  - model: weak LGBM `l2`
  - weight policy: `actual_sqrt`
  - min output ratio: `0.10`
  - correction clip: `±0.025 * capacity`
  - applied groups: `kpx_group_1`, `kpx_group_3`
  - `kpx_group_2`: baseline 그대로

OOF 근거:

| Variant | Mean score | Mean nMAE | Mean FiCR | Worst fold |
|---|---:|---:|---:|---:|
| `PINN0.45/TCN0.55` baseline | `0.627009` | `0.133310` | `0.387328` | `0.615951` |
| weather direct-delta TREE stacker, group1+3, clip 2.5% | `0.629552` | `0.131489` | `0.390593` | `0.616597` |

출력:

- `results/submission_pinn45_tcn55_tree_stack_g13_clip025.csv`
- `results/submission_pinn45_tcn55_tree_stack_g13_clip025_diagnostics.csv`

검증:

- shape: `8760 x 5`
- nulls: `0`
- capacity bounds: OK
- correction diagnostics:
  - group1 mean `+0.008774`, p05 `-0.025`, p50 `+0.025`, p95 `+0.025`
  - group2 all `0`
  - group3 mean `+0.006410`, p05 `-0.025`, p50 `+0.025`, p95 `+0.025`

주의: correction이 clip 경계에 많이 붙는다. public 안정성은 높지 않을 수 있으므로 결과 확인 후 즉시 판정.

Public 결과:

| File | Public score | 1-nMAE | nMAE | FiCR | 판단 |
|---|---:|---:|---:|---:|---|
| `submission_pinn45_tcn55_tree_stack_g13_clip025.csv` | `0.6335692297` | `0.8667639269` | `0.1332360731` | `0.4003745325` | 기각 |

해석: fixed-delta 후처리(`0.627514`)보다는 크게 회복했지만, 현재 best public `0.6370788926` 및 기존 단순 3-branch 계열보다 낮다. OOF에서는 `PINN+TCN` baseline 대비 stacker가 개선됐지만, public에서는 `PINN+TREE+TCN` 단순 앙상블의 TREE 직접 예측 체급을 넘지 못했다. 따라서 v1 true stacker는 제출 후보에서 제외한다.

---

작성일: 2026-07-10 KST

실험명: PINN+TCN group1/3 calibrated delta submission candidate

목적: `PINN+TCN baseline -> TREE stacker` 실험 중 발견된 group1/group3 underprediction bias를 이용해, test submission 후보를 1개 생성한다. 사용자가 명시적으로 submission 후보 생성을 요청했으므로 test CSV를 만들었다.

설정:

- Script: `experiments/create_pinn_tcn_delta_submission.py`
- PINN input: `results/submission_pinn_effective_grid_g1_year_bagging.csv`
- TCN family input:
  - `0.30 * submission_seqnn_short_tcn_w24_v1.csv`
  - `0.40 * submission_seqnn_mid_tcn_w72_v1.csv`
  - `0.30 * submission_seqnn_long_tcn_w168_v1.csv`
- Blend: `0.45 * PINN + 0.55 * TCN_family`
- Calibration:
  - `kpx_group_1 += 0.0375 * capacity`
  - `kpx_group_2 += 0`
  - `kpx_group_3 += 0.0375 * capacity`

OOF 근거:

| Variant | Mean score | Mean nMAE | Mean FiCR | Worst fold |
|---|---:|---:|---:|---:|
| `PINN0.45/TCN0.55` baseline | `0.627009` | `0.133310` | `0.387328` | `0.615951` |
| group1+3 fixed delta `0.0375`, all-OOF selected | `0.632393` | `0.135740` | `0.400525` | `0.623607` |
| group1+3 fixed delta, cross-year selected | `0.631336` | `0.136354` | `0.399026` | `0.623389` |

출력:

- `results/submission_pinn45_tcn55_g13_delta0375.csv`

검증:

- shape: `8760 x 5`
- nulls: `0`
- capacity bounds: OK
- `results/submission.csv`는 변경하지 않음.

Public 결과:

| File | Public score | 1-nMAE | nMAE | FiCR | 판단 |
|---|---:|---:|---:|---:|---|
| `submission_pinn45_tcn55_g13_delta0375.csv` | `0.6275136665` | `0.8582049759` | `0.1417950241` | `0.3968223571` | 기각 |

해석: OOF에서 보였던 group1/group3 positive delta 이득이 public에서는 재현되지 않았다. 특히 1-nMAE가 기존 best 대비 크게 낮아져, uniform positive delta가 public 분포에서는 과보정으로 작동한 것으로 판단한다. 이 계열은 제출 후보에서 제외하고, 이후에는 fixed delta만으로 만든 후보를 쓰지 않는다.

---

작성일: 2026-07-10 KST

실험명: PINN+TCN baseline -> TREE residual stacker v1

목적: 사용자가 제안한 true stacking 구조를 검증한다. `PINN`은 물리식 기반 강건성, `TCN family`는 시계열 shape/ramp 포착을 담당하는 1층 baseline으로 두고, 2층 `TREE`는 baseline이 못 잡는 비선형 residual/correction만 학습한다.

설정:

- Branch: `codex/stacking-v1`
- Scripts:
  - `experiments/evaluate_pinn_tcn_tree_stacking_v1.py`
  - `experiments/evaluate_pinn_tcn_tree_residual_weather_v1.py`
- Baseline: `base = w * PINN + (1-w) * TCN_family`
- TREE stacker:
  - prediction/context-only residual stacker
  - weather-feature residual/direct-delta stacker using `group_family_quota65_v1`
- Leakage guard: OOF `pred_year` holdout. No test submission created.

결과:

| Variant | Mean score | Mean nMAE | Mean FiCR | Worst fold | 판단 |
|---|---:|---:|---:|---:|---|
| `PINN0.40/TCN0.60` baseline | `0.627202` | `0.133162` | `0.387566` | `0.615487` | 기준 |
| prediction/context-only TREE residual best | `0.625677` | `0.132776` | `0.384130` | `0.614245` | 기각 |
| weather residual, all groups, weak clip 1% | `0.627618` | `0.131570` | `0.386806` | `0.615059` | 평균 소폭 상승, worst 하락 |
| weather direct-delta, group1+3 only, `PINN0.45/TCN0.55`, clip 2.5% | `0.629552` | `0.131489` | `0.390593` | `0.616597` | true stacking 후보 |
| same medium TREE capacity | `0.628950` | `0.131717` | `0.389618` | `0.615829` | 약한 모델보다 나쁨 |

해석:

- 예측값/time/lead context만 보는 TREE stacker는 baseline보다 하락. TREE가 비선형성을 살리려면 weather feature가 필요하다.
- weather feature를 넣고 target을 `direct_delta`로 바꾼 뒤, group1/group3에만 correction을 적용하면 baseline 대비 `+0.00254`, FiCR `+0.00327`, worst fold `+0.00065`.
- group2는 TREE correction을 적용하면 대체로 손해라 v1에서는 gate-out이 맞다.
- medium capacity는 weak보다 낮아 현재는 강한 TREE보다 보수적인 correction이 낫다.

중요 sanity check:

`PINN0.45/TCN0.55` baseline에 group1/group3 fixed positive delta를 cross-year로 고르면:

| Calibration | Mean score | Mean nMAE | Mean FiCR | Worst fold |
|---|---:|---:|---:|---:|
| group1+3 fixed delta, cross-year selected | `0.631336` | `0.136354` | `0.399026` | `0.623389` |

즉 v1 TREE stacker가 잡은 이득의 상당 부분은 비선형 residual이라기보다 group1/group3 underprediction bias다. 다음 단계는 `PINN+TCN + cross-year group delta`를 calibrated baseline으로 삼고, 그 위에 TREE residual을 다시 얹는 구조가 맞다.

출력:

- `results/pinn_tcn_tree_stacking_v1_scores.csv`
- `results/pinn_tcn_tree_stacking_v1_summary.csv`
- `results/pinn_tcn_tree_residual_weather_v1_g13_fine_scores.csv`
- `results/pinn_tcn_tree_residual_weather_v1_g13_fine_summary.csv`
- `results/pinn_tcn_tree_residual_weather_v1_g13_fine_predictions.csv`
- `results/pinn_tcn_tree_residual_weather_v1_g13_fine_diagnostics.csv`

---

작성일: 2026-07-10 KST

실험명: three-branch leakage-safe stacking v1 OOF

목적: 기존 `PINN=0.25, TREE=0.40, TCN=0.35` 고정 weight 앙상블을, 예측 연도 holdout을 지킨 OOF stacking으로 바꿨을 때 큰 개선 여지가 있는지 확인.

설정:

- Branch: `codex/stacking-v1`
- Script: `experiments/evaluate_three_branch_stacking_v1.py`
- Components:
  - PINN: `results/pinn_effective_grid_g1_year_bagging_lgbm_time_oof_oof_predictions.csv`
  - TREE: `results/power_lgbm_best_v2_l1_group_family_quota65_v1_predictions.csv`
  - TCN family: `results/oof_tcn_family_w24_0.30_w72_0.40_w168_0.30.csv`
- Leakage guard: each `pred_year` is scored using a meta model/weight table trained only on the other years' OOF rows.
- No test submission created.

결과:

| Variant | Mean score | Mean nMAE | Mean FiCR | Worst fold | 판단 |
|---|---:|---:|---:|---:|---|
| fixed `PINN25/TREE40/TCN35` | `0.630215` | `0.127306` | `0.387737` | `0.615819` | 기준 |
| `cellw_group_lead_bin_pred_bin_min300_s800` | `0.632266` | `0.127445` | `0.391977` | `0.618775` | 유망 |
| `cellw_group_disagreement_bin_pred_bin_min300_s800` | `0.631042` | `0.127671` | `0.389755` | `0.616885` | 후보 |
| best direct meta regressor (`huber_global_all`) | `0.620876` | `0.136585` | `0.378337` | `0.613299` | 기각 |

해석/정정:

- 이 실험은 true stacking이 아니라 **regime별 convex weight calibration**이다. 사용자가 지적한 대로 true level-2 stacking으로 부르지 않는다.
- 직접 regression stacker(Ridge/Huber/HistGB)는 FiCR 또는 nMAE를 크게 망가뜨려 현재 경로로는 부적합.
- 반면 convex weight를 `group + lead_bin + pred_bin` regime별로 고르고 shrinkage로 global weight에 당기는 방식은 평균 `+0.00205`, FiCR `+0.00424`, worst fold `+0.00296` 개선.
- 개선은 nMAE보다는 FiCR 쪽이다. 이 대회 metric 구조상 6%/8% error band 안으로 더 많이 밀어넣는 효과로 보인다.
- 단, cell weight 방식은 과적합 위험이 있으므로 바로 제출 후보로 보지 않고 `min_rows`, `shrinkage`, `cell definition`, component set 안정성 검증이 필요하다.

출력:

- `results/three_branch_stacking_v1_scores.csv`
- `results/three_branch_stacking_v1_summary.csv`
- `results/three_branch_stacking_v1_predictions.csv`
- `results/three_branch_stacking_v1_cell_weights.csv`
---

작성: 2026-07-10 KST

실험명: TCN FiCR-only loss OOF v1

목적: 기존 TCN `weighted_l1` 대신 soft FiCR 항만 최적화했을 때 FiCR/총점이 올라가는지 OOF로 확인.

설정:

- Branch: `main`
- Loss: `ficr_only`
- Model: TCN W24/W72/W168
- Weight policy: 기존 `actual_sqrt` 입력은 유지하지만, `ficr_only` loss에서는 직접 사용하지 않음.
- TCN family: W24 0.30 / W72 0.40 / W168 0.30
- No test submission.

결과:

| Variant | OOF score | nMAE | FiCR | Worst fold | 판단 |
|---|---:|---:|---:|---:|---|
| `seqnn_tcn_w24_ficr_only_v1` | `0.627956` | `0.139290` | `0.395203` | `0.618440` | 기존 W24 대비 상승 |
| `seqnn_tcn_w72_ficr_only_v1` | `0.628969` | `0.139635` | `0.397573` | `0.620544` | 단일 window 최고 |
| `seqnn_tcn_w168_ficr_only_v1` | `0.628887` | `0.138909` | `0.396684` | `0.616338` | 상승 |
| `tcn_family_ficr_only_w24_030_w72_040_w168_030` | `0.633524` | `0.136600` | `0.403648` | `0.624694` | 유의미한 개선 |

3-branch OOF sanity:

| Variant | OOF score | nMAE | FiCR | Worst fold |
|---|---:|---:|---:|---:|
| 기존 TCN family, fixed `PINN25/TREE40/TCN35` | `0.630885` | `0.127596` | `0.389365` | `0.616669` |
| FiCR-only TCN family, fixed `PINN25/TREE40/TCN35` | `0.632463` | `0.128552` | `0.393477` | `0.619558` |
| FiCR-only TCN family, grid best `PINN10/TREE15/TCN75` | `0.635223` | `0.132392` | `0.402837` | `0.624631` |

해석:

- TCN에서는 FiCR-only loss가 OOF 기준 꽤 강하게 먹힌다. nMAE 손실이 있지만 FiCR 상승폭이 더 커서 branch/family 점수가 오른다.
- 3-branch grid가 TCN 쪽으로 크게 기울어졌으므로 public overfit 가능성은 있다. 바로 best 교체로 단정하지 않고, submission은 사용자 확인 후 진행한다.
- PINN FiCR-only는 같은 실행에 포함했으나, OOF script가 fold마다 SCADA teacher를 재학습하는 구조라 비효율적이었다. 중복 프로세스 하나를 정리했고, 정상 프로세스도 중단했다. PINN loss ablation은 teacher prediction cache 경로를 만든 뒤 재실행하는 것이 맞다.

출력:

- `results/oof_seqnn_tcn_w24_ficr_only_v1.csv`
- `results/oof_seqnn_tcn_w72_ficr_only_v1.csv`
- `results/oof_seqnn_tcn_w168_ficr_only_v1.csv`
- `results/oof_tcn_family_ficr_only_w24_030_w72_040_w168_030.csv`
- `results/three_branch_oof_blend_tcn_ficr_only_v1_summary.csv`

---

작성: 2026-07-11 KST

실험명: Metric floor postprocess v1

목적: 공식 평가는 실제 출력이 설비용량 10% 이상인 행만 보므로, 제출 예측도 group capacity의 10% 아래로 내려가지 않게 후처리했을 때 OOF와 제출 후보가 좋아지는지 확인.

설정:

- Postprocess: `pred = clip(pred, 0.10 * capacity, capacity)`
- 기존 제출/OOF 파일은 덮어쓰지 않고 `_floor10` suffix로 별도 저장.
- Script: `experiments/apply_metric_floor_submission.py`, `experiments/evaluate_metric_floor_oof.py`

OOF 결과:

| Input OOF | Raw score | Floor10 score | 변화 | 메모 |
|---|---:|---:|---:|---|
| `results/oof_tcn_family_ficr_only_w24_030_w72_040_w168_030.csv` | `0.633524` | `0.633531` | `+0.000007` | 이미 대부분 floor 위라 변화 작음 |
| `results/repro_oof_pinn_tree_tcnfamily_w24_030_w72_040_w168_030_best_oof.csv` | `0.630514` | `0.630745` | `+0.000231` | nMAE `0.126514 -> 0.126334`, FiCR `0.387541 -> 0.387823` |

제출 후보:

- `results/submission_current_best_floor10.csv`
- `results/submission_pinn10_tree15_tcn75_tcn_ficr_only_family_v1_floor10.csv`
- `results/submission_pinn25_tree40pseudo_tcn35_ficr_only_family_v1_floor10.csv`
- `results/submission_pinn25_tree40_tcn35_tree_g3_vestas_pseudo2022_w010_rebuilt_floor10.csv`

판단:

- 공식 metric의 valid 영역 밖으로 예측이 내려가는 것을 막는 보정이라 구조적으로 손해가 거의 없는 편.
- 단, TCN FiCR-only 제출은 이미 floor 아래 예측이 적어서 개선 폭이 작을 가능성이 큼.
- `submission_current_best_floor10.csv` public은 `0.6364805561`로, 일반 `submission.csv` 계열은 개선했지만 pseudo2022 best `0.6370788926`는 넘지 못함. 따라서 진짜 비교 대상은 pseudo2022 best 조합을 재합성한 `_rebuilt_floor10` 파일임.
- `_rebuilt_floor10` public은 `0.6354556595`로 기각. 원래 0.637 best 파일이 로컬/duck에 남아 있지 않아 `rebuilt`가 exact 재현이 아니었고, 특히 pseudo2022 TREE component가 원 제출본과 다르다고 판단한다. floor10 자체는 `submission.csv` 계열에서 `0.6357518532 -> 0.6364805561`로 개선을 확인했지만, 원래 best CSV를 찾기 전까지 best+floor10 검증은 보류.
---

작성: 2026-07-11 KST

실험명: one-sided max(TREE,TCN) lift validation v1

목적: quantile/power mean 계열이 "큰 예측을 더 믿는" 방향에서 개선되었으므로, 예측을 낮추지는 않고 TREE/TCN 상단 신호가 있을 때만 끌어올리는 더 단순한 postprocess를 검증한다. 제출 파일은 만들지 않았다.

공식:

```text
mean = w_pinn * PINN_floor + w_tree * TREE + w_tcn * TCN
upper = max(TREE, TCN)
final = mean + alpha * max(upper - mean, 0)
final = clip(final, 0.10 * capacity, capacity)
```

결과:

| Variant | OOF score | nMAE | FiCR | Worst fold | 판단 |
|---|---:|---:|---:|---:|---|
| quantile best `floor27.5, 25/25/50, q60, alpha60` | `0.636351` | `0.129560` | `0.402263` | `0.627227` | 기존 강한 후보 |
| coarse best `floor25, 25/20/55, lift max(TREE,TCN), alpha60` | `0.637951` | `0.130567` | `0.406469` | `0.628948` | 강한 개선 |
| focused best `floor23.75, 30/22.5/47.5, lift max(TREE,TCN), alpha64` | `0.638038` | `0.130947` | `0.407022` | `0.629650` | 현재 OOF 최고 후보 |

판단:

- PINN을 upper 후보에서 제외하고 TREE/TCN 중 높은 쪽만 lift source로 쓰는 게 잘 먹힌다.
- 개선은 FiCR/worst fold 중심이다. nMAE 손해가 있으므로 public 제출 후보로는 공격형에 가깝다.
- 주변 조합이 `0.6379~0.6380`에 밀집되어 있어 OOF 안정성은 quantile 후보보다 좋아 보인다.
- 저장:
  - `results/one_sided_lift_validation_v1_summary.csv`
  - `results/one_sided_lift_validation_v1_scores.csv`
  - `results/one_sided_lift_focused_grid_v1_summary.csv`

---

작성: 2026-07-11 KST

실험명: branch quantile blend validation v1

목적: power mean이 "큰 예측을 더 믿는" 방향으로 OOF를 올렸기 때문에, 같은 직관을 branch 분위수 기반으로 더 직접 적용해본다. 제출 파일은 만들지 않고 OOF만 평가했다.

설정:

- Components: PINN effective-grid OOF, TREE `power_lgbm_best_v2_l1_predictions`, TCN weighted-l1 family OOF.
- Baseline: `PINN floor35 + PINN25 / TREE20 / TCN55 + final floor10`.
- Quantile: branch prediction ratio를 capacity 기준으로 정규화한 뒤 weighted step quantile 사용.
- Best tested form: `(1 - alpha) * weighted_mean + alpha * weighted_quantile_step(q=0.60)`.

결과:

| Variant | OOF score | nMAE | FiCR | Worst fold | 판단 |
|---|---:|---:|---:|---:|---|
| baseline `floor35, 25/20/55` | `0.633639` | `0.128357` | `0.395634` | `0.622407` | 기준 |
| fixed weight `floor35, 25/20/55, q60, alpha60` | `0.636108` | `0.130103` | `0.402320` | `0.626722` | 강한 개선 |
| targeted best `floor27.5, 25/25/50, q60, alpha60` | `0.636351` | `0.129560` | `0.402263` | `0.627227` | 현재 OOF 최고 후보 |

판단:

- 순수 median/quantile 단독보다 기존 weighted mean에 상위 분위수를 섞는 방식이 안정적이다.
- 개선은 nMAE보다 FiCR/worst fold에서 크게 나온다. public에서는 과상향 리스크가 있으므로 바로 대량 제출 후보로 확정하지 말고 1개 보수 후보부터 보는 편이 낫다.
- 저장:
  - `results/quantile_branch_blend_validation_v1_summary.csv`
  - `results/quantile_branch_blend_validation_v1_scores.csv`
  - `results/quantile_branch_blend_targeted_grid_v1_summary.csv`
---

작성: 2026-07-11 KST

실험명: TREE target-scale family 1/3/5/7/13/25 v1

목적: TCN W24/W72/W168처럼 TREE도 target smoothing horizon을 달리한 branch를 만들면 초단기/단기/중기 감각의 앙상블 효과가 있는지 확인한다. 제출 파일은 만들지 않았다.

설정:

- Branch: TREE only.
- Feature profile: `aggressive_minimal_rollmean_v1`.
- Target: centered rolling mean target, horizon `1, 3, 5, 7, 13, 25`.
- Validation: leave-one-year OOF.
- 실행: duck `/home/yunjun0914/.venvs/WindForecast`.

결과:

| Variant | OOF score | nMAE | FiCR | Worst fold | 판단 |
|---|---:|---:|---:|---:|---|
| h1 | `0.623916` | `0.128277` | `0.376109` | `0.606363` | 기준 |
| h3 | `0.621904` | `0.128229` | `0.372036` | `0.603249` | 하락 |
| h5 | `0.619356` | `0.128101` | `0.366812` | `0.600707` | 하락 |
| h7 | `0.614977` | `0.128954` | `0.358908` | `0.597462` | 크게 하락 |
| h13 | `0.596373` | `0.134515` | `0.327261` | `0.575369` | 기각 |
| h25 | `0.553492` | `0.153794` | `0.260778` | `0.531863` | 기각 |
| best blend `0.90*h1 + 0.10*h5` | `0.624080` | `0.128063` | `0.376223` | `0.606698` | 소폭 개선 |

판단:

- TREE target smoothing은 단독 branch로는 horizon이 길수록 FiCR을 크게 잃는다.
- 5h를 10%만 섞으면 h1 대비 `+0.000164`로 소폭 개선하지만, 제출권을 쓸 정도의 강한 개선은 아니다.
- 13h/25h가 명확히 무너지므로 월 단위 target smoothing은 현재 채점식에서는 부적합하다고 판단한다.
- 저장:
  - `results/tree_multi_horizon_target_1_3_5_7_13_25_v1_blend_summary.csv`
  - `results/tree_multi_horizon_target_1_3_5_7_13_25_v1_horizon_scores.csv`
---

작성: 2026-07-11 KST

실험명: LGBM quantile TREE q55/q60/q65/q70/q75 OOF v1

목적: 기존 TREE가 평균/중앙 예측에 가까워 고출력 가능성을 누르는지 확인하기 위해, LightGBM `objective=quantile` 모델을 TREE branch로 학습한다. `q55~q75` upper TREE가 단독 TREE와 three-branch one-sided lift에서 개선되는지 OOF로만 평가했다. 제출 파일은 만들지 않았다.

설정:

- Branch: TREE only + three-branch diagnostic.
- Feature profile: `aggressive_minimal_rollmean_v1`.
- Baseline TREE params: `results/power_lgbm_hyperparams_v2_l1_20_best.csv`.
- Quantile objectives: `q55, q60, q65, q70, q75`.
- Three-branch diagnostic base: `PINN floor25 + PINN25 / TREE20 / TCN55`.
- 실행: duck `/home/yunjun0914/.venvs/WindForecast`.

TREE 단독 결과:

| Variant | OOF score | nMAE | FiCR | Worst fold | 판단 |
|---|---:|---:|---:|---:|---|
| mean ref | `0.623916` | `0.128277` | `0.376109` | `0.606363` | 기준 |
| q55 | `0.625315` | `0.130448` | `0.381077` | `0.609761` | 개선 |
| q60 | `0.626912` | `0.133660` | `0.387483` | `0.613418` | TREE 단독 최고 |
| q65 | `0.625766` | `0.137944` | `0.389476` | `0.615576` | FiCR는 높지만 nMAE 손해 |
| q70 | `0.622246` | `0.144312` | `0.388803` | `0.616519` | 과상향 |
| q75 | `0.616810` | `0.151985` | `0.385605` | `0.613620` | 기각 |

Three-branch/lift 결과:

| Variant | OOF score | nMAE | FiCR | Worst fold | 판단 |
|---|---:|---:|---:|---:|---|
| `base + 0.70 * max(mean TREE, TCN)` | `0.639074` | `0.129916` | `0.408064` | `0.627110` | 기준 upper가 가장 강함 |
| `TREE base=q55, weight 25/25/50, upper=max(mean TREE,TCN), alpha70` | `0.639128` | `0.130239` | `0.408496` | `0.627147` | +0.000054, 미미 |
| best direct quantile upper `max(q55,TCN), alpha70` | `0.638795` | `0.131006` | `0.408596` | `0.627457` | 기준보다 낮음 |

판단:

- quantile TREE는 TREE 단독 branch 개선에는 의미가 있다. 특히 `q60`이 `0.623916 -> 0.626912`로 뚜렷하게 상승한다.
- 하지만 최종 three-branch lift에서는 이미 `max(mean TREE, TCN)`이 upper signal을 충분히 제공한다. quantile TREE를 upper로 쓰는 것은 기준을 넘지 못했다.
- base TREE를 `q55/q60`으로 교체하는 조합은 최고 `+0.00005` 수준이라 제출 후보로 삼기엔 너무 작다.
- 앞으로는 quantile TREE를 단독 TREE family 후보로 보관하되, 최종 제출 후보의 핵심 레버는 여전히 one-sided `max(mean TREE, TCN)` lift 쪽으로 둔다.
- 저장:
  - `results/tree_quantile_lgbm_q55_q60_q65_q70_q75_v1_tree_summary.csv`
  - `results/tree_quantile_lgbm_q55_q60_q65_q70_q75_v1_branch_lift_summary.csv`
  - `results/tree_quantile_lgbm_q55_q60_q65_q70_q75_v1_branch_treebase_grid_summary.csv`

---

작성: 2026-07-11 KST

실험명: empirical wind->power proxy gfs850 OOF v1

목적: Leustagos식 풍향 조건부 forecast-to-power proxy가 실제로 신호가 있는지, LGBM teacher 없이 단순 `direction sector x wind-speed bin -> train 평균 발전량` 테이블로 확인했다. 제출 파일은 만들지 않았다.

설정:

- Branch: proxy 단독 진단. TREE 재학습 없음.
- Source: `gfs850`.
- Validation: leave-one-year OOF. `kpx_group_3` 2022는 target 부족으로 NaN이며 summary 평균에서 사실상 제외.
- Variants:
  - `speed_bin`: 풍속 bin별 평균 발전량.
  - `dir_speed_bin`: 풍향 sector + 풍속 bin별 평균 발전량, 부족한 cell은 speed/sector/global 평균 fallback.
- Fit rows: actual >= 10% capacity만 사용.
- Bins: direction 16 sectors, wind-speed bin width 0.5 m/s, min cell count 20.

결과:

| Variant | OOF score | nMAE | FiCR | Worst fold | 판단 |
|---|---:|---:|---:|---:|---|
| `speed_bin` | `0.545467` | `0.168754` | `0.259687` | `0.528717` | 풍속만 쓴 proxy |
| `dir_speed_bin` | `0.550824` | `0.168442` | `0.270090` | `0.534822` | 풍향 sector 추가로 개선 |

판단:

- 단순 테이블 proxy임에도 방향 sector 추가가 `+0.00536` OOF, FiCR `+0.01040`로 일관 개선했다.
- 이 값 자체는 최종 모델 score가 아니라 proxy 단독 score다. 다만 방향 조건부 power proxy가 유효하다는 강한 신호다.
- 다음 후보는 `gfs100`, `gfs10`, bin/sector sweep을 확인한 뒤, 가장 강한 proxy를 TREE feature로 `baseline + empirical_proxy` 형태로 넣는 것이다.

저장:

- `results/empirical_wind_power_proxy_oof_v1_gfs850_scores.csv`
- `results/empirical_wind_power_proxy_oof_v1_gfs850_summary.csv`
- `results/empirical_wind_power_proxy_oof_v1_gfs850_group_summary.csv`
- `results/empirical_wind_power_proxy_oof_v1_gfs850_fold_means.csv`
  - `results/empirical_wind_power_proxy_oof_v1_gfs850_predictions.csv`

---

작성: 2026-07-11 KST

문서 정리: current public best handoff refresh

목적: public 최고 모델이 `PINN25/TREE40/TCN35 + group3 pseudo2022`에서 `PINN floor35 + PINN25/TREE20/TCN55 + final floor10`로 바뀌었으므로, 다음 세션/에이전트가 혼동하지 않도록 handoff 문서를 갱신했다.

현재 최고:

```text
results/submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_finalfloor10_v1.csv
public score = 0.6386205415
1-nMAE = 0.8682636645
FiCR = 0.4089774184
submitted = 2026-07-11 01:21:21 KST
```

구조:

```text
PINN_floor = clip(PINN, 0.35 * capacity, capacity)
TCN_family = 0.30 * W24 + 0.40 * W72 + 0.30 * W168
final_raw = 0.25 * PINN_floor + 0.20 * TREE + 0.55 * TCN_family
final = clip(final_raw, 0.10 * capacity, capacity)
```

갱신 파일:

- `docs/best_model_usage.md`: 현재 최고, fast rebuild, from-scratch branch regeneration, hyperparameters, public best timeline.
- `AGENTS.md`: current best 및 작업 규칙 최신화.
- `.agents/windforecast_agent_context.md`: 다음 세션용 handoff 최신화.
- `docs/current_best_structure_for_review.md`: 리뷰용 현재 최고 요약 최신화.
- `docs/current_pipeline_map.md`: 현재 best 기준 branch/file map 최신화.
- `docs/model_feature_inventory.md`, `docs/pipeline_contract.md`, `docs/best model pipeline.md`: stale snapshot 경고 추가.

검증:

- 현재 최고 CSV는 `submission_pinn_lgbm_teacher_year_bagging.csv` + `submission_tree_lgbm_best_v2_l1.csv` + weighted-L1 TCN family를 사용해 `PINN floor35 / weights 0.25,0.20,0.55 / final floor10`으로 재현 가능함.
- 직접 비교에서 max absolute difference는 `3.64e-12`로 수치적으로 동일했다.

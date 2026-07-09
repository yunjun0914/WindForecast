# Pipeline Contract

작성일: 2026-07-09 KST
버전: v1

목적: TREE, PINN, SeqNN, teacher, feature builder, blend가 같은 입출력 규칙을 따르게 만든다. 리팩토링 후에도 현재 최고 구조를 재현하고, 새 실험이 어디를 바꾸는지 바로 추적할 수 있어야 한다.

## 0. 원칙

1. 현재 최고 기준은 `PINN 50 : TREE 50`이다.
2. validation은 leave-one-year-out OOF를 기본으로 한다.
3. test submission은 큰 OOF 개선 또는 사용자 명시 요청이 있을 때만 만든다.
4. train row도 test 상황과 맞게 OOF/crossfit teacher feature를 사용한다.
5. feature, teacher, model, blend, calibration은 서로 분리한다.
6. 실험 전에는 목적, 변경점, 고정점, validation, 기대효과, 위험, 생성 파일을 먼저 설명한다.

## 1. Canonical Entities

### Time Columns

| Name | Type | Meaning |
|---|---|---|
| `forecast_kst_dtm` | datetime64 | 예측 대상 시각 |
| `data_available_kst_dtm` | datetime64 | 예보 데이터 사용 가능 시각 |
| `kst_dtm` | datetime64 | label/SCADA 관측 시각 |
| `pred_year` | int | OOF에서 예측한 holdout year |
| `train_years` | string | OOF에서 학습한 years, 예: `2022,2023` |

### Target Groups

| Group | Capacity kWh | Note |
|---|---:|---|
| `kpx_group_1` | `21600` | VESTAS |
| `kpx_group_2` | `21600` | VESTAS |
| `kpx_group_3` | `21000` | UNISON |

### Fold Contract

| Fold | Train years | Predict year |
|---|---|---|
| 1 | `2023,2024` | `2022` |
| 2 | `2022,2024` | `2023` |
| 3 | `2022,2023` | `2024` |

모든 OOF 결과는 이 3개 fold를 기본으로 한다. 추가 fold나 다른 validation은 실험명에 명시한다.

## 2. Raw Data Contract

### Required Train Inputs

| File | Required columns |
|---|---|
| `data/train/ldaps_train.csv` | `forecast_kst_dtm`, `data_available_kst_dtm`, weather columns |
| `data/train/gfs_train.csv` | `forecast_kst_dtm`, `data_available_kst_dtm`, `grid_id`, `latitude`, `longitude`, weather columns |
| `data/train/train_labels.csv` | `kst_dtm`, `kpx_group_1`, `kpx_group_2`, `kpx_group_3` |
| `data/train/scada_vestas_train.csv` | `kst_dtm`, VESTAS turbine SCADA columns |
| `data/train/scada_unison_train.csv` | `kst_dtm`, UNISON turbine SCADA columns |

### Required Test Inputs

| File | Required columns |
|---|---|
| `data/test/ldaps_test.csv` | `forecast_kst_dtm`, `data_available_kst_dtm`, weather columns |
| `data/test/gfs_test.csv` | `forecast_kst_dtm`, `data_available_kst_dtm`, `grid_id`, `latitude`, `longitude`, weather columns |
| `data/sample_submission.csv` | `forecast_id`, `forecast_kst_dtm`, target group columns |

### Date Rules

- All datetime columns must be parsed with `pd.to_datetime`.
- Feature tables must be sorted by `forecast_kst_dtm`.
- Forward-fill is allowed for source gaps.
- Backfill is not allowed if it can leak later timestamps into earlier rows. If unavoidable, it must be documented.

## 3. Feature Table Contract

All feature builders must return a dataframe with this shape:

```text
forecast_kst_dtm
data_available_kst_dtm, optional but preferred
feature columns...
```

Rules:

- No target columns inside feature tables.
- No raw SCADA columns inside feature tables unless the feature is explicitly OOF/crossfit predicted.
- Train and test must be built by the same function path.
- Feature columns must be numeric after preprocessing.
- Infinite values must be replaced.
- Missing values must be handled inside the feature builder, not inside model code.

### Feature Family Names

New features should be named with a family prefix.

| Family | Prefix examples |
|---|---|
| calendar | `cal_`, or existing `sin_doy`, `cos_hod` |
| raw wind | `ldaps_ws`, `gfs_ws` |
| physics wind | `phys_` |
| density/power | `rho_`, `wind_power_`, `phys_*_air_density` |
| direction | `dir_`, `wd_`, `axis_`, `upwind_` |
| spatial | `grid_`, `near_`, `idw_`, `gradient_` |
| teacher | `scada_`, `teacher_` |
| power curve | `power_curve_` |
| weather ramp | `ramp_`, `delta_` |

Feature family is not cosmetic. Ablation, pruning, and importance must report family-level summaries.

## 4. Teacher Contract

Teacher means weather-trained proxy features. Raw SCADA cannot be injected into validation/test rows.

### Teacher Function Signature

Target shape:

```python
train_out, pred_out = build_teacher_features(
    train_weather,
    pred_weather,
    train_scada,
    group,
    train_years,
    pred_year=None,
    config=None,
)
```

### Teacher Output Rules

- `train_out` and `pred_out` must preserve row order and row count.
- Output must include all original weather columns plus teacher feature columns.
- Train rows must use OOF/crossfit predictions.
- Pred rows must use a model fit only on `train_years`.
- Teacher features must be clipped to physically reasonable ranges.
- Teacher backend must be recorded in metadata.

### Current Teacher Targets

Current PINN teacher targets:

- `scada_ws_mean`
- `scada_ws_std`
- `scada_ws_p10`
- `scada_ws_p50`
- `scada_ws_p75`
- `scada_ws_p90`
- `scada_ws_max`
- `scada_ws_cubic`
- `scada_ws_ramp`
- `scada_wd_sin`
- `scada_wd_cos`
- `scada_wd_concentration`
- `scada_wd_spread`
- `scada_wd_sin_std`
- `scada_wd_cos_std`
- `scada_ws_dir_sin`
- `scada_ws_dir_cos`

Current stable backend:

- `lgbm_time_oof`

Allowed backend candidates:

- `lgbm_time_oof`
- `rf_oob`
- `extra_trees_oof`
- `multioutput_lgbm`

Any backend change is a separate experiment.

## 5. Power-Curve Feature Contract

Power-curve features are SCADA-derived and must be leakage-safe.

Rules:

- Train rows use OOF power curves.
- Validation/test rows use power curve fitted only on training-period SCADA.
- Dirty SCADA power samples may be filtered by physical constraints.
- Output column must be prefixed with `power_curve_`.

Current legacy column:

- `power_curve_est`

Target future column:

- `power_curve_est`
- or more explicit variants, e.g. `power_curve_gfs_ws100_est`

## 6. Model Branch Contract

Every model branch should expose the same conceptual interface.

```python
fit_oof(config) -> oof_predictions, oof_scores, metadata
predict_test(config) -> test_predictions, metadata
```

### Branches

| Branch | Role | Current status |
|---|---|---|
| `tree_lgbm_best` | tuned group-specific LGBM | stable |
| `pinn_effective` | SCADA teacher + PINN | stable |
| `seqnn_direct` | DLinear/GRU/TCN/WaveNet direct forecast | planned |
| `calibration` | final postprocess only | planned |

### Branch Rules

- Branch code must not write `results/submission.csv` directly unless it is the blend/submission step.
- Branch output must be either standard OOF CSV or standard test prediction CSV.
- Branch-specific scores must be computed with the same metric code.
- Model internals may differ, but prediction schema must not.

## 7. Standard OOF Prediction Schema

All new OOF predictions must use long format.

Required columns:

| Column | Type | Meaning |
|---|---|---|
| `forecast_kst_dtm` | datetime64 | target timestamp |
| `pred_year` | int | holdout year |
| `train_years` | string | training years |
| `model_family` | string | `tree`, `pinn`, `seqnn`, `blend`, `calibration` |
| `model_name` | string | specific model id |
| `group` | string | target group |
| `actual` | float | true target |
| `pred` | float | unclipped or clipped prediction, see `is_clipped` |
| `is_clipped` | bool | whether `pred` has already been capacity-clipped |

Optional columns:

- `capacity`
- `fold_id`
- `feature_set`
- `teacher_backend`
- `train_policy`
- `model_seed`
- `notes`

File naming:

```text
results/oof_{model_name}_{feature_set}_{YYYYMMDD_HHMM}.csv
```

During migration, legacy wide OOF files are allowed only with an adapter.

## 8. Standard OOF Score Schema

Required columns:

| Column | Type | Meaning |
|---|---|---|
| `pred_year` | int or `all` | holdout year or overall |
| `train_years` | string | training years |
| `model_family` | string | model family |
| `model_name` | string | model id |
| `group` | string | target group or `fold_mean`/`overall_mean` |
| `score` | float | `0.5 * (1 - nmae) + 0.5 * ficr` |
| `nmae` | float | group/fold NMAE |
| `ficr` | float | group/fold FICR |
| `n_rows` | int | scored rows |

Summary files must include:

- `mean_score`
- `mean_nmae`
- `mean_ficr`
- `worst_fold`
- `std_score`
- `n_folds`

## 9. Standard Test Prediction Schema

Branch-level test prediction must use wide submission-like format.

Required columns:

| Column | Type |
|---|---|
| `forecast_id` | string |
| `forecast_kst_dtm` | datetime64/string |
| `kpx_group_1` | float |
| `kpx_group_2` | float |
| `kpx_group_3` | float |

Rules:

- Row order must match `data/sample_submission.csv`.
- `forecast_id` must exactly match sample submission.
- No missing predictions.
- Predictions must be clipped to `[0, capacity]` before final submission.
- Branch-level output name must not be `results/submission.csv`.

File naming:

```text
results/testpred_{model_name}_{feature_set}.csv
results/submission_{model_name}.csv
```

`results/submission.csv` is reserved for final selected blend only.

## 10. Blend Contract

Blend input:

- two or more standard test prediction CSVs
- or two or more standard OOF prediction CSVs for validation

Default:

```text
final = 0.5 * pinn_effective + 0.5 * tree_lgbm_best
```

Rules:

- Default `PINN 50 : TREE 50` must not be changed without user approval.
- Weight search is validation-only unless user asks for test submission.
- Blend output must be capacity-clipped.
- Blend metadata must record all input files and weights.

## 11. Calibration Contract

Calibration is a final postprocess, not part of model training.

Allowed calibration candidates:

- delta calibration
- quantile/FICR calibration
- disagreement-based correction
- residual correction

Rules:

- Calibration must consume frozen model predictions.
- It must not modify feature builders or teacher code.
- It must be evaluated on OOF before test.
- Calibration output must be named separately from base blend.

## 12. Experiment Contract

Before any experiment, write or state:

```text
실험명:
목적:
변경점:
고정점:
사용 데이터/feature:
validation:
기대 효과:
위험:
생성 파일:
승격 기준:
```

Suggested promotion criteria:

- mean score `+0.005` 이상
- group3 개선
- worst fold 악화 없음
- FICR 개선이 NMAE 큰 손상을 만들지 않음
- public 제출은 OOF `+0.01`급 개선 또는 사용자 명시 요청

## 13. Directory Contract

Target structure:

```text
utils/
  data/
  features/
  teacher/
  models/
  validation.py
  metrics.py
  submission.py

experiments/
  run_tree_oof.py
  run_pinn_oof.py
  run_seqnn_oof.py
  run_blend_oof.py
  archive/

docs/
  current_pipeline_map.md
  pipeline_contract.md
  external_code_reference.md
```

Rules:

- Root-level scripts should be stable entrypoints only.
- Diagnostic scripts go under `experiments/`.
- Deprecated experiments go under `experiments/archive/` or are deleted after their conclusion is recorded.
- Result artifacts not needed for current reproduction should be archived or removed after user approval.

## 14. Migration Plan

Step 1: Keep current best path untouched.

- `predict_pinn_effective_grid_g1_year_bagging.py`
- `predict_power_lgbm_best.py`
- `blend_submission_files.py`

Step 2: Add adapters that convert legacy OOF files to standard OOF schema.

Step 3: Move shared data/feature logic behind stable functions.

Step 4: Port TREE branch first because it is faster and easier to verify.

Step 5: Port PINN branch after teacher contract is stable.

Step 6: Add SeqNN branch only after OOF/test prediction schemas are stable.

## 15. Non-Negotiable Checks

Every branch or blend must check:

- columns exist
- row count matches expected data
- timestamps align
- no missing predictions
- predictions are finite
- final predictions are clipped
- OOF rows do not use holdout-year raw SCADA
- train/test feature columns are identical

Failure should raise an error, not silently continue.

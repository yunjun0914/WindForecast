# Aggressive Minimal Feature Set

작성일: 2026-07-09 01:19:50 +09:00

목적: 기존 TREE의 511개 feature를 대회 코드 스타일에 맞춰 관리 가능한 70개 feature profile로 줄인다. 기존 full feature 경로는 control로 유지한다.

## Profile

| 항목 | 값 |
|---|---|
| profile name | `aggressive_minimal_v1` |
| 대상 | TREE branch |
| feature count | `70` |
| control | `full_v2` 511 features |
| test submission | 생성하지 않음 |

## 설계 원칙

- `all_meteo` raw 전체 투입을 중단하고, 명시적으로 선택한 meteo만 유지한다.
- feature family 단위로 줄인다.
- 대량 polynomial/regime feature는 제거한다.
- cloud/precipitation/meteo_other는 제거한다.
- wind, direction, spatial, pressure/physics, calendar 중심으로 남긴다.

## 유지 Family

| Family | 내용 |
|---|---|
| calendar | `sin/cos doy`, `sin/cos hod`, hour-day 조합 |
| core wind | LDAPS 10m/50m, GFS 10m/80m/100m/850hPa |
| spatial wind | grid max/range/p90, near, gradient, upwind-downwind |
| direction | axis/cross/southwest/upwind component |
| physics | `rho * ws^3`, shear, gust, lapse, pressure, PBL |
| selected meteo | GFS shortwave, 850 humidity, 700 temp, LDAPS surface pressure |
| power curve | `power_curve_est`는 OOF 단계에서 추가 |

## 제거 Family

- `wind_regime_poly` 대량 feature
- cloud
- precipitation
- meteo_other
- 과도한 raw meteo
- 과도한 shear/veer 조합
- 중복 raw GFS 500hPa u/v

## OOF 결과

같은 tuned LGBM hyperparameter를 사용하고 feature profile만 바꿨다.

| Model | Feature count | Mean score | Mean nMAE | Mean FICR | Worst fold | 판단 |
|---|---:|---:|---:|---:|---:|---|
| full_v2 TREE | 511 | `0.62361` | `0.12851` | `0.37573` | 기존 기준 | control |
| aggressive_minimal_v1 TREE | 70 | `0.62085` | `0.12980` | `0.37150` | `0.60369` | 유지 후보 |
| aggressive_minimal_rolling_v1 TREE | 160 | `0.62396` | `0.12809` | `0.37602` | `0.60563` | rolling 유지 후보 |
| aggressive_minimal_context_v1 TREE | 211 | `0.62343` | `0.12794` | `0.37481` | `0.60494` | 제출 후보 아님 |

하락폭은 약 `-0.00276`으로, 계획의 허용 기준인 `-0.005` 이내다. 관리성 개선을 고려하면 다음 feature 설계의 기준 profile로 쓸 가치가 있다.

## Rolling Extension

작성일: 2026-07-09 01:39:51 +09:00

`aggressive_minimal_rolling_v1`은 70개 minimal feature에 주요 wind/weather time-context 90개를 추가한다.

추가 대상:

- `gfs_ws10_speed`
- `gfs_ws100_speed`
- `gfs_ws850_speed`
- `gfs_surface_0_gust`
- `ldaps_ws10_speed`
- `ldaps_ws50_max_speed`
- `phys_gfs_ws850_grid_max`
- `phys_gfs_ws850_grid_p90`
- `phys_ldaps_ws50max_grid_max`

각 대상에 대해 다음 feature를 추가한다.

- `lag1`, `lag3`
- `lead1`, `lead3`
- `roll3_mean`, `roll3_std`
- `roll6_mean`, `roll6_std`, `roll6_max`
- `lead1_minus_lag1`

판단:

- 70개 minimal 대비 `+0.00311`.
- 기존 full_v2 대비 `+0.00035`.
- 큰 점프는 아니지만, 외부 대회/연구 코드에서 반복되는 time-context family가 우리 OOF에서도 유효함을 확인했다.
- 다음 후보는 rolling family를 유지한 상태에서 LDAPS/GFS sister model 또는 direction-aware spatial aggregation을 붙이는 방향이 자연스럽다.

## Context Extension

작성일: 2026-07-09 01:50:58 +09:00

`aggressive_minimal_context_v1`은 rolling profile 위에 외부 대회/연구 코드에서 자주 쓰이는 세 family를 compact하게 추가했다.

추가 family:

- forecast lead-time / issue-cycle: `forecast_lead_hours`, lead 24h cycle, `data_available` hour/day cycle
- direction circular: 주요 u/v level의 normalized `dir_cos`, `dir_sin`, 일부 vertical veer
- spatial summary: 주요 wind level의 grid `q25`, `q75`, `IQR`, 기존 `std`, `near_minus_grid_mean`

결과:

- rolling 단독 대비 mean score `-0.00053`.
- nMAE는 `0.12809 -> 0.12794`로 개선.
- FICR은 `0.37602 -> 0.37481`로 악화.
- group3는 소폭 좋아졌지만 group2 FICR 하락이 더 컸다.

판단:

- 1/3/4를 한꺼번에 붙이는 방식은 제출 후보가 아니다.
- lead/cycle, direction, spatial을 각각 분리 ablation해야 한다.
- 현재 TREE feature profile 기준은 `aggressive_minimal_rolling_v1`이 더 좋다.

## 생성 파일

- `results/power_lgbm_best_v2_l1_aggressive_minimal_v1_scores.csv`
- `results/power_lgbm_best_v2_l1_aggressive_minimal_v1_predictions.csv`
- `results/power_lgbm_best_v2_l1_aggressive_minimal_v1_summary.csv`
- `results/standard_oof_tree_lgbm_best_v2_l1_aggressive_minimal_v1.csv`
- `results/power_lgbm_best_v2_l1_aggressive_minimal_rolling_v1_scores.csv`
- `results/power_lgbm_best_v2_l1_aggressive_minimal_rolling_v1_predictions.csv`
- `results/power_lgbm_best_v2_l1_aggressive_minimal_rolling_v1_summary.csv`
- `results/standard_oof_tree_lgbm_best_v2_l1_aggressive_minimal_rolling_v1.csv`
- `results/power_lgbm_best_v2_l1_aggressive_minimal_context_v1_scores.csv`
- `results/power_lgbm_best_v2_l1_aggressive_minimal_context_v1_predictions.csv`
- `results/power_lgbm_best_v2_l1_aggressive_minimal_context_v1_summary.csv`
- `results/standard_oof_tree_lgbm_best_v2_l1_aggressive_minimal_context_v1.csv`

## 다음 판단

이 profile은 당장 최고 제출 후보가 아니라, 다음 리팩토링/feature 재설계의 기준점이다. 다음 단계에서는 70개 profile 위에서 direction-conditioned effective wind 또는 SCADA availability weighting처럼 큰 구조 후보를 붙이는 편이 낫다.

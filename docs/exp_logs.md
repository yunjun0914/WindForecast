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

# 실험 로그 (exp_logs)

이 세션에서 작성한 스크립트들을 시간 순서대로 정리한 기록입니다. 각 스크립트가 뭘 검증했고 결과가 어땠는지, 그 결과로 무슨 결정을 내렸는지 위주로 씁니다.

## 공통 모듈

| 파일 | 역할 |
|---|---|
| `utils/preprocessing.py` | LDAPS/GFS 격자 집계, 바람 파생피처(speed), 시간주기 인코딩, `build_weather_features()`/`build_group_dataset()` |
| `utils/metrics.py` | 실제 대회 채점식 구현 — `group_nmae_ficr`(NMAE+FICR), `total_score`, `make_group_scorer`(sklearn scorer) |
| `utils/power_curve.py` | SCADA(풍속-발전량) 기반 그룹별 경험적 파워커브 피팅 (`fit_group_power_curve`, `add_power_curve_feature`) |
| `models/random_forest.py`, `models/lgbm.py`, `models/xgb.py` | 각각 `train(X, y)` — train/val 랜덤분리 + 학습된 모델 반환 |

## 스크립트별 로그

### `train.py` — 그룹×모델 검증
그룹별(1/2/3) RF/LGBM/XGB 학습 + 단순평균 앙상블, `group_nmae_ficr`로 검증. 매번 피처셋이 바뀔 때마다 재실행해서 회귀 여부 확인하는 용도로 계속 사용.

- 최초(바람 101개 피처): 그룹별 NMAE 7.7~8.3%
- 바람전용 58개로 축소 후: NMAE 소폭 상승(8.0~8.5%)이지만 실제 리더보드는 더 좋았음(아래 참고)
- FICR 도입 후(10% 미만 출력 제외 + 정산 임계값 반영): 그룹별 total_score 0.63~0.66 (랜덤분리 기준, 나중에 이 지표 자체가 낙관적이었다는 게 밝혀짐)

### `tune.py` — RandomizedSearchCV 하이퍼파라미터 탐색
그룹별로 RF/LGBM/XGB 각각 `RandomizedSearchCV`(scoring=커스텀 total_score, cv=`KFold(shuffle=True)`)로 15회×3-fold 탐색.

- **결과**: LGBM/XGB는 튜닝 후 CV 점수가 크게 개선(예: LGBM group_2 0.6535→0.7009), RF는 오히려 악화
- **실제 리더보드에서는 정반대**: 튜닝된 LGBM+XGB 조합이 0.6068(튜닝 전 베이스라인)보다 낮은 0.6029로 하락 → **과적합 확인**. 랜덤 K-fold가 2022~2024 안에서만 섞이다 보니 "미래 연도로 일반화되는지"를 전혀 테스트 못 했던 게 원인으로 결론

### `evaluate_tuned_time_holdout.py` — 튜닝 결과, 검증방식만 맞춰서 재확인
`tune.py`가 보여준 "튜닝 후 0.68대" 점수가 랜덤 K-fold(`tune.py`)와 시간기준 홀드아웃(`evaluate_time_holdout.py`)이라는 **서로 다른 검증방식**끼리 비교된 것이었다는 점을 뒤늦게 인지. 같은 시간기준 홀드아웃(2022~2023 학습/2024 검증)으로 튜닝된 하이퍼파라미터를 다시 채점해서, 디폴트와 공정 비교.

| 모델 | 디폴트(시간홀드아웃) | 튜닝됨(시간홀드아웃) | 차이 |
|---|---:|---:|---:|
| RF | 0.5925 | 0.5934 | +0.0009 |
| LGBM | 0.5957 | 0.5899 | -0.0058 |
| XGB | 0.5869 | 0.5889 | +0.0020 |
| 앙상블(3개) | 0.5978 | 0.5953 | -0.0025 |

- **핵심 정정**: 같은 방식으로 비교하면 튜닝 전/후 차이가 ±0.005 수준으로 미미함. 앞서 봤던 "0.68 vs 0.60"의 8%p 격차는 대부분 **튜닝의 과적합이 아니라 랜덤 K-fold 자체가 원래 낙관적**이었기 때문
- 다만 이 공정 비교에서도 앙상블 기준 튜닝이 미세하게 손해(-0.0025)이고, 이는 실제 리더보드에서 튜닝 조합이 디폴트보다 낮았던 것과 방향이 일치 → **"튜닝 보류, 디폴트 사용"이라는 결론 자체는 유지**하되, 근거를 "튜닝이 심각한 과적합을 유발해서"에서 "검증방식이 원래 낙관적이었고, 공정 비교해도 튜닝 이득이 없어서"로 정정

### `evaluate_ensemble.py` — 앙상블 조합 비교
RF(디폴트)/LGBM(튜닝)/XGB(튜닝)의 모든 조합(단일~3개 평균)을 OOF로 비교.

- **결과**: 3그룹 전부 `lgbm+xgb`(RF 제외) 조합이 1등. RF를 섞으면 어떤 조합이든 점수가 떨어짐
- 이 결론대로 RF를 뺀 제출을 올렸지만 실제 리더보드는 더 나빠짐(0.6029) → 위 튜닝 과적합 문제와 같은 원인으로 폐기, RF 다시 포함

### `evaluate_time_holdout.py` — 신뢰할 수 있는 검증체계 구축
2022~2023 학습 / 2024 검증으로 시간기준 홀드아웃 구성. 실제 미래 연도 일반화와 방향이 맞는지 확인하고, `permutation_importance`로 피처 중요도 계산.

- **핵심 발견**: 이 지표 기준 그룹별 점수(0.55~0.63)가 랜덤분리 기준(0.66~0.70)보다 확실히 낮게 나와, 실제 리더보드가 낮았던 것과 방향이 일치 → 이후 모든 의사결정의 기준 지표로 채택
- **피처 중요도**: `gfs_ws850_speed`(850hPa, 상공 풍속)가 압도적 1위. 풍향(direction) 계열 전부, 임계값 인디케이터(`is_below_cutin` 등), 요일/월 주기 인코딩이 거의 0 또는 마이너스 → 이후 62개→42개 피처로 정리하는 근거가 됨

### `diagnose_bias.py` — 출력구간별 편향 진단
2024 홀드아웃에서 앙상블 예측을 실제값 십분위로 나눠 편향(bias) 확인.

- **핵심 발견**: 저출력 구간은 과대예측(+4~8%), 고출력(피크) 구간은 심하게 과소예측(-11%~-25%, group_3이 최악). 트리 앙상블의 "평균으로 수렴" 현상. FICR은 실제 발전량 가중평균이라 이 피크 편향이 정산금 점수에 직격탄

### `calibrate_and_evaluate.py` — Isotonic 후보정
2022~2023 OOF 예측으로 `IsotonicRegression`(raw_pred→actual) 학습, 2024 홀드아웃으로 검증. 그룹별 개별 보정 / 3그룹 풀링(설비용량 %로 정규화 후 합산) / group_2 단독 보정 세 가지 비교.

- **그룹별 개별 보정**: group_1/2는 개선(+0.008/+0.004), group_3은 데이터가 적어서(2023년 1년치) 소폭 악화(-0.002)
- **3그룹 풀링 보정이 항상 제일 좋음**: group_3도 개별보정보다 나아지고 raw와 거의 같아짐 → 최종 채택
- **group_2 단독 보정**은 풀링보다 항상 약간 못함 → 기각

### `evaluate_group2_transfer.py`, `evaluate_group2_only.py` — group_2 모델 전이 실험
"group_1/3 데이터는 아예 안 쓰고 group_2로만 학습한 모델 하나를 설비용량 비율로 스케일링해서 3그룹에 다 쓰면 어떨까"라는 아이디어 검증.

- **결과**: 실제로 3그룹 다 개선됨(group_1 0.6027→0.6146, group_3 0.5587→0.5792) — group_1/3 자체 라벨에 날씨로 설명 안 되는 노이즈가 많다는 방증
- **다만 실제 제출에는 채택 안 함**: 대회 규정상 3개 그룹을 각자 의미 있게 예측해야 하는 구조라, 그룹 구분 없이 값을 복사하는 방식은 최종 제출 전략으로는 부적절하다고 판단해 이 결과는 진단/참고용으로만 남기고 최종 파이프라인에는 반영하지 않음

### `predict.py` — 최종 제출 파이프라인 (계속 갱신됨)
버전별 변화:
1. RF+LGBM+XGB 디폴트, 바람전용 58개 피처 → 실제 리더보드 0.6068 (`baseline(only wind)`, 이 세션 최고 기록 중 하나)
2. LGBM+XGB만 튜닝 파라미터 사용 → 0.6029로 하락 (과적합)
3. RF(디폴트)+LGBM/XGB(튜닝) → 0.6053
4. 피처 정리(42개+파워커브) + 전부 디폴트 3모델 앙상블 → 제출
5. **위 4번 + 풀링 isotonic 보정** → 실제 리더보드 **0.6087** (최종, 이 세션 최고 기록)

### `predict_rf_only.py`, `predict_group2_only.py`
각각 "RF만 단독 사용", "group_2 모델만으로 3그룹 예측" 아이디어를 실제 test set에 적용해보는 용도로 만든 1회성 스크립트. 최종 파이프라인(`predict.py`)에는 병합되지 않음.

### `train_pinn.py`, `models/pinn.py`, `utils/pinn_data.py` — PINN 물리모델 재정비
트리 앙상블의 평균수렴/피크 과소예측을 보완하려고 물리 기반 PINN을 별도 라인으로 구축. 초기 구조는 `P = 0.5*rho*A*v^3*C_eff` + 선형 ODE EMA 정확해 + collocation 물리손실 + HOD bias.

초기 PINN은 물리식 자체보다 입력 풍속이 너무 단순한 게 병목이었다. 최초엔 LDAPS 10m 16격자 평균을 hub height로 외삽하고 제조사별 선형 보정만 적용했으며, 성능은 HOD-only 기준 대략:

| 버전 | group_1 | group_2 | group_3 | 평균 |
|---|---:|---:|---:|---:|
| 초기 PINN HOD-only | 0.5656 | 0.5657 | 0.5459 | 0.5590 |

핵심 수정:
- **EMA tail 보존**: `K=24`에서 잘린 무한 EMA tail mass를 가장 오래된 lag에 합쳐 정상상태 gain을 1로 보존. 수식상 맞는 수정이지만 학습된 `tau≈2h`라 점수 영향은 작았음.
- **capacity-normalized HOD bias**: 기존 `hod_bias`가 kWh 단위로 직접 학습되어 Adam step 때문에 `±20kWh` 근처에서 멈췄던 버그를 발견. 실제 HOD 잔차는 `1,000~2,000kWh` 규모였으므로 bias를 설비용량 비율로 parameterize하고 `capacity*bias_ratio`로 적용. 이 수정만으로 평균이 약 `0.5357→0.5590`으로 상승.
- **평가 clamp**: 학습은 raw prediction으로 유지하고, 평가/제출에서만 `clip(pred, 0, capacity)` 적용. 물리 백본 자체는 음수가 없었지만 calendar bias 추가 후 낮은 풍속에서 음수 예측이 생겼음.
- **per-hour train-only bias**: train loss는 크게 줄였지만 validation 점수는 거의 중립/소폭 악화. HOD 신호를 일부 훔쳐가는 경향이 있어 기본 비활성화.
- **moy/year bias**: month 반복 패턴과 train-only year baseline을 시도했지만 검증 악화. `g_moy`와 중복되고 train-period weather persistence를 먹는 것으로 판단해 기본 비활성화.

### `diagnose_pinn_bias.py`, `diagnose_pinn_physics.py` — PINN bias/물리 진단
`diagnose_pinn_bias.py`로 HOD별 실제 잔차 평균과 학습 bias 크기를 비교. capacity-normalized 수정 전에는 bias RMS가 `~20kWh`에 불과했고, 수정 후 `1,600~2,250kWh`로 실제 잔차 규모를 흡수함을 확인.

`diagnose_pinn_physics.py`로 `C_eff`, `P_phys`, group prediction 음수 여부를 확인. 물리 백본(`C_eff`, `P_phys`, no-bias prediction)은 음수 0건이었고, 최종 음수는 calendar bias 때문에 발생한다는 결론. 따라서 학습 clamp가 아니라 평가/제출 clamp만 채택.

### `docs/pinn_data_strategy.md` — PINN 데이터 전략 정리
PINN 실험의 핵심 결론을 별도 문서로 정리. 결론은 명확함: 병목은 지배방정식이 아니라 **forecast field에서 실제 현장 풍속분포를 복원하는 단계**.

작동한 데이터 개선:

| 버전 | group_1 | group_2 | group_3 | 평균 |
|---|---:|---:|---:|---:|
| HOD-only baseline | 0.5656 | 0.5657 | 0.5459 | 0.5590 |
| + GFS 850 wind correction | 0.5841 | 0.5871 | 0.5613 | 0.5775 |
| + 확장 forecast→SCADA ridge 보정 | 0.5961 | 0.6050 | 0.5644 | 0.5885 |

확장 보정에는 LDAPS hub-proxy wind, LDAPS 50m max/min wind, boundary-layer wind, BLH, GFS 80/100m wind, GFS 850hPa wind/u/v, GFS gust를 사용. 기존 permutation importance에서 `gfs_ws850_speed`가 강했던 것이 PINN에서도 그대로 확인됨.

LDAPS 16격자 `v_std`를 사용한 Gaussian wind distribution 적분도 구현. UNISON에는 도움이 됐지만 VESTAS에는 단독으로는 손해였고, 이후 SCADA teacher 기반 `v_std`로 대체하는 방향이 더 강력함.

### `evaluate_scada_wind_teacher.py` — SCADA를 teacher로 쓰는 wind distribution 예측
SCADA를 test 입력으로 직접 쓰는 대신, train SCADA를 teacher target으로 사용해서 forecast 변수로 현장 풍속분포를 예측할 수 있는지 검증.

Target:
- `scada_ws_mean`
- `scada_ws_std`
- `scada_ws_p10`
- `scada_ws_p50`
- `scada_ws_p90`

2022~2023 학습 / 2024 검증에서 RandomForest teacher의 예측 가능성:

| group | mean R2 | std R2 | p90 R2 |
|---|---:|---:|---:|
| group_1 | 0.805 | 0.341 | 0.821 |
| group_2 | 0.814 | 0.627 | 0.817 |
| group_3 | 0.806 | 0.575 | 0.811 |

`ws_mean/p50/p90`이 2024 holdout에서도 R2≈0.8 수준이라, SCADA teacher를 통해 site wind reconstruction이 가능하다고 판단.

### SCADA teacher를 PINN 입력으로 연결
`train_pinn.py`에서 group별 teacher를 fit하고, PINN 입력을 다음처럼 교체:

```text
v     = predicted scada_ws_mean
v_std = 0.5*predicted scada_ws_std + 0.5*(predicted p90 - predicted p10)/2.563
```

두 모드로 검증:

| teacher fit 방식 | group_1 | group_2 | group_3 | 평균 | 해석 |
|---|---:|---:|---:|---:|---|
| 전체 SCADA 사용 | 0.7056 | 0.6869 | 0.6349 | 0.6758 | 2024 SCADA 누수 포함, 상한선 |
| 2024 이전 SCADA만 사용 | 0.6341 | 0.6339 | 0.5886 | 0.6189 | honest time-holdout |

중요: 전체 SCADA fit 결과는 2024 검증에 대한 누수라 제출 성능 추정으로 쓰면 안 됨. 하지만 상한선으로는 의미가 크고, honest 버전도 기존 PINN 최고 `0.5885`에서 `0.6189`로 크게 상승. 목표였던 `0.62~0.63`에 거의 도달.

현재 기본은 `HONEST_SCADA_TEACHER_HOLDOUT=True`로, time-holdout 검증 시 teacher가 2024 SCADA를 보지 않도록 설정.

## 최종 결론 요약

| 항목 | 채택된 방식 |
|---|---|
| 피처 | 바람 관련 42개(원본+파생 speed) + 그룹별 SCADA 파워커브 1개 = 43개. 풍향/임계값인디케이터/요일·월 주기는 permutation importance로 제거 |
| 모델 | 그룹별 RF+LGBM+XGB 디폴트 하이퍼파라미터 단순평균 앙상블 (튜닝은 공정 비교(시간홀드아웃) 시 이득 없어 보류) |
| 후보정 | 3그룹 풀링 isotonic 보정 (설비용량 % 정규화 후 학습, 그룹별 적용) |
| 검증 지표 | 시간기준 홀드아웃(2022~2023 학습/2024 검증) — 랜덤 K-fold는 실제 성능과 방향이 안 맞아 폐기 |
| 실제 리더보드 최종 점수 | 0.6087 (1-nMAE 0.8654, FICR 0.3520) |

## 현재 PINN 결론 요약

| 항목 | 현재 최선 |
|---|---|
| 핵심 병목 | forecast→site wind distribution 복원 |
| PINN 입력 | SCADA teacher가 예측한 group별 `ws_mean/ws_std/quantile spread` |
| 물리식 | `0.5*rho*A*v^3*C_eff`, Gaussian wind distribution 적분, EMA tail 보존 |
| bias | capacity-normalized HOD bias만 기본 사용 |
| 비활성화 | per-hour train-only, moy bias, year bias |
| honest time-holdout 점수 | group_1 0.6341 / group_2 0.6339 / group_3 0.5886 / 평균 0.6189 |
| 다음 후보 | group_3 개선: VESTAS/UNISON teacher sharing, RF teacher→LGBM/XGB/ensemble, teacher target/feature 정교화 |

## 앞으로의 계획

### 트리 앙상블에도 SCADA teacher 붙이기
현재 RF/LGBM/XGB 메인 제출 파이프라인은 SCADA를 직접 쓰지 않고, forecast weather + SCADA power curve feature 정도만 사용한다. PINN 실험에서 확인된 가장 큰 이득은 `forecast → site SCADA wind distribution` teacher였으므로, 이 신호를 트리 앙상블에도 넣는 실험이 필요하다.

후보 피처:
- group별 `pred_scada_ws_mean`
- group별 `pred_scada_ws_std`
- group별 `pred_scada_ws_p10/p50/p90`
- `pred_scada_ws_p90 - pred_scada_ws_p10`
- 기존 forecast wind와 teacher wind의 차이/비율

검증 방식:
- teacher는 반드시 time-holdout 기준으로 train 기간 SCADA만 fit해야 함
- 2024 검증에서는 2024 SCADA를 teacher fit에 사용하지 않는 honest setup 유지
- RF/LGBM/XGB 각각에 teacher 피처를 추가한 뒤 기존 default ensemble 및 isotonic calibration과 비교

기대 효과:
- 트리 모델은 이미 0.60대 성능을 갖고 있으므로, SCADA teacher 피처가 들어가면 PINN에서 확인한 site wind reconstruction 이득을 더 직접적으로 흡수할 가능성이 큼.
- 특히 group_3은 PINN teacher honest에서도 아직 0.5886으로 낮으므로, 트리 모델 + SCADA teacher + 기존 isotonic 보정 조합이 다음 주요 상승 후보.

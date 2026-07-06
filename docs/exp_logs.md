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

### 데이터 중심 추가 개선 아이디어
이번 세션의 가장 큰 결론은 모델 구조보다 **forecast field에서 실제 현장 상태를 얼마나 잘 복원하느냐**가 성능을 좌우한다는 점이다. PINN 자체, tree 자체보다 `SCADA teacher`, `group_2/VESTAS transfer`, `mixed teacher`, `group별 ensemble recipe`가 더 큰 개선을 만들었다. 현재 검증상 `0.6307`까지 왔고, 1등권(`0.6456`)과 목표 상단(`0.65~0.66`)을 노리려면 다음은 데이터 재구성 쪽이 핵심이다.

후보 아이디어:

1. **SCADA teacher 고도화**
   - 현재 teacher는 RandomForest multi-output 하나로 `ws_mean/std/p10/p50/p90`을 예측한다.
   - 다음은 RF/LGBM/XGB/ExtraTrees/CatBoost teacher ensemble, target별 모델 분리, group별 teacher weight 탐색을 시도.
   - teacher target도 단순 풍속 통계뿐 아니라 `power-curve inverse wind`, `effective wind`, `ramp`, `gust-adjusted wind`, `turbulence proxy`로 확장.

2. **group_3 VESTAS prior 정교화**
   - group_3는 UNISON-only보다 VESTAS/group_2 신호를 섞을 때 좋아졌다.
   - 현재는 `30% UNISON + 70% VESTAS group2 teacher`와 group2 tree transfer가 강함.
   - 다음은 시간대/풍속구간/계절별로 VESTAS prior weight를 다르게 주는 adaptive teacher mixing.
   - 예: low wind에서는 UNISON teacher, rated 근처에서는 VESTAS/group2 teacher를 더 신뢰하는 piecewise mixing.

3. **공간정보/격자 선택 재검토**
   - LDAPS 16격자 평균, GFS nearest grid만 쓰는 방식은 terrain speed-up을 충분히 표현 못 할 수 있음.
   - 격자별 풍속을 평균/최대/상위분위수/풍상방향 weighted average로 재구성.
   - 풍향을 단순 제거하지 말고, ridge orientation 또는 grid upwind/downwind relation과 결합해 terrain-aware feature로 재도입.

4. **lead-time / forecast issue-time bias**
   - 동일한 forecast_kst_dtm이라도 예보 생성 시점과 lead time에 따라 bias가 다를 수 있음.
   - LDAPS/GFS 원본에 issue time 또는 forecast lead 정보가 있으면 lead별 bias correction/teacher feature를 추가.
   - 특히 FICR은 피크 시간대 오차에 민감하므로 ramp 구간 lead-time bias가 중요할 가능성.

5. **SCADA 품질/가동상태 필터링**
   - SCADA teacher target을 만들 때 정지/curtailment/센서 이상치를 더 강하게 제거하면 teacher가 발전가능 wind를 더 잘 배울 수 있음.
   - 풍속은 높은데 발전량이 낮은 curtailment-like 구간, turbine별 outlier, missing turbine count를 feature/weight로 반영.

6. **시간 alignment 재검토**
   - SCADA는 10분 단위, labels/weather는 시간 단위라 집계 방식이 중요하다.
   - 현재 hourly mean 중심인데, 발전량은 `v^3` 비선형이라 mean wind보다 `E[v^3]` 또는 상위분위 풍속이 더 맞을 수 있음.
   - 정각 기준 window를 `[-50,0]`, `[0,+50]`, centered 등으로 바꿔 teacher target을 만들어 비교.

7. **label noise / group transfer 활용**
   - group_2 모델이 group_1/group_3에도 잘 먹힌 것은 일부 group label에 설명 안 되는 noise가 있음을 시사.
   - group별 label을 독립 target으로만 보지 말고, capacity-normalized shared latent power index를 만들고 각 group을 보정하는 구조를 시도.
   - tree/PINN 모두 `site common power index + group residual` 형태로 재구성 가능.

8. **validation split 다변화**
   - 현재는 2024 단일 holdout에 최적화되어 있다.
   - 가능한 범위에서 rolling yearly/monthly holdout, 고풍속 이벤트 holdout, 계절별 holdout을 추가해 데이터 recipe가 특정 2024 패턴만 외우는지 확인.

우선순위:
1. SCADA teacher ensemble/target 확장
2. group_3 adaptive VESTAS prior
3. 격자 선택/terrain-aware wind reconstruction
4. SCADA quality filtering과 hourly aggregation 재검토

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

### `evaluate_scada_teacher_time_holdout.py`, `calibrate_scada_teacher_time_holdout.py` — 트리 모델에 SCADA teacher 피처 추가
PINN에서 큰 효과를 낸 SCADA teacher wind distribution을 RF/LGBM/XGB 입력 피처로 추가. Honest setup으로 teacher는 2024 이전 SCADA만 보고 fit하고, 2024 검증에는 예측된 teacher 피처만 사용.

추가 피처:
- `pred_scada_ws_mean/std/p10/p50/p90`
- `pred_scada_ws_iqr`
- `pred_scada_ws_sigma_q`
- `pred_scada_ws_mean_minus_gfs100`
- `pred_scada_ws_mean_minus_gfs850`

Raw time-holdout 결과:

| 모델 | group_1 | group_2 | group_3 | 평균 |
|---|---:|---:|---:|---:|
| RF + teacher | 0.5991 | 0.6368 | 0.5611 | 0.5990 |
| LGBM + teacher | 0.6005 | 0.6214 | 0.5571 | 0.5930 |
| XGB + teacher | 0.5962 | 0.6192 | 0.5520 | 0.5891 |
| all3 ensemble + teacher | 0.6006 | 0.6282 | 0.5585 | 0.5958 |

Pooled isotonic까지 붙인 결과:

| 모델셋 | raw 평균 | pooled isotonic 평균 |
|---|---:|---:|
| RF-only + teacher | 0.5990 | 0.5986 |
| all3 + teacher | 0.5958 | 0.5967 |

해석:
- SCADA teacher 피처는 RF 단독에는 기존 RF time-holdout 평균(약 0.5925) 대비 개선.
- 그러나 LGBM/XGB와 all3 단순 앙상블에는 아직 이득이 뚜렷하지 않음. 기존 all3 baseline 평균(약 0.5978)보다 낮거나 비슷.
- PINN에서는 teacher가 직접 물리 입력(`v/v_std`)이 되므로 큰 효과가 났지만, 트리 모델은 이미 원 forecast 피처와 power curve feature를 비선형으로 활용하고 있어 teacher 피처가 중복/노이즈로 작용할 수 있음.
- 다음 시도는 단순 feature 추가보다, RF-only teacher 모델 활용, 모델별 feature selection, 또는 teacher feature를 calibration/stacking 단계에 쓰는 방향이 더 적합해 보임.

### `evaluate_pinn_tree_blend.py` — 최고 PINN + 기존 트리 앙상블 블렌드
현재 최고 PINN(SCADA teacher honest, HOD bias)과 기존 RF+LGBM+XGB 트리 앙상블을 2024 time-holdout에서 단순 가중평균으로 블렌드. 트리 쪽은 raw ensemble과 pooled isotonic calibrated ensemble을 둘 다 비교.

기준점:

| 모델 | 평균 점수 |
|---|---:|
| RF+LGBM+XGB raw | 0.5978 |
| RF+LGBM+XGB pooled isotonic | 0.6021 |
| PINN 단독 | 0.6189 |

블렌드 최고:

| tree variant | PINN weight | group_1 | group_2 | group_3 | 평균 |
|---|---:|---:|---:|---:|---:|
| pooled isotonic | 0.70 | 0.6357 | 0.6517 | 0.5862 | 0.6245 |

해석:
- 트리 단독보다 PINN이 확실히 강하지만, 두 모델의 오차가 완전히 같지는 않아 단순 평균만으로도 `0.6189 → 0.6245` 개선.
- 최적 weight가 PINN 70%, tree 30% 근처라 PINN을 메인 예측기로 두고 트리를 residual stabilizer처럼 쓰는 구조가 적합해 보임.
- 특히 group_2가 `0.6339 → 0.6517`로 크게 좋아져서, PINN의 물리/SCADA teacher 신호와 트리의 경험적 보정 신호가 잘 섞이는 케이스로 판단.
- 다음 단계는 제출용 `predict.py`에 PINN checkpoint prediction + calibrated tree prediction을 같은 weight로 결합하는 별도 submission path를 만드는 것.

### `evaluate_group3_transfer_blend.py` — group_3 전용 VESTAS/group_2 transfer 블렌드
이전 실험에서 group_3 own tree보다 group_2(VESTAS) 기반 transfer가 더 좋았던 기억을 다시 검증. group_3만 별도 후보군을 만들고, 현재 최고 UNISON PINN과 섞어서 weight를 탐색.

단독 후보:

| candidate | group_3 score | nmae | ficr |
|---|---:|---:|---:|
| UNISON PINN | 0.5886 | 0.1470 | 0.3243 |
| VESTAS PINN group_2 same-time proxy | 0.5828 | 0.1642 | 0.3299 |
| group_2 tree transfer on group_3 features | 0.5804 | 0.1440 | 0.3048 |
| group_2 tree same-time proxy | 0.5792 | 0.1452 | 0.3036 |
| group_3 own tree raw | 0.5587 | 0.1496 | 0.2669 |
| VESTAS backbone on group_3 weather | 0.5501 | 0.1488 | 0.2490 |

블렌드 최고:

| group_3 recipe | PINN weight | group_3 score | nmae | ficr |
|---|---:|---:|---:|---:|
| UNISON PINN + group_2 tree same-time proxy | 0.60 | 0.5994 | 0.1386 | 0.3373 |
| UNISON PINN + group_2 tree transfer on group_3 features | 0.65 | 0.5993 | 0.1389 | 0.3375 |
| UNISON PINN + VESTAS PINN group_2 same-time proxy | 0.60 | 0.5965 | 0.1488 | 0.3417 |

해석:
- group_3 own tree는 PINN에 섞으면 도움이 안 되고, 최적 weight가 PINN 1.0으로 돌아감.
- 반대로 group_2/VESTAS 기반 proxy는 단독으로 UNISON PINN보다 약하지만, 오차 구조가 달라서 섞으면 `0.5886 → 0.5994`로 상승.
- 이전 best group_1/group_2(`0.6357`, `0.6517`)를 유지하고 group_3만 이 조합으로 교체하면 평균은 약 `0.6289`.
- 현재 가장 강한 검증 조합은 group별 recipe를 다르게 쓰는 방향:
  - group_1/group_2: `PINN 0.70 + calibrated tree 0.30`
  - group_3: `UNISON PINN 0.60 + group_2 tree same-time proxy 0.40`

### `evaluate_group3_pinn_teacher_transfer.py` — group_3 PINN teacher 자체를 VESTAS/mixed로 교체
후단 proxy뿐 아니라, group_3 PINN 입력 teacher 자체도 UNISON 고정이 맞는지 확인. 기존 checkpoint는 덮어쓰지 않고, group_3 single-group PINN을 teacher recipe별로 새로 학습(`save=False`)해서 2024 holdout 비교.

단독 PINN 결과:

| physical backbone | teacher | group_3 score | nmae | ficr |
|---|---|---:|---:|---:|
| UNISON | 30% UNISON g3 + 70% VESTAS g2 teacher | 0.5961 | 0.1431 | 0.3353 |
| UNISON | VESTAS group1/group2 avg teacher | 0.5953 | 0.1413 | 0.3320 |
| VESTAS | VESTAS group1/group2 avg teacher | 0.5949 | 0.1423 | 0.3320 |
| UNISON | VESTAS group2 teacher | 0.5933 | 0.1430 | 0.3295 |
| UNISON | UNISON group3 teacher | 0.5880 | 0.1471 | 0.3231 |

추가로 teacher-transfer PINN과 group_2 tree proxy를 다시 블렌드:

| group_3 recipe | PINN weight | group_3 score | nmae | ficr |
|---|---:|---:|---:|---:|
| UNISON PINN(30% UNISON g3 + 70% VESTAS g2 teacher) + group_2 tree transfer on group_3 features | 0.70 | 0.6024 | 0.1382 | 0.3430 |
| UNISON PINN(30% UNISON g3 + 70% VESTAS g2 teacher) + group_2 tree same-time proxy | 0.70 | 0.6022 | 0.1385 | 0.3428 |
| UNISON PINN(VESTAS g2 teacher) + group_2 tree transfer on group_3 features | 0.70 | 0.6019 | 0.1382 | 0.3420 |
| VESTAS PINN(VESTAS g1/g2 avg teacher) + group_2 tree transfer on group_3 features | 0.70 | 0.6018 | 0.1377 | 0.3412 |

해석:
- group_3 PINN 자체도 UNISON teacher만 쓰는 것보다 VESTAS/group2 teacher를 많이 섞는 쪽이 더 좋음.
- 물리 backbone은 VESTAS로 완전히 바꾸는 것보다 UNISON backbone을 유지하고 teacher만 VESTAS 쪽으로 당기는 구성이 가장 좋음.
- group_2 tree transfer를 후단에 다시 섞으면 group_3가 `0.5994 → 0.6024`로 추가 상승.
- 이전 best group_1/group_2(`0.6357`, `0.6517`)를 유지하면 평균은 약 `0.6299`.
- 현재 최고 검증 recipe:
  - group_1/group_2: `PINN 0.70 + calibrated tree 0.30`
  - group_3: `UNISON-physics PINN(mixed teacher) 0.70 + group_2 tree transfer 0.30`

### `evaluate_final_tree_ensemble.py` — 현재 최고 recipe에 tree계열 추가 앙상블
현재 최고 recipe를 anchor로 두고, 여기에 tree계열 예측을 한 번 더 섞었을 때 개선되는지 확인.

기준 anchor:
- group_1/group_2: `PINN 0.70 + calibrated tree 0.30`
- group_3: `UNISON-physics PINN(mixed teacher) 0.70 + group_2 tree transfer 0.30`

결과:

| experiment | group_1 | group_2 | group_3 | 평균 |
|---|---:|---:|---:|---:|
| anchor | 0.6357 | 0.6517 | 0.6024 | 0.6299 |
| 공통 global extra raw/calibrated tree | - | - | - | 0.6299 이하 |
| group_2만 raw tree 15% 추가 | 0.6357 | 0.6530 | 0.6024 | 0.6304 |
| group_3만 own calibrated tree 10% 추가 | 0.6357 | 0.6517 | 0.6034 | 0.6303 |
| group_2 raw tree 15% + group_3 own calibrated tree 10% | 0.6357 | 0.6530 | 0.6034 | 0.6307 |

최종 후보 recipe:

```text
group_1 = 0.70 * PINN + 0.30 * calibrated tree
group_2 = 0.85 * (0.70 * PINN + 0.30 * calibrated tree) + 0.15 * raw tree
group_3 = 0.90 * (0.70 * mixed-teacher PINN + 0.30 * group2-transfer tree)
        + 0.10 * own calibrated tree
```

해석:
- tree를 모든 group에 공통 weight로 더 섞는 것은 group_1 손해 때문에 평균 개선이 없음.
- 다만 group_2와 group_3는 현재 recipe 이후에도 tree residual이 조금 남아 있어, group별 소량 추가 앙상블이 이득.
- 개선폭은 작지만 목표였던 `0.63`선을 검증상 넘김(`0.6307`).

### `evaluate_multi_year_generalization.py` — 2024 단일 holdout 과적합 점검
LB에서 final aggressive ensemble이 기대보다 낮고, PINN-only도 `0.6038`에 그치면서 2024 단일 holdout에 과적합됐을 가능성이 커짐. 그래서 2023/2024 yearly holdout과 2024 quarter block holdout을 추가해 검증판을 넓힘.

검증 구성:
- `year_2023`: 2022 train → 2023 validation. group_1/2만 평가(group_3은 2022 label 부족).
- `year_2024`: 2022~2023 train → 2024 validation. group_1/2/3 평가.
- `q2024_1~4`: quarter별 tree-only block validation. train은 각 quarter 이전 기간.

Yearly mean 결과:

| candidate | mean | worst fold | best fold |
|---|---:|---:|---:|
| PINN mixed group_3 teacher | 0.6136 | 0.6058 | 0.6214 |
| PINN standard | 0.6125 | 0.6058 | 0.6192 |
| calibrated tree + PINN 50% | 0.6112 | 0.5973 | 0.6251 |
| calibrated tree + PINN 30% | 0.6027 | 0.5886 | 0.6168 |
| tree calibrated | 0.5881 | 0.5755 | 0.6006 |
| tree raw | 0.5847 | 0.5726 | 0.5969 |

Yearly group breakdown:

| fold | group | tree calibrated | PINN standard | PINN mixed | tree+PINN 50% |
|---|---|---:|---:|---:|---:|
| 2023 | group_1 | 0.5475 | 0.5858 | 0.5858 | 0.5733 |
| 2023 | group_2 | 0.6036 | 0.6259 | 0.6259 | 0.6213 |
| 2024 | group_1 | 0.6105 | 0.6333 | 0.6333 | 0.6310 |
| 2024 | group_2 | 0.6371 | 0.6369 | 0.6369 | 0.6539 |
| 2024 | group_3 | 0.5542 | 0.5875 | 0.5942 | 0.5903 |

2024 quarter tree-only:

| candidate | mean | worst quarter | best quarter |
|---|---:|---:|---:|
| tree calibrated | 0.5992 | 0.5874 | 0.6164 |
| tree raw | 0.5957 | 0.5728 | 0.6135 |

해석:
- train 내부 multi-year 기준으로는 PINN이 tree보다 훨씬 강하고, 2023/2024 모두 방향은 일관됨.
- 하지만 실제 2025 LB에서 PINN-only가 `0.6038`이므로, 2022~2024 내부 검증만으로도 2025 drift를 완전히 잡지는 못함.
- 즉 문제는 단순히 "2024만 봐서"뿐 아니라, **2025 forecast/site relationship 자체가 2022~2024와 다르게 움직이는 drift**도 있는 것으로 보임.
- 다음 선택 기준은 2024 최고점이 아니라 `multi-year 평균`, `worst fold`, 실제 LB feedback을 함께 보는 쪽이어야 함.
- 당장 제출 후보는 aggressive recipe보다 보수적인 `tree baseline`, `PINN-only`, `tree/PINN 50% 내외`를 LB로 직접 비교하며 anchor를 다시 잡는 방향.

### `evaluate_scada_teacher_effective_targets.py`, `evaluate_pinn_effective_wind_teacher.py` — PINN effective wind target 재구성
PINN은 입력 풍속 `v`에 매우 민감하므로, 기존 `scada_ws_mean` 대신 발전량 물리에 더 가까운 effective wind target을 시도.

추가 teacher feature:
- lead hour (`forecast_kst_dtm - data_available_kst_dtm`)
- LDAPS grid speed distribution: mean/std/min/max/p75
- GFS grid speed distribution: mean/std/min/max/p75
- gust grid statistics

추가 teacher target:
- `scada_ws_cubic = mean(ws^3)^(1/3)`
- `scada_ws_p75`, `scada_ws_p90`, `scada_ws_max`, `scada_ws_ramp`

Teacher 진단 결과:

| fold | target | base R2 | extended R2 |
|---|---|---:|---:|
| 2023 | mean | 0.7548 | 0.7682 |
| 2023 | p90 | 0.7737 | 0.7857 |
| 2023 | cubic | - | 0.7775 |
| 2024 | mean | 0.7988 | 0.8093 |
| 2024 | p90 | 0.8091 | 0.8194 |
| 2024 | cubic | - | 0.8148 |

PINN 2024 holdout 결과:

| variant | group_1 | group_2 | group_3 | 평균 |
|---|---:|---:|---:|---:|
| p90 effective wind | 0.6212 | 0.6421 | 0.5897 | 0.6177 |
| cubic effective wind | 0.6305 | 0.6374 | 0.5905 | 0.6195 |
| 50% cubic + 50% p90 | 0.6271 | 0.6426 | 0.5863 | 0.6187 |

group_3 effective teacher + VESTAS prior:

| variant | group_3 score |
|---|---:|
| UNISON cubic only | 0.5893 |
| 30% UNISON + 70% VESTAS cubic | 0.5976 |
| 50% UNISON + 50% VESTAS cubic | 0.5965 |
| 30% UNISON + 70% VESTAS p90 | 0.5977 |

해석:
- Extended teacher feature는 SCADA target 예측 R2를 안정적으로 올림.
- PINN에는 `cubic = mean(ws^3)^(1/3)`가 가장 자연스럽고, group_1을 크게 개선.
- group_3는 여전히 VESTAS/group_2 prior를 섞어야 강함.
- 다만 실제 LB에서 standard PINN-only가 낮았기 때문에, effective PINN도 단독 제출보다는 tree baseline과 보수적으로 blend해서 확인하는 것이 안전.

생성한 제출 후보:
- `results/submission_tree_only.csv`: 기존 tree-only isotonic anchor
- `results/submission_effective_pinn.csv`: effective PINN-only
- `results/submission_tree80_effective20.csv`
- `results/submission_tree70_effective30.csv`
- `results/submission_tree50_effective50.csv`

추가로 더 보수적인 후보도 생성:
- `results/submission_tree95_effective05.csv`
- `results/submission_tree90_effective10.csv`

현재 `results/submission.csv`는 LB-safe 성격을 우선해 `tree90 + effective PINN10`으로 설정해 둠. tree-only anchor는 `results/submission_tree_only.csv`에 보관.

### `evaluate_effective_tree_time_holdout.py` — effective teacher/grid feature를 tree에 직접 추가
PINN에서 먹힌 effective wind teacher와 grid distribution을 tree feature로 직접 넣어봄. 후보:
- `effective_teacher`: 기존 tree feature + predicted effective wind targets
- `effective_teacher_grid`: 위 + selected lead/grid distribution stats

2024 time-holdout 결과:

| feature set | model | group_1 | group_2 | group_3 | 평균 |
|---|---|---:|---:|---:|---:|
| effective_teacher | RF | 0.5973 | 0.6367 | 0.5519 | 0.5953 |
| effective_teacher | LGBM | 0.6062 | 0.6295 | 0.5514 | 0.5957 |
| effective_teacher | ensemble | 0.6037 | 0.6314 | 0.5481 | 0.5944 |
| effective_teacher_grid | RF | 0.5950 | 0.6346 | 0.5557 | 0.5951 |
| effective_teacher_grid | LGBM | 0.6108 | 0.6300 | 0.5535 | 0.5981 |
| effective_teacher_grid | ensemble | 0.6048 | 0.6325 | 0.5540 | 0.5971 |

해석:
- effective teacher/grid feature는 일부 group/model에서는 개선되지만, 기존 tree baseline/pooled calibration을 확실히 넘지는 못함.
- 특히 group_3는 여전히 약하고, tree는 feature 추가보다 보정/앙상블이 더 중요해 보임.
- effective wind는 tree 직접 feature보다는 PINN 입력 또는 tree/PINN 소량 blend로 쓰는 쪽이 더 타당.

### `evaluate_tree_feature_blocks_fast.py`, `predict_tree_feature_block.py` — raw wind grid/lead block 재검토
기존 tree feature는 LDAPS 평균, GFS nearest grid 중심이라 원천 예보장의 공간분포/lead-time 정보가 사라진다. 그래서 PINN teacher에서 쓰던 확장 feature 중 일부를 tree에도 직접 붙여봄.

후보 block:
- `lead`: forecast lead hour
- `ldaps`: LDAPS wind grid mean/std/min/max/p75
- `gfs`: GFS wind grid mean/std/min/max/p75
- `lead_ldaps_gfs`: 위 전체

빠른 LGBM-only 결과:

| fold type | baseline | best block | best score |
|---|---:|---|---:|
| 2024 yearly | 0.5953 | `lead_ldaps_gfs` | 0.5976 |
| 2024 quarter mean | 0.5928 | `lead_ldaps_gfs` | 0.5964 |

하지만 제출 구조와 같은 RF+LGBM+XGB + pooled isotonic으로 다시 검증하면:

| feature block | raw | pooled isotonic |
|---|---:|---:|
| baseline | 0.5978 | 0.6021 |
| `lead_ldaps_gfs` | 0.5961 | 0.6009 |

해석:
- raw grid/lead 정보는 LGBM 단독에는 도움이 되지만, RF/XGB까지 포함한 현재 all3 ensemble에서는 noise/중복 신호가 더 커짐.
- 후보 제출 파일은 생성했지만(`submission_tree_feature_block.csv`, `submission_tree75_feature25.csv`, `submission_tree50_feature50.csv`), 메인으로 승격하지 않음.

### `evaluate_tree_meteo_feature_blocks_fast.py`, `evaluate_tree_meteo_submission_style.py`, `predict_tree_meteo.py` — 비바람 meteo feature 추가
기존 tree pipeline은 wind feature 위주였는데, 2025 drift를 생각하면 공기밀도/기압/온습도/경계층/복사/구름/강수 같은 기상 상태가 발전량과 forecast bias에 영향을 줄 수 있다. LDAPS는 16격자 평균, GFS는 farm nearest grid를 사용해 non-wind meteo feature block을 만들었다.

후보 block:
- `thermo`: 온도, 이슬점/습도, 비습, 지상기압, 해면기압, 상층 온도/습도, BLH
- `radiation_cloud`: 단파/장파 복사, 구름량, 강수/적설
- `all_meteo`: wind raw를 제외한 meteo 전체
- `lead_all_meteo`: `all_meteo` + lead hour

빠른 LGBM-only 결과:

| fold type | baseline | `all_meteo` | gain |
|---|---:|---:|---:|
| 2024 yearly | 0.5953 | 0.6033 | +0.0080 |
| 2024 quarter mean | 0.5928 | 0.6014 | +0.0086 |

제출 구조와 같은 RF+LGBM+XGB + pooled isotonic 결과:

| feature block | raw | pooled isotonic |
|---|---:|---:|
| baseline | 0.5978 | 0.6021 |
| `all_meteo` | 0.5997 | 0.6079 |

해석:
- 이번에는 all3 ensemble과 pooled isotonic에서도 개선이 유지됨.
- NMAE도 좋아지고(`0.1351 -> 0.1326`), FICR도 좋아짐(`0.3392 -> 0.3484`)이라 단순 평균 오차만 줄인 것이 아님.
- 현재 `results/submission.csv`는 이 `all_meteo` tree 후보로 갱신.
- 백업 후보로 effective PINN을 소량 섞은 `submission_tree_meteo95_effective05.csv`, `submission_tree_meteo90_effective10.csv`도 생성.

### `evaluate_tree_meteo_multi_year.py` — meteo tree 다년 일반화 확인
`all_meteo` 개선이 2024 단일 holdout 착시인지 확인하려고 2023/2024 yearly와 2024 quarter split에서 기존 tree baseline과 다시 비교했다. 구조는 제출식과 맞춘 RF+LGBM+XGB + pooled isotonic.

결과:

| fold type | baseline pooled | `all_meteo` pooled | gain |
|---|---:|---:|---:|
| yearly mean(2023, 2024) | 0.5881 | 0.5933 | +0.0053 |
| yearly worst | 0.5755 | 0.5822 | +0.0067 |
| quarter mean(2024) | 0.5992 | 0.6051 | +0.0059 |
| quarter worst | 0.5874 | 0.5877 | +0.0003 |

해석:
- 2023/2024 양쪽 yearly에서 모두 개선되고, quarter 평균도 개선되어 `all_meteo`는 제출 후보로 유지할 근거가 충분함.
- 다만 worst quarter 개선은 작아서, 실제 LB에는 pure meteo tree를 우선 제출하고 PINN blend는 작은 weight 후보로만 보는 전략이 안전.

### `evaluate_scada_teacher_meteo_targets.py`, `evaluate_pinn_meteo_wind_teacher.py` — meteo feature를 PINN teacher에 추가
tree에서 meteo feature가 먹혔으므로, SCADA wind teacher에도 같은 non-wind meteo feature를 붙였다. 먼저 SCADA target R2를 확인하고, 이후 PINN holdout까지 연결했다.

SCADA teacher mean R2:

| fold | target | extended | extended + meteo |
|---|---|---:|---:|
| 2023 | mean | 0.7682 | 0.7719 |
| 2023 | cubic | 0.7775 | 0.7792 |
| 2023 | p90 | 0.7857 | 0.7880 |
| 2024 | mean | 0.8093 | 0.8135 |
| 2024 | cubic | 0.8148 | 0.8193 |
| 2024 | p90 | 0.8194 | 0.8238 |

PINN 2024 holdout:

| variant | group_1 | group_2 | group_3 | 평균 |
|---|---:|---:|---:|---:|
| meteo cubic | 0.6268 | 0.6363 | 0.5971 | 0.6201 |
| meteo p90 | 0.6165 | 0.6448 | 0.5958 | 0.6190 |
| meteo 50% cubic + 50% p90 | 0.6237 | 0.6434 | 0.5942 | 0.6205 |

해석:
- meteo는 SCADA teacher target 예측 R2를 일관되게 올림.
- PINN 전체 평균 개선은 작지만, group_3가 기존 effective teacher보다 좋아지는 쪽이 의미 있음.
- group_1은 기존 non-meteo cubic effective가 더 좋았던 구간도 있어, meteo PINN 단독을 메인 제출로 쓰기보다는 tree 보조 후보로 유지.

### `evaluate_group3_meteo_teacher_mix.py`, `predict_pinn_meteo_only.py` — group3 meteo teacher + VESTAS prior
group3는 계속 VESTAS/group2 prior를 섞을 때 좋아졌으므로, meteo teacher에서도 같은 구조를 검증했다.

group3 결과:

| variant | group_3 score |
|---|---:|
| meteo UNISON cubic | 0.5968 |
| meteo 30% UNISON + 70% VESTAS cubic | 0.6007 |
| meteo 50% UNISON + 50% VESTAS cubic | 0.5996 |
| meteo 30% UNISON + 70% VESTAS p90 | 0.6015 |

제출 후보:
- `results/submission_pinn_meteo.csv`: group1 meteo cubic, group2 meteo p90, group3 meteo 30% UNISON + 70% VESTAS p90
- `results/submission_tree_meteo95_pinn_meteo05.csv`
- `results/submission_tree_meteo90_pinn_meteo10.csv`

현재 판단:
- 실제 LB에서 PINN-only 계열이 내부 검증보다 낮게 나왔던 점을 감안하면, 메인 `results/submission.csv`는 pure `all_meteo` tree로 유지.
- PINN meteo blend는 후보로만 보관하고, 제출 여유가 있으면 5% blend부터 확인.

### `evaluate_meteo_tree_pinn_blend.py` — meteo tree + meteo PINN blend weight sweep
`all_meteo` tree와 `PINN meteo`를 같은 2024 holdout 시간축에서 직접 섞어 weight를 탐색했다. tree는 RF+LGBM+XGB + pooled isotonic, PINN은 group1 meteo cubic / group2 meteo p90 / group3 meteo 30% UNISON + 70% VESTAS p90.

2024 holdout mean:

| candidate | mean score |
|---|---:|
| tree meteo | 0.6045 |
| PINN meteo | 0.6223 |
| global 50% PINN + 50% tree | 0.6242 |
| per-group best weights | 0.6270 |

Per-group best weights:

| group | PINN weight | score |
|---|---:|---:|
| group_1 | 0.55 | 0.6293 |
| group_2 | 0.55 | 0.6551 |
| group_3 | 0.70 | 0.5967 |

생성한 강한 후보:
- `results/submission_tree_meteo50_pinn_meteo50.csv`
- `results/submission_tree_meteo_pinn_meteo_holdout_best.csv`

해석:
- 내부 2024 기준으로는 PINN weight를 꽤 크게 주는 쪽이 best.
- 하지만 이전 실제 LB에서 PINN-only/공격적 blend가 내부 검증 대비 크게 떨어진 전례가 있어, `submission.csv`는 아직 pure `all_meteo` tree로 유지한다.
- 제출 횟수 여유가 있으면 순서는 `pure all_meteo tree` → `tree95+pinn05` → `tree50+pinn50` 또는 `holdout_best`로 보는 것이 합리적.

### `evaluate_meteo_tree_pinn_multi_year_blend.py` — meteo tree/PINN blend 다년 검증
위 blend sweep이 2024에만 맞은 것인지 확인하기 위해 2023/2024 yearly fold로 다시 평가했다. 2023은 group3 train label이 부족하므로 group1/2만 평균에 들어간다.

결과:

| candidate | 2023 | 2024 | mean | worst |
|---|---:|---:|---:|---:|
| pure meteo tree | 0.5822 | 0.6045 | 0.5933 | 0.5822 |
| global PINN 50% | 0.6022 | 0.6255 | 0.6139 | 0.6022 |
| 2024 holdout-best weights | 0.6034 | 0.6283 | 0.6158 | 0.6034 |
| pure PINN meteo | 0.6102 | 0.6254 | 0.6178 | 0.6102 |

해석:
- 다년 내부 검증에서는 `PINN meteo`가 평균/worst 모두 가장 좋다.
- 2024에서 최적이던 group별 holdout-best weight는 2023 worst가 PINN 단독보다 낮아, 다년 안정성은 PINN 단독 쪽이 더 낫다.
- 다만 실제 LB 피드백에서 PINN-only 계열이 내부 검증보다 낮게 나온 적이 있으므로, 운영상 메인 제출 파일은 아직 `all_meteo tree`로 유지하되, 새 후보 우선순위는 조정:
  1. `results/submission.csv` (`all_meteo tree`, 안전 anchor)
  2. `results/submission_pinn_meteo.csv` (multi-year 내부 best)
  3. `results/submission_tree_meteo50_pinn_meteo50.csv`
  4. `results/submission_tree_meteo_pinn_meteo_holdout_best.csv`

### `evaluate_scada_teacher_model_variants.py`, `evaluate_pinn_meteo_teacher_model_blend.py` — SCADA teacher 모델 앙상블
기존 meteo PINN teacher는 RandomForest 기반이었다. SCADA target 자체를 더 잘 예측하는 모델이 있는지 RF, ExtraTrees, HistGradientBoosting, 그리고 평균 앙상블을 비교했다.

SCADA target 평균 R2:

| fold | RF | ExtraTrees | HistGBR | RF+ExtraTrees | RF+HistGBR |
|---|---:|---:|---:|---:|---:|
| 2023 | 0.7817 | 0.7845 | 0.7811 | 0.7852 | 0.7857 |
| 2024 | 0.8203 | 0.8213 | 0.8202 | 0.8224 | 0.8240 |

가장 좋은 `RF+HistGBR` 평균 teacher를 PINN에 연결한 2024 holdout:

| teacher | group_1 | group_2 | group_3 | 평균 |
|---|---:|---:|---:|---:|
| RF | 0.6265 | 0.6464 | 0.5999 | 0.6243 |
| RF+HistGBR avg | 0.6300 | 0.6448 | 0.6015 | 0.6254 |

해석:
- teacher target R2 개선이 PINN 점수에도 작게 이어짐(+0.0012).
- group2는 살짝 내려가지만 group1/group3가 개선된다.
- 개선폭은 작지만 0.65 근처로 가는 데이터 방향에서는 유효한 조각.

생성한 제출 후보:
- `results/submission_pinn_meteo_teacher_blend.csv`
- `results/submission_tree_meteo50_pinn_meteo_teacher_blend50.csv`
- `results/submission_tree_meteo_pinn_meteo_teacher_blend_holdout_best.csv`

현재 제출 우선순위:
1. `results/submission.csv` (`all_meteo tree`, 안전 anchor)
2. `results/submission_pinn_meteo.csv` (multi-year 내부 best, RF teacher)
3. `results/submission_pinn_meteo_teacher_blend.csv` (RF+HistGBR teacher)
4. `results/submission_tree_meteo50_pinn_meteo50.csv`
5. `results/submission_tree_meteo50_pinn_meteo_teacher_blend50.csv`

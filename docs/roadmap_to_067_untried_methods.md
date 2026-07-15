# 0.67 Roadmap: Untried Methods Only

작성일: 2026-07-15

## 0. 목적

현재 public 최고는 `0.63999`이고 목표는 `0.67`이다. 필요한 차이는 약 `+0.030`이다.

현재 최고 제출:

```text
results/submission_jointmix_p50_t5_c45_pb50_cb25_v1.csv

PINN 0.50 + TREE 0.05 + TCN 0.45
PINN floor 0.20 * capacity
final floor 0.10 * capacity
```

공개 리더보드에서 보이는 차이는 대략 다음과 같다.

| 지표 | 현재 | 선두권 | 차이 |
|---|---:|---:|---:|
| 총점 | 0.63999 | 0.66912 | 0.02913 |
| 1-nMAE | 0.86772 | 0.88483 | 0.01711 |
| FiCR | 0.41217 | 0.45342 | 0.04125 |

따라서 `floor`, 작은 blend 조정, 단일 하이퍼파라미터 탐색만으로는 목표에 도달하기 어렵다. 이 문서는 이미 수행한 실험을 후보에서 제외하고, **입력 정보량을 늘리는 방법부터 문제 정의와 모델 구조를 바꾸는 방법까지** 실제 우선순위를 정한다.

예상 향상 폭은 보장이 아니라 실험 우선순위를 위한 가설 범위다. 모든 승격 판단은 동일한 OOF 구성과 hard Score로 한다.

---

## 1. 먼저 고정할 실험 계약

새 아이디어보다 먼저 이것을 고정하지 않으면 OOF와 public의 관계가 다시 깨진다.

### 1.1 OOF 학습

각 validation year `y`에 대해:

1. 나머지 연도로 학습한다.
2. `y`의 실제 hard Score를 매 epoch 계산한다.
3. `y`에서 최고 Score인 checkpoint를 그 fold의 모델로 저장한다.
4. 해당 checkpoint가 만든 `y` 예측을 OOF로 저장한다.
5. 세 outer-year 예측을 이어 붙인 뒤 group별 metric을 한 번 계산하고 세 group을 동일 가중한다.
6. 연도별/그룹별 점수는 안정성 진단으로 같이 남긴다.

후보끼리 비교할 때는 seed, fold, row mask, capacity, clipping, floor를 모두 동일하게 둔다. 결측 때문에 후보별 평가 행이 달라지면 공통 행 점수를 별도로 반드시 계산한다.

### 1.2 Test 추론

OOF를 만든 fold checkpoint가 그대로 test를 예측해야 한다.

```text
fold-2022 checkpoint -> test prediction A
fold-2023 checkpoint -> test prediction B
fold-2024 checkpoint -> test prediction C
fixed OOF aggregation(A, B, C) -> final test prediction
```

금지:

- OOF에서 찾은 epoch를 full-data 재학습에 그대로 넣기
- test용으로 별도 모델을 다시 학습하고 OOF 모델과 같다고 간주하기
- public 결과를 보고 fold weight, floor, blend를 다시 선택하기
- candidate마다 다른 결측 행으로 점수를 비교하기

### 1.3 승격 기준

| 후보 종류 | 최소 승격 기준 |
|---|---|
| 값싼 feature/data 후보 | pooled OOF `+0.002` 이상, 다수 group-year에서 악화 없음 |
| 새 단일 모델 | standalone `+0.005` 또는 기존 최고와 blend `+0.003` 이상 |
| 큰 구조 변경 | 기존 최고 대비 `+0.005` 이상 또는 명확한 새로운 오류 상보성 |
| 제출 후보 | OOF→test 학습 경로가 완전히 동일하고, ablation으로 이득 원인이 확인됨 |

목표가 `+0.030`이므로 첫 두 단계에서 합계 `+0.010`도 만들지 못하면 모델 크기만 키우지 않고 입력/타깃 가정을 다시 점검한다.

---

## 2. 지금까지 확인된 병목

이미 수행한 진단을 새 아이디어처럼 다시 실험하지 않는다. 다만 방향을 정하는 근거로 사용한다.

1. 같은 시각 실제 SCADA wind를 outer-train power curve에 넣은 oracle은 약 `0.834~0.839`였다.
2. 실제 total + predicted share oracle은 `0.791575`였다.
3. predicted total + actual share oracle은 `0.664789`였다.
4. hard branch oracle은 약 `0.745`였지만 학습한 gate는 oracle을 재현하지 못했다.
5. 결론은 **그룹 간 share보다 전체 발전 가능한 바람과 total power를 복원하는 과정이 우선 병목**이라는 것이다.

현재 모델의 체급을 올리는 가장 유력한 두 축은 다음이다.

- 기존 deterministic LDAPS/GFS에 없는 예보 정보 추가
- `NWP -> power`를 바로 풀지 않고 `NWP -> 발전에 유효한 wind/distribution -> power decision`으로 문제 재정의

---

## 3. 이미 해본 것: 활성 후보에서 제외

아래는 `docs/exp_logs.md`와 현재 코드에서 확인된 항목이다. 동일한 형태로 다시 제안하지 않는다.

| 범주 | 이미 수행한 내용 |
|---|---|
| 기본 전처리 | Standard/MinMax/MaxAbs scaling, 전체 변수 이상치·결측·중복·강수 단위·RH·zero-wind 점검 |
| Tree | LGBM, XGBoost, ExtraTrees, quantile LGBM, multi-target ExtraTrees, custom ScoreBoost, leaf decoder |
| 시간 모델 | TCN W24/W72/W168, issue24/full context, h128 L3, GRU, DLinear, group/per-turbine TCN |
| 공간 모델 | raw full-grid LGBM/ExtraTrees, spatial CNN, flattened MLP, static/dynamic/teacher/direct GNN, spatio-temporal GNN, NWP-to-turbine attention |
| NWP 조합 | LDAPS/GFS sister experts, 고정·lead·disagreement gate, optimal grid, 방향 조건 grid, upwind interpolation |
| 앙상블 | OLS/GLM, Ridge, residual LGBM, horizon interaction, 시간 smoothing, 다양한 blend/floor/lift |
| 타깃/loss | cube-root, group loss, pure 4/6/6~8% band, projected target, ordinal bins, Decision-Q, residual interval classifier |
| 글로벌 구조 | multi-target/global TCN, long-format global group model, total/share, common+residual decomposition |
| 물리/SCADA | PINN, hub-height/shear/density/terrain/wake feature, RF/LGBM teacher, E2E wind supervision, curtailment mask |
| 유사일/전문가 | Analog Ensemble, issue analog, output-bin MoE, weather-quantile MoE, horizon experts |

다음 역시 우선순위에서 제외한다.

- 모델 zoo를 무작정 늘리는 generic KNN/MLP/GNN/CNN
- 기존 예측에 작은 상수 보정을 반복하는 실험
- 동일 feature set에서 LGBM depth/leaf만 더 넓게 탐색하는 실험
- 설명되지 않는 gate를 oracle weight에 맞추는 실험
- same-time AWS/SCADA를 test 입력으로 사용하는 실험

---

## 4. 새 핵심 자산: GEFS

GEFS는 이번 문서의 제안 데이터가 아니라 **이미 확보된 입력 자산**이다.

### 4.1 확보 상태

| 항목 | 내용 |
|---|---|
| 원천 | NOAA GEFS, `s3://noaa-gefs-pds` |
| 기간 | 2021-12 ~ 2025-12, 월별 49개월 |
| cycle | 목표일 D 기준 D-2 18Z |
| lead | `f021, f024, ..., f048` |
| pressure/near-surface | 0.5도, 17x17, u/v at 10m/925/850/700 hPa |
| gust | 0.25도, 17x17, surface gust |
| ensemble summary | `geavg` mean + `gespr` spread |
| 증빙 | 원본 URL과 S3 Last-Modified가 월별 meta CSV에 저장됨 |
| 누락 메타 | 없음 |

현재 collector는 목표 행의 기준 시각보다 충분히 먼저 공개되는 보수적인 cycle만 사용한다. 제출 전에는 `data_available_kst_dtm`과 S3 `Last-Modified`를 행 단위로 다시 검증해 provenance manifest를 만든다.

현재 환경에는 parquet reader인 `pyarrow`가 없으므로 첫 실행 전에 의존성을 고정하거나, 한 번만 표준 중간 포맷으로 변환해야 한다.

### 4.2 왜 GEFS가 중요한가

LDAPS/GFS를 더 복잡하게 읽는 것과 달리 GEFS는 두 종류의 새 정보를 준다.

1. **독립 예보원**: 기존 두 예보가 같이 틀리는 상황에서 새로운 평균장이 오차를 줄일 수 있다.
2. **예보 불확실성**: spread가 크면 해당 시각의 점 예측을 과신하면 안 된다는 직접적인 신호가 된다.

NOAA GEFS는 21개 예보로 초기조건과 모델 불확실성을 표현하는 시스템이다. 현재 수집본은 개별 21개 member가 아니라 mean과 spread만 저장했다. 따라서 가장 싼 실험은 이미 가능하고, member별 분포 모양은 후속 수집 과제다.

---

## 5. 후보 전체 지도

| ID | 후보 | 변화 축 | 비용 | 가설 향상 | 우선순위 |
|---|---|---|---|---:|---:|
| A1 | GEFS mean-only 추가 | 새 정보 | 낮음 | `+0.003~0.012` | 1 |
| A2 | GEFS spread/disagreement 추가 | 불확실성 | 낮음 | `+0.002~0.008` | 2 |
| A3 | 연도 recency weighting | 분포 이동 | 낮음 | `0~+0.004` | 6 |
| A4 | CatBoost baseline | 새 tabular learner | 낮음~중간 | `+0.001~0.006` | 5 |
| B1 | AWS-supervised MOS | NWP 보정 타깃 | 중간 | `+0.003~0.015` | 3 |
| B2 | power-equivalent wind target | 문제 재정의 | 중간 | `+0.004~0.015` | 4 |
| B3 | monotone environmental power curve | 물리적 제약 | 중간 | `+0.002~0.008` | 7 |
| C1 | bounded conditional density | 분포 예측 | 중간 | `+0.002~0.010` | 8 |
| C2 | score-optimal density decoder | 의사결정 | 낮음 | `+0.001~0.008` | 9 |
| C3 | cross-source uncertainty fusion | 표현 학습 | 중간~높음 | `+0.003~0.012` | 10 |
| D1 | power+wind auxiliary multi-task | 표현 학습 | 높음 | `+0.003~0.012` | 11 |
| D2 | joint 24h spatial-temporal encoder | 구조 변경 | 높음 | `+0.004~0.015` | 12 |
| D3 | TFT-style multi-horizon model | 구조 변경 | 높음 | `0~+0.012` | 13 |
| E1 | GEFS individual members | 추가 수집 | 높음 | `+0.002~0.010` | 14 |
| E2 | lagged GEFS cycles | 추가 수집 | 중간 | `+0.002~0.008` | 15 |
| E3 | KIM G128 | 추가 예보원 | 높음 | `+0.002~0.010` | 16 |

가설 향상 폭은 독립적으로 더할 수 없다. A1/A2/B1/C1처럼 같은 정보를 이용하는 후보는 이득이 겹친다.

---

## 6. 단순하고 먼저 할 후보

### A1. GEFS mean-only 정보 이득

**직관**

동일한 날에 LDAPS와 GFS가 같이 틀리면 기존 모델은 아무리 커도 답을 찾을 수 없다. GEFS 평균장이 그 오류와 독립적이면 단순 feature 추가만으로 체급이 오른다.

**첫 실험에서는 spread를 넣지 않는다.** 새 예보 평균의 정보 이득과 불확실성 이득을 분리하기 위해서다.

**입력 생성**

1. GEFS valid UTC를 KST target hour로 변환한다.
2. 3시간 간격 u/v를 시간축에서 선형 보간한다.
3. 풍속과 풍향을 직접 보간하지 않고 u/v를 보간한 뒤 파생한다.
4. 발전소별 nearest point, 3x3 mean/std, 중심차분 gradient를 만든다.
5. 10m/925/850/700의 speed, direction sin/cos, vertical shear를 만든다.
6. gust mean과 `gust / max(ws10, eps)`를 만든다.

첫 feature set은 수십 개로 제한한다. 17x17 raw flatten은 첫 실험에 넣지 않는다.

**모델**

- cheap branch: 현재 strongest tabular pipeline에 GEFS compact feature만 추가
- temporal branch: 현재 strongest TCN의 시간별 weather channel에 GEFS compact feature만 추가

**Ablation**

```text
base
base + GEFS 10m mean
base + GEFS 10m/925/850/700 mean
base + GEFS vertical/gust features
```

**중단 기준**

- wind proxy MAE와 power Score 모두 개선이 없으면 raw-grid/큰 모델로 확장하지 않는다.
- GEFS가 특정 연도만 개선하면 cycle/model-version drift를 먼저 확인한다.

### A2. GEFS spread와 source disagreement

**직관**

spread는 예측값 자체가 아니라 “이 시각 예보를 얼마나 믿어야 하는가”를 알려준다. 기존 sister gate는 deterministic source disagreement만 썼지만 GEFS spread는 ensemble이 직접 표현한 불확실성이라 정보가 다르다.

**feature**

```text
spread(u), spread(v), approximate spread(speed)
gust spread
spread vertical profile
GEFS mean - GFS
GEFS mean - LDAPS
pairwise source vector disagreement
spread x lead
spread x wind regime
24h maximum/mean spread
```

주의: `sqrt(sprd_u^2 + sprd_v^2)`는 풍속 표준편차와 정확히 같지 않다. 첫 단계의 proxy로만 쓰고, member 수집 후에는 실제 member speed 분포로 교체한다.

**사용 순서**

1. 일반 feature로 넣어 점 예측이 좋아지는지 본다.
2. 예측 분산 head의 입력으로 넣는다.
3. 마지막에만 source reliability fusion에 사용한다.

처음부터 gate를 만들지 않는다. spread가 실제 OOF 절대오차·FiCR miss와 상관이 있는지 calibration plot부터 확인한다.

### A3. 연도 recency weighting

**직관**

NWP 버전, 터빈 상태, 운영 정책이 변했다면 오래된 해와 test 직전 해를 동일 가중하는 것이 틀릴 수 있다.

**실험**

각 fold에서 training year의 시간 거리만 이용한다.

```text
uniform
linear recency
exponential half-life 1 year
exponential half-life 2 years
```

validation year의 label 분포를 보고 weight를 정하면 안 된다. weight family는 세 outer fold pooled hard Score로 한 번 선택한다. 개선이 작더라도 GEFS/AWS MOS 학습에 공통으로 적용할 수 있다.

### A4. CatBoost baseline

CatBoost는 현재 로그에 없는 강한 tabular baseline이다. generic 모델 추가가 목적이 아니라, ordered boosting과 범주형 변수를 이용해 지금 LGBM이 잘 못 쓰는 구조를 확인하는 진단이다.

**범주형 후보**

```text
group_id, turbine_id, manufacturer, model
LDAPS grid id, GFS grid id, GEFS grid id
lead bucket, hour, month, wind-direction sector
```

**연속형 후보**

기존 compact weather + GEFS mean/spread + source disagreement만 넣는다. 첫 실험에서 raw full grid를 넣지 않는다.

**판정**

- CatBoost standalone이 낮아도 residual 상관이 TCN/PINN과 낮으면 blend 가치를 본다.
- 동일 feature에서 LGBM과 차이가 없으면 tabular learner가 아니라 입력/타깃 병목이라는 증거다.

---

## 7. 관측을 누수 없이 쓰는 후보

### B1. AWS-supervised dynamic MOS

**핵심 구분**

same-time AWS를 test feature로 쓰는 것은 누수다. 그러나 AWS를 **훈련 시 NWP 오차를 배우는 supervision target**으로만 쓰고, 추론 입력을 NWP로 제한하면 구조가 다르다.

고정 bias도 아니다. 시간, lead, 풍향, 고도, source disagreement, GEFS spread에 따라 달라지는 조건부 Model Output Statistics다.

**학습 문제**

```text
input : as-issued LDAPS/GFS/GEFS + station static + calendar + lead
target: AWS observed u/v
output: corrected u/v or NWP residual u/v
test input: NWP + static + calendar + lead only
```

풍속/풍향 대신 u/v residual을 학습한다. 관측 풍향 0/360 경계 문제를 피하기 위해서다.

**공간 연결**

1. 발전소 주변 AWS station을 거리·고도·방향으로 매핑한다.
2. station별 모델 하나보다 station embedding을 가진 global MOS를 우선한다.
3. corrected station wind를 발전소로 보간하거나, site head가 직접 발전소 corrected wind를 출력한다.
4. AWS가 없는 시간은 mask하고 loss에서 제외한다.

**OOF 규칙**

outer validation year의 AWS를 MOS 학습에 절대 넣지 않는다. 각 fold의 training years AWS만 사용한다. test에서는 2025 AWS를 읽지 않는다.

**진단 순서**

1. raw NWP -> AWS u/v MAE
2. MOS NWP -> AWS u/v MAE
3. raw/MOS wind와 SCADA wind·group power의 상관
4. corrected wind를 동일 power model에 넣은 hard Score

AWS MAE가 좋아져도 power Score가 나빠지면 station-level surface wind와 rotor-effective wind의 간극이 원인이다. 이때 B2로 넘어간다.

### B2. Power-equivalent wind target

**직관**

실제 발전량 `P`로부터 “이 발전량을 만들었어야 할 유효풍속” `w_eff`를 train-only power curve의 역함수로 구한다.

```text
w_eff = f_train^{-1}(P / capacity)
NWP -> w_eff -> monotone power curve -> P
```

이 target은 계측 풍속이 아니라 발전량을 설명하는 latent wind다. wake, 지형, rotor, 운영 상태가 합쳐진 결과를 하나의 물리적으로 해석 가능한 좌표로 바꾸는 시도다. 기존 cube-root target이나 measured-wind teacher와 다르다.

**구성**

1. fold training data만으로 group/manufacturer별 monotone power curve `f_train`을 적합한다.
2. training power를 역변환해 `w_eff` target을 만든다.
3. LDAPS/GFS/GEFS로 `w_eff`를 예측한다.
4. 예측한 `w_eff`를 같은 fold의 `f_train`에 넣어 power로 복원한다.
5. curve 밖 값은 물리 범위로 clipping한다.

**중요한 ablation**

```text
direct power
cube-root power (기수행 기준)
measured SCADA wind target (진단 기준)
power-equivalent wind target
```

**위험**

- curtailment/outage가 `w_eff`를 가짜 저풍속으로 만든다.
- 포화구간은 역함수가 불안정하다.

대응은 관측을 삭제하는 것이 아니라 curve inversion confidence를 weight로 주는 것이다. 중간 출력 구간을 강하게, 포화·정지 구간을 약하게 학습한다.

### B3. Monotone environmental power curve

기존 1D isotonic curve보다 한 단계만 확장한다.

```text
P / capacity = monotone_f(
    corrected effective wind,
    direction sector,
    air-density proxy,
    vertical shear,
    GEFS spread
)
```

wind에 대해서만 단조 증가 제약을 걸고 나머지 변수는 조건부 보정으로 둔다. 후보 구현은 monotone GAM, constrained spline, monotone boosting이다.

장점은 power curve의 기본 물리를 지키면서 방향·밀도·불확실성에 따른 곡선 이동을 설명할 수 있다는 점이다. PINN보다 단순하며 failure mode를 그릴 수 있다.

---

## 8. FiCR을 제대로 쓰는 후보

### C1. Bounded conditional density

직접 hard FiCR loss만 최적화하면 gradient가 희소하고 모델이 특정 구간으로 collapse하기 쉽다. 이미 pure-band와 Decision-Q에서 이 문제가 나타났다.

새 접근은 두 단계를 분리한다.

1. `p(y | x)`를 proper likelihood/CRPS로 학습한다.
2. 학습된 분포에서 competition Score 기대값이 최대인 단일 제출값을 고른다.

발전률은 `[0, 1]`에 묶이고 0 또는 포화 부근 질량이 있을 수 있으므로 첫 모델은 복잡한 flow보다 다음이 적합하다.

```text
zero/one-inflated Beta mixture
or logistic-normal mixture
```

출력:

```text
pi_zero, pi_continuous, pi_one
continuous distribution parameters
```

GEFS spread는 scale/mixture weight head에 직접 넣는다.

**평가**

- NLL/CRPS: 분포 calibration 확인
- interval coverage: 4%, 6%, 8% band 안 실제 포함률
- hard competition Score: 최종 승격 판단

### C2. Score-optimal density decoder

분포가 학습되면 제출값 `a`는 평균이나 중앙값으로 고정할 이유가 없다.

```text
a*(x) = argmax_a E[competition_score(a, Y) | x]
```

구현은 간단하다.

1. 예측 분포에서 sample 또는 CDF grid를 만든다.
2. capacity-normalized `a` 후보를 촘촘히 둔다.
3. repo의 canonical hard Score 함수를 각 후보에 적용한다.
4. 기대 Score가 가장 높은 `a`를 선택한다.

이 방식은 MSE와 FiCR을 한 loss에서 어정쩡하게 섞지 않는다. 분포는 정직하게 배우고, 대회 규칙은 마지막 의사결정에서 정확히 쓴다.

**Decision-Q와의 차이**

Decision-Q는 불연속 reward를 직접 학습했다. 여기서는 먼저 proper scoring rule로 연속 분포를 calibration한 뒤 score를 계산하므로 target collapse 위험이 작다.

### C3. Cross-source uncertainty fusion

고정 sister weight가 아니라 source별 encoder를 분리한다.

```text
LDAPS encoder --\
GFS encoder ----- uncertainty-conditioned fusion -> temporal head -> distribution
GEFS encoder ----/
```

fusion weight 입력:

```text
lead, hour, season
GEFS spread
source pair disagreement
vertical shear disagreement
gust regime
```

각 encoder는 자기 source의 시간축을 먼저 읽고 latent에서 합친다. raw feature를 처음부터 섞는 현재 방식보다 source 오류를 분리할 수 있다.

규제:

- source dropout: 학습 중 한 source를 가끔 가려 특정 source 독점을 방지
- entropy floor: gate가 항상 한 source만 선택하지 않게 초기 단계에서만 사용
- gate plot: lead/regime별 평균 weight를 반드시 저장

---

## 9. 구조를 바꾸는 후보

### D1. Power-primary, wind-auxiliary multi-task

teacher 예측을 feature로 넣지 않는다. 하나의 NWP encoder가 주 target인 group power와 보조 target인 AWS/SCADA wind를 동시에 학습한다.

```text
shared NWP encoder
  -> group power distribution head (primary)
  -> AWS u/v head (train-only auxiliary)
  -> SCADA/effective wind head (train-only auxiliary)
```

추론에는 power head만 사용한다. 보조 head는 encoder가 실제 풍장과 관계없는 shortcut을 배우지 못하게 하는 규제 역할이다.

loss:

```text
L = L_power_distribution
  + lambda_aws * masked_L_aws_uv
  + lambda_scada * masked_L_scada_or_effective_wind
```

`lambda`는 작게 시작하고, power Score가 좋아질 때만 유지한다. wind MAE 개선 자체가 승격 기준은 아니다.

### D2. Joint 24h spatial-temporal encoder

기존 spatial CNN은 시간별 공간장을 독립적으로 봤고, TCN은 공간장을 compact feature로 줄인 뒤 시간축을 봤다. 아직 남은 구조적 공백은 **공간과 24시간 이동을 동시에 convolution**하는 것이다.

입력:

```text
[batch, source, time=24, variable, y, x]
```

구조 후보:

```text
source-specific shallow CNN per time
-> ConvLSTM or factorized 3D convolution
-> source fusion
-> three group distribution heads
```

처음부터 큰 U-Net을 쓰지 않는다. 데이터가 3년뿐이므로 spatial kernel 3x3, time kernel 3, channel 16~32의 작은 모델부터 시작한다.

핵심 질문은 “풍장이 공간적으로 이동하는 모양이 compact grid feature보다 추가 정보를 주는가”다. A1에서 GEFS compact feature가 이미 강하면 그 다음에만 raw GEFS grid를 연다.

### D3. TFT-style multi-horizon model

TFT 계열은 현재 문제의 입력 구분과 잘 맞는다.

```text
static: group/turbine/site/manufacturer/capacity
known future: LDAPS/GFS/GEFS 24h forecast trajectory, hour, lead
observed training-only: SCADA/AWS auxiliary labels
target: 24h group power distribution
```

장점:

- variable selection으로 source/variable 중요도를 시간별로 선택
- multi-horizon attention으로 하루 전체 관계를 사용
- quantile/distribution head 확장이 쉬움

위험:

- 표본 수가 작아 TCN보다 쉽게 과적합
- attention이 설명 가능해 보여도 실제 안정성은 별도 검증 필요

따라서 GEFS와 conditional density에서 정보 이득을 먼저 확인한 뒤 진행한다. generic Transformer 교체 실험으로 하면 안 된다.

---

## 10. 추가 데이터 확장

### E1. GEFS individual members

현재 mean/spread만으로는 skew, multimodality, 극단 member를 알 수 없다. A1/A2가 유효하면 21개 member를 추가 수집한다.

파생:

```text
member wind-speed quantiles
member power-curve transformed distribution
member disagreement by direction
probability of high/low/ramp regimes
member-wise score-optimal decision
```

가장 직접적인 사용법은 각 member wind를 monotone power curve에 통과시켜 21개의 power scenario를 만든 뒤 C2 decoder로 제출값을 고르는 것이다. 신경망 없이도 가능한 단순하고 설명 가능한 구조다.

### E2. Lagged GEFS cycles

D-2 18Z 하나뿐 아니라 그보다 이른 12Z/06Z cycle도 기준 시각 전에 공개되어 있다. 여러 cycle의 forecast 변화는 run-to-run uncertainty다.

```text
latest safe mean
previous safe mean
cycle-to-cycle delta
cycle ensemble spread
```

단, cycle별 S3 Last-Modified가 모든 row의 `data_available_kst_dtm`보다 빠른지 증빙해야 한다. D-1 00Z처럼 경계에 가까운 cycle은 다운로드 가능 시각이 불안정하므로 사용하지 않는다.

### E3. KIM G128

KIM은 독립적인 KMA global NWP라는 장점이 있지만 현재 수집본이 완성되지 않았고 API quota/permission 문제가 있다. GEFS 이득이 확인된 뒤에만 수집 비용을 쓴다.

우선순위는 다음과 같다.

```text
GEFS mean/spread 활용 완료
-> individual GEFS / lagged cycle
-> KIM G128
-> 그 외 외부 NWP
```

외부 NWP는 “데이터가 많다”가 아니라 기존 source와 오류 상관이 낮을 때만 가치가 있다.

---

## 11. 실제 실행 순서

### 실행 환경 원칙

- 긴 OOF 학습은 Duck에서 실행한다.
- CPU 모델의 `n_jobs`는 기본 8로 제한하고 첫 fold 시간으로 전체 시간을 다시 계산한다.
- SSH는 실험 하나당 한 번 접속해 전처리와 fold 실행을 묶는다. 짧은 연결을 병렬로 반복하지 않는다.
- OOF prediction, checkpoint, score, diagnostics는 Git에 올리지 않는다.
- 사용자 승인을 받은 최종 submission만 `git add -f`로 추가한다.
- 아래 시간은 첫 구현 전 추정치다. Duck의 실제 첫 fold 측정 후 문서의 예상 시간을 갱신한다.

### 구현 파일 지도

| 단계 | 신규 파일 후보 | 책임 |
|---|---|---|
| GEFS 공통 | `utils/gefs_dataset.py` | parquet schema, 시간 정렬, 공간 crop/compact feature |
| GEFS audit | `experiments/audit_gefs_alignment.py` | coverage, publication time, valid-time, 분포 확인 |
| GEFS probe | `experiments/evaluate_gefs_compact_oof.py` | mean/spread ablation과 common-row OOF |
| CatBoost | `experiments/evaluate_catboost_gefs_oof.py` | categorical/continuous compact baseline |
| AWS MOS | `utils/aws_mos.py` | train-only station mapping과 u/v residual correction |
| MOS 평가 | `experiments/evaluate_aws_mos_oof.py` | NWP wind와 power downstream 평가 |
| Effective wind | `utils/power_equivalent_wind.py` | fold-local monotone curve와 역변환 |
| Effective wind 평가 | `experiments/evaluate_power_equivalent_wind_oof.py` | direct/cube-root/effective target 비교 |
| 분포 예측 | `models/bounded_power_distribution.py` | zero/one-inflated bounded density head |
| Score decoder | `utils/score_optimal_decoder.py` | canonical hard Score 기대값 최대화 |
| Source fusion | `models/cross_source_temporal.py` | LDAPS/GFS/GEFS encoder와 uncertainty fusion |
| 시공간 모델 | `models/spatiotemporal_weather.py` | compact 3D CNN/ConvLSTM |

기존 공통 metric과 dataset builder는 복제하지 않고 호출한다. 다만 현재 dirty worktree의 사용자 변경을 건드리거나 되돌리지 않는다.

### 예상 실행 비용

| 단계 | Duck 예상 시간 | 주 자원 | 비고 |
|---|---:|---|---|
| GEFS alignment/audit | 20~60분 | CPU/RAM | parquet 최초 로드와 cache 포함 |
| GEFS compact CatBoost/LGBM 3-fold | 1~4시간 | CPU 8 jobs | feature set별 |
| GEFS TCN 3-fold | 3~10시간 | GPU | 기존 TCN 대비 channel 증가만 |
| AWS MOS 3-fold | 1~5시간 | CPU | station 수와 model에 따라 변동 |
| Power-equivalent target 3-fold | 2~8시간 | CPU/GPU | downstream learner에 따라 변동 |
| Bounded density 3-fold | 3~10시간 | GPU | 작은 TCN head 기준 |
| Cross-source fusion 3-fold | 6~18시간 | GPU | source별 encoder |
| Joint spatial-temporal 3-fold | 8~30시간 | GPU | raw-grid I/O 포함 |

각 단계는 먼저 한 group·한 fold smoke test를 하고, schema와 Score가 맞을 때만 전체 3-fold로 확장한다. smoke score로 모델을 선택하지는 않는다.

### Phase 0. 재현성 잠금

산출물:

```text
baseline_oof_predictions
baseline_fold_checkpoints
baseline_test_fold_predictions
score_by_group_year
inference_manifest
```

완료 조건:

- 동일 코드 재실행 시 OOF와 test fold prediction이 재현됨
- full-data 재학습 경로가 없음
- current-best floor/blend가 고정됨

### Phase 1. GEFS 자체 가치 확인

1. `pyarrow`와 schema/version 고정
2. 시간·공간 alignment audit
3. GEFS mean-only compact feature 생성
4. CatBoost/LGBM cheap probe
5. strongest TCN에 동일 feature 추가
6. spread와 disagreement ablation

필수 결과표:

| variant | pooled Score | 1-nMAE | FiCR | g1 | g2 | g3 | worst year |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | | | | | | | |
| + GEFS mean | | | | | | | |
| + GEFS vertical/gust | | | | | | | |
| + GEFS spread | | | | | | | |

Go 조건: mean 또는 spread가 pooled `+0.002` 이상.

### Phase 2. 유효풍 복원

1. AWS MOS u/v correction
2. power-equivalent target
3. GEFS+AWS MOS
4. monotone environmental power curve

필수 비교:

```text
raw NWP wind proxy MAE
corrected wind proxy MAE
direct power Score
effective-wind power Score
```

Go 조건: 기존 최고 branch 대비 standalone `+0.005` 또는 blend `+0.003`.

### Phase 3. FiCR decision

1. zero/one-inflated bounded density
2. calibration 확인
3. canonical hard Score decoder
4. GEFS spread 유무 ablation

분포가 calibration되지 않으면 decoder 결과를 제출하지 않는다.

### Phase 4. 큰 구조

1. cross-source uncertainty fusion
2. power-primary multi-task
3. compact joint spatial-temporal encoder
4. 마지막에 TFT-style model

각 단계는 앞 단계 best input/target만 상속한다. architecture와 데이터/target을 동시에 바꾸지 않는다.

### Phase 5. 추가 수집

GEFS mean/spread가 유효할 때만 individual members와 lagged cycles를 수집한다. KIM은 그 다음이다.

---

## 12. 결과에 따른 의사결정

### GEFS mean이 바로 개선됨

입력 정보량이 병목이었다. spread, individual member, source fusion으로 확장한다. 모델 zoo는 뒤로 미룬다.

### GEFS wind proxy는 개선되는데 power Score가 그대로임

surface/free-atmosphere wind와 발전 유효풍의 차이가 병목이다. AWS MOS만 키우지 말고 power-equivalent target과 monotone environmental curve로 간다.

### GEFS spread가 error와 상관 있지만 점 예측은 안 좋아짐

정상적인 결과일 수 있다. spread를 mean model feature로 억지 사용하지 말고 conditional scale/density와 score decoder에 사용한다.

### AWS MOS는 좋아지는데 SCADA/power가 안 좋아짐

관측 위치·높이 차이 또는 rotor/wake가 병목이다. AWS를 final feature로 사용하지 않고 auxiliary supervision으로만 낮춘다.

### 모든 tabular 모델이 비슷함

learner가 아니라 표현이 병목이다. joint spatial-temporal encoder 또는 power-equivalent target으로 이동한다.

### density calibration은 좋은데 hard Score가 안 오름

FiCR 기대 이득이 1-nMAE 손실보다 작거나 decoder 구현이 잘못된 것이다. threshold별 기대 reward와 실제 OOF reward를 대조한다.

### 큰 모델만 train score가 오름

데이터 크기 대비 과적합이다. 모델 크기를 더 키우지 않고 source dropout, auxiliary supervision, fold-bagging으로 돌아간다.

---

## 13. 하지 않을 것

1. GEFS를 넣으면서 loss, architecture, floor까지 동시에 변경하지 않는다.
2. GEFS 17x17 raw grid부터 큰 CNN에 넣지 않는다.
3. AWS same-time 관측을 validation/test 입력으로 사용하지 않는다.
4. OOF best epoch를 full-data fixed epoch로 재학습하지 않는다.
5. `+0.000x` OOF 후보를 public으로 반복 확인하지 않는다.
6. 실패한 gate/OLS/Ridge/floor를 새로운 이름으로 반복하지 않는다.
7. 단일 aggregate 점수만 보고 group3 또는 특정 연도 붕괴를 숨기지 않는다.
8. 외부 데이터 publication timestamp와 라이선스 증빙 없이 제출하지 않는다.

---

## 14. 최종 우선순위 결론

지금 가장 가능성이 높은 경로는 다음이다.

```text
정확한 fold-bag baseline
-> GEFS mean-only
-> GEFS spread/disagreement
-> AWS-supervised MOS + power-equivalent wind
-> bounded conditional density
-> hard Score-optimal decoder
-> cross-source / spatial-temporal model
```

핵심은 GEFS를 단순히 feature 몇 개 추가하고 끝내지 않는 것이다.

- mean은 기존 deterministic NWP의 오차를 줄인다.
- spread는 예보를 얼마나 믿을지 알려준다.
- AWS는 train-only MOS supervision을 준다.
- power-equivalent wind는 발전량에 맞는 유효풍 좌표를 만든다.
- conditional density는 FiCR을 직접 최적화할 수 있는 정보를 만든다.
- score decoder가 마지막 단일 제출값을 고른다.

이 경로는 `모델을 무지성으로 늘리는 것`이 아니라 **새 정보 -> 물리적으로 맞는 latent target -> 확률 예측 -> 대회 의사결정** 순서로 문제의 체급을 올린다.

---

## 15. 참고 원문

### 데이터와 ensemble forecast

- NOAA, Global Ensemble Forecast System: https://www.ncei.noaa.gov/products/weather-climate-models/global-ensemble-forecast
- NOAA GEFS on AWS Open Data: https://registry.opendata.aws/noaa-gefs/

### NWP calibration / MOS

- Improved wind speed forecasts using numerical weather prediction and MOS: https://www.sciencedirect.com/science/article/pii/S0360544214007555
- Machine-learning wind-speed bias correction and MOS comparison: https://www.sciencedirect.com/science/article/pii/S1674283423000144
- NWP bias correction using SCADA and continuous learning: https://arxiv.org/abs/2402.13916

### 확률 풍력 예측

- NGBoost: https://proceedings.mlr.press/v119/duan20a.html
- Conditional normalizing flow for probabilistic wind power: https://arxiv.org/abs/2206.02433
- Ensemble mixture density neural network for wind power: https://www.sciencedirect.com/science/article/pii/S0960148115303645

### 구조와 물리

- ConvLSTM: https://papers.nips.cc/paper_files/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html
- Temporal Fusion Transformer: https://arxiv.org/abs/1912.09363
- Monotonic power-curve regression: https://www.sciencedirect.com/science/article/pii/S0960148119312479
- Rotor-equivalent wind speed and rotor-area cubic weighting: https://wes.copernicus.org/articles/10/1187/2025/wes-10-1187-2025.html
- CatBoost ordered boosting: https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html

### 풍력 대회 구조

- GEFCom2014 wind track winner, probabilistic gradient boosting: https://www.sciencedirect.com/science/article/pii/S0169207016000145
- GEFCom2014 competition summary: https://robjhyndman.com/papers/gefcom2014.pdf

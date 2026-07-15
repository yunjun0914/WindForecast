# Source Expert Pipeline Plan

Updated: 2026-07-16 KST  
Status: active implementation plan  
Owner: WindForecast team

Active branch:

```text
codex/source-expert-pipeline
```

이 문서는 LDAPS/GFS/GEFS를 제대로 사용하는 다음 핵심 실험의 실행 기준이다.
구현 중 판단이 흔들리면 최근 임시 실험보다 이 문서를 우선한다.

## 1. 문제 정의

현재 파이프라인은 좋은 데이터를 추가해도 정보가 잡음에 묻히기 쉬운 구조다.

- `raw_source` 경로도 실제 원본 전문가가 아니라 quota65 목록에서 반대 소스 이름만 제거한다.
- 기본 전처리는 LDAPS 16개 격자를 평균내고 GFS 9개 격자는 최근접 1개만 남긴다.
- 풍향을 `u/v` 구조로 끝까지 보존하지 않고 많은 통계 파생량으로 바꾼다.
- TREE `full_v2`는 약 510개 피처이며 정확한 중복 열과 의미가 잘못된 파생식이 있다.
- 현재 TCN 입력에도 중복 열과 일부 test 결측의 0 치환이 남아 있다.
- GEFS는 원본 공간장과 ensemble spread가 있지만 mean을 23개 compact 통계량으로 축약해 기존 입력에 추가했다.
- 결과적으로 새 예보원을 독립적으로 평가하지 못했고, 기존 모델의 평균화 성향만 강화했다.

따라서 다음 실험은 피처 수를 늘리는 실험이 아니다. 데이터원을 분리하고 각 데이터가 단독으로 설명하는 신호와 서로 다른 오차를 검증하는 실험이다.

## 2. 고정 기준

현재 public best는 다음 제출로 동결한다.

```text
results/submission_jointmix_p50_t5_c45_pb50_cb25_v1.csv
public score = 0.63999 (user-reported)
```

이 계획을 수행하는 동안 다음 항목은 사용자 승인 없이 변경하지 않는다.

- current-best weight
- PINN floor와 final floor
- current-best branch 파일
- test submission

새 실험은 기존 모델 코드를 덮어쓰지 않고 별도 경로에서 수행한다.

## 3. 목표 파이프라인

```text
LDAPS 원본만 -> LDAPS 전문가 -> group 1/2/3 예측
GFS 원본만   -> GFS 전문가   -> group 1/2/3 예측
GEFS 원본만  -> GEFS 전문가  -> group 1/2/3 예측

세 전문가 outer-year OOF 예측
-> pooled actual hard Score 기반 단순 가중 앙상블
-> current-best와 정확히 matched된 OOF 비교
-> 통과한 경우에만 동일 fold checkpoint로 test fold-bagging
```

SCADA teacher, optimal-grid, PINN 출력은 세 weather source 전문가의 입력에 넣지 않는다. 이들은 나중에 독립 branch로만 비교한다.

## 4. 금지 사항

다음 방식은 이 계획에서 사용하지 않는다.

- `full_v2` 전체 입력
- `group_family_quota65`와 고정 feature quota
- GEFS compact 23개 feature append
- 개별 feature importance 순위에 따른 무한 추가
- 같은 원천값의 raw/alias/physics 중복 입력
- 풍속의 제곱, 세제곱, sigmoid, 분위수, rolling 통계 무더기 생성
- LDAPS/GFS/GEFS를 처음부터 하나의 거대한 입력 행렬에 혼합
- OOF와 다른 full-data 고정 epoch 재학습으로 test 예측
- OOF provenance가 다른 current-best와의 근사 비교
- 사용자 요청 없는 submission 생성

## 5. 소스별 입력 계약

### 5.1 공통 원칙

- 각 source는 독립 tensor와 독립 scaler를 가진다.
- 풍향은 각 높이의 `u/v`를 그대로 유지한다.
- 풍속 `sqrt(u^2 + v^2)`은 각 vector pair에 정확히 한 번만 추가한다.
- scaling은 outer-train에서 변수별로 학습하고 같은 변수의 모든 grid에 공유한다.
- grid별로 따로 표준화해 공간 bias를 제거하지 않는다.
- 시간 입력은 lead, DOY sin/cos, HOD sin/cos만 사용한다.
- 같은 issue의 24시간 예보는 모두 동시에 공개된 값이므로 issue 내부 시간 보간은 허용한다.
- 결측을 0으로 숨기지 않고 보간 후 missing mask를 별도 제공한다.
- train/test에서 동일한 grid ordering, crop, channel ordering을 강제 검증한다.

### 5.2 LDAPS 전문가

공간 범위:

```text
제공된 16개 grid 전체, grid identity 유지
```

Core:

```text
heightAboveGround_50_50MUmax / 50MVmax
heightAboveGround_50_50MUmin / 50MVmin
heightAboveGround_10_10u / 10v
각 vector pair의 speed 1개
```

후속 변수군 ablation:

```text
L1: heightAboveGround_5_XBLWS / YBLWS와 speed
L2: 2m temperature, surface pressure, boundary-layer height
```

초기 제외:

```text
cloud, radiation, precipitation, snow 계열 전체
```

### 5.3 GFS 전문가

공간 범위:

```text
제공된 9개 grid 전체, grid identity 유지
```

Core:

```text
heightAboveGround_80_u / v
heightAboveGround_100_100u / 100v
surface_0_gust
각 vector pair의 speed 1개
```

후속 변수군 ablation:

```text
G1: 10m u/v, 850 hPa u/v와 speed
G2: 700 hPa u/v와 speed, surface pressure, 2m/850 hPa temperature
```

초기 제외:

```text
planetaryBoundaryLayer_0_VRATE
cloud, radiation, precipitation 계열 전체
```

`VRATE`는 vertical wind가 아니라 ventilation rate이므로 초기 wind core에서 제외한다.

### 5.4 GEFS 전문가

원본:

```text
NOAA GEFS 18Z
f021..f048, 3시간 간격
geavg ensemble mean
gespr ensemble spread
```

공간 범위:

```text
pressure 0.5 degree: farm 중심 7x7 crop
gust 0.25 degree: farm 중심 9x9 crop
```

Core mean:

```text
10m u/v
925 hPa u/v
850 hPa u/v
surface gust
각 vector pair의 speed 1개
```

후속 변수군 ablation:

```text
E1: 700 hPa u/v와 speed
E2: core와 동일한 u/v/gust ensemble spread
```

GEFS spread는 평균 예측값과 섞어 만든 통계량이 아니라 독립 uncertainty channel로 제공한다.

GEFS publication audit와 causal fallback은 기존 검증 결과를 유지한다. unsafe issue는 이전 safe issue로 교체하고 fallback mask를 제공한다.

## 6. 전문모델 v1

데이터 효과와 architecture 효과를 분리하기 위해 세 source에 동일한 모델 골격을 사용한다.

```text
input: [issue, 24 hours, channels, height, width]

per-hour shallow spatial MLP
-> 24-hour temporal TCN
-> three group-specific output heads
-> normalized group power prediction
```

초기 고정값:

```text
spatial encoder: shallow MLP
temporal hidden: 64
temporal receptive field: about 29 hours
output: direct official group targets
loss: pure 6% reward
checkpoint metric: validation actual hard Score
```

설계 이유:

- 기존 raw-grid 실험에서 CNN보다 flattened spatial MLP가 강했다.
- 24시간 전체 예보 궤적은 같은 issue에 속하므로 temporal model이 모두 사용할 수 있다.
- group별 head를 분리해 서로 다른 발전 곡선을 유지한다.
- shared source encoder는 group3의 짧은 label 기간에도 공통 풍황 표현을 제공한다.
- group3의 2022 결측은 loss mask로만 처리한다.
- 터빈별 인공 target이나 정적 share를 만들지 않고 공식 group target을 직접 학습한다.

첫 core 실험에서는 hidden size, depth, dropout, loss를 동시에 탐색하지 않는다.

## 7. OOF와 Test 계약

Outer folds:

```text
2022 validation <- 나머지 연도 train
2023 validation <- 나머지 연도 train
2024 validation <- 나머지 연도 train
```

규칙:

- held-out year의 actual hard Score로 checkpoint를 선택한다.
- 선택된 checkpoint가 해당 outer validation과 test를 즉시 예측한다.
- test는 세 outer-fold checkpoint prediction을 평균한다.
- full-data fixed epoch 재학습으로 test prediction을 교체하지 않는다.
- outer predictions를 모두 연결한 뒤 group별 NMAE와 FiCR을 한 번 계산한다.
- 최종 OOF Score는 세 group을 동일 가중한다.
- group-year 평균은 안정성 진단에만 사용한다.
- OOF, test prediction, checkpoint metadata에 code/config/data fingerprint를 기록한다.

## 8. 실행 단계

### Phase 0. 기준과 provenance 동결

- current public best와 구성 branch fingerprint 기록
- 최신 OOF와 public 결과의 provenance 분리
- 구형 문서의 current-best 표기를 활성 기준으로 사용하지 않도록 정리

### Phase 1. Source contract와 tensor audit

- source별 channel, unit, grid, time, missingness 표 생성
- train/test grid ordering과 분포 검증
- GEFS mean/spread/publication/fallback 검증
- 24시간 issue coverage 검증
- 파생 speed 중복과 alias 부재 검증

### Phase 2. Core source experts

다음 세 모델의 1-seed full outer-year OOF를 생성한다.

```text
LDAPS-core
GFS-core
GEFS-mean-core
```

각 expert에 대해 다음을 기록한다.

```text
pooled Score
NMAE
FiCR
group score
group-year score
lead bin score
target output bin score
prediction variance
current branch와의 residual correlation
```

### Phase 3. Core source ensemble

세 source expert의 OOF prediction만 사용한다.

```text
prediction = w_ldaps * pred_ldaps
           + w_gfs   * pred_gfs
           + w_gefs  * pred_gefs

w >= 0
sum(w) = 1
```

초기 방식:

```text
actual pooled hard Score 최적화
2.5% coarse grid
0.5% local refinement
모든 group에 공통 source weight
절편 없음
음수 weight 없음
```

group-specific weight는 source 우열이 여러 outer year에서 같은 방향일 때만 연다.

### Phase 4. 변수군 ablation

Core 결과를 확인한 뒤 아래 순서대로 한 변수군씩 추가한다.

```text
LDAPS: L1 -> L2
GFS: G1 -> G2
GEFS: E1 -> E2
```

개별 feature를 하나씩 고르는 방식은 사용하지 않는다. 변수군은 source standalone 또는 source ensemble에서 의미 있는 개선을 보일 때만 유지한다.

### Phase 5. Horizon interaction 진단

source별 상대 우열을 다음 lead bin에서 확인한다.

```text
12-17h
18-23h
24-29h
30-35h
```

여러 outer year에서 같은 패턴이 반복될 때만 horizon별 source weight를 허용한다. 패턴이 불안정하면 common weight를 유지한다.

### Phase 6. Promotion validation

- 1-seed 승격 모델만 고정 3-seed bagging
- current-best와 정확히 matched된 OOF 비교
- NMAE와 FiCR tradeoff 확인
- 2025 test prediction은 OOF에서 사용한 동일 fold checkpoint로만 생성
- 사용자 승인 전 submission CSV 생성 금지

## 9. 승격과 중단 기준

변수군 유지 조건:

```text
source standalone pooled Score +0.002 이상
또는 source ensemble pooled Score +0.001 이상
그리고 평가 가능한 group-year 중 최소 5개 개선
```

3-seed 승격 조건:

```text
core source ensemble이 1-seed에서 비교 기준 대비 +0.003 이상
```

최종 후보 조건:

```text
current-best matched OOF 대비 +0.005 이상
8개 group-year 중 최소 6개 개선
NMAE 또는 FiCR 한쪽의 심각한 붕괴 없음
```

기준 미달 결과는 진단으로만 기록한다. 작은 개선을 새 baseline이나 submission 후보로 포장하지 않는다.

## 10. 구현 파일

예정 신규 파일:

```text
utils/source_expert_dataset.py
models/source_expert_tcn.py
experiments/evaluate_source_experts_oof.py
experiments/blend_source_experts_oof.py
tests/test_source_expert_dataset.py
```

기존 current-best 학습/추론 파일은 첫 구현에서 수정하지 않는다.

## 11. 산출물과 실행 규칙

Duck 결과 경로:

```text
/home/yunjun0914/windforecast_runs/source_experts_v1/
```

예정 결과:

```text
ldaps_core_oof_predictions.csv
gfs_core_oof_predictions.csv
gefs_mean_core_oof_predictions.csv
source_expert_fold_scores.csv
source_expert_diagnostics.csv
source_expert_blend_summary.csv
source_expert_run_manifest.json
```

실험 산출물은 Git에 올리지 않는다. 코드, 테스트, 짧은 `docs/exp_logs.md` 기록만 관리한다.

Duck 실행 원칙:

- 코드는 한 번에 묶어서 전송한다.
- 여러 짧은 SSH 연결을 만들지 않는다.
- GPU 학습은 source별로 순차 실행한다.
- CPU 작업은 최대 8 jobs로 제한한다.
- 필요한 긴 작업은 하나의 remote session에서 이어서 실행한다.

예상 시간:

```text
source contract/tensor audit: 1-2 hours CPU
three 1-seed core OOF experts: 6-10 hours RTX 4090
promoted three-seed confirmation: additional 10-18 hours
```

## 12. 진행 체크리스트

- [x] 기존 pipeline과 feature 사용 방식 재점검
- [x] source-expert 방향과 금지 사항 합의
- [ ] current-best/OOF provenance 동결
- [ ] source contract와 raw tensor 구현
- [ ] source dataset unit tests
- [ ] LDAPS-core 1-seed OOF
- [ ] GFS-core 1-seed OOF
- [ ] GEFS-mean-core 1-seed OOF
- [ ] core source ensemble
- [ ] 변수군 ablation 승인
- [ ] promoted model 3-seed 확인
- [ ] current-best matched OOF 비교
- [ ] 사용자 승인 후 test submission 후보 생성

## 13. 계획 변경 규칙

다음 단계로 자동 진행하지 않는다.

- Phase 결과를 사용자에게 OOF 파일명과 함께 보고한다.
- 계획을 바꿀 때는 이 문서를 먼저 갱신한다.
- 피처나 모델을 추가할 때 목적, 기대 효과, validation, 예상 시간, 결과 파일을 먼저 설명한다.
- 새로운 아이디어가 생겨도 core source expert 결과가 나오기 전에 기존 혼합 feature 방식으로 돌아가지 않는다.


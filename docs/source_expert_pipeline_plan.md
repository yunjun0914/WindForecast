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
- 성능이 낮게 나온 뒤 보고 없이 weight, alpha, subset, seed 또는 하이퍼파라미터를 바꾸어 재실행
- 여러 비율을 임의로 섞은 뒤 가장 좋은 작은 개선만 결과로 제시
- 실패한 원래 설정을 생략하고 사후 튜닝 결과만 보고

## 4.1 실험 승인 게이트

이 계획에 적힌 후속 단계는 자동 실행 허가가 아니다.

각 실험 전 다음 내용을 사용자에게 먼저 보고하고 승인을 받는다.

```text
실험 목적
입력 데이터와 정확한 변수군
모델 구조와 loss
학습 fold와 checkpoint 선택 방식
비교 baseline
실행할 configuration의 정확한 개수
예상 실행 시간
수정 파일
결과 파일명
```

사용자가 승인한 configuration까지만 실행한다. 결과가 나오면 프로세스를 멈추고 다음 형식으로 먼저 보고한다.

```text
1. 실제 실행한 구조
2. 사용한 데이터와 피처
3. 학습과 validation 방식
4. pooled OOF Score, NMAE, FiCR
5. group 및 group-year 결과
6. baseline 대비 차이
7. 성공 또는 실패 해석
8. 다음 선택지 제안, 실행은 하지 않음
```

성능이 나오지 않으면 그대로 실패로 보고한다. 실패를 만회하려고 다음 항목을 임의로 실행하지 않는다.

```text
blend weight grid search
몇 퍼센트 소량 혼합
alpha sweep
feature subset 재조합
loss 교체
hidden size/depth/dropout 변경
seed cherry-pick
horizon interaction
floor/clip/후처리
다른 branch와 추가 앙상블
```

예를 들어 `GEFS 5%`, `GEFS 10%`, `GEFS 15%`를 뒤에서 섞어 가장 좋은 `+0.001`만 제시하는 행위는 금지한다. 이런 탐색이 필요하다고 판단되면 먼저 이유, 범위, 과적합 위험을 설명하고 별도 승인을 받는다.

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

Phase 2 결과를 사용자에게 먼저 보고하고 별도 승인을 받은 경우에만 실행한다.

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

각 변수군은 자동으로 이어서 실행하지 않는다. 직전 결과를 보고한 뒤 사용자가 승인한 변수군 하나만 실행한다.

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

패턴이 확인되어도 자동 실행하지 않고 진단 결과와 예정 weight 구조를 먼저 보고한다.

### Phase 6. Promotion validation

- 1-seed 승격 모델만 고정 3-seed bagging
- current-best와 정확히 matched된 OOF 비교
- NMAE와 FiCR tradeoff 확인
- 2025 test prediction은 OOF에서 사용한 동일 fold checkpoint로만 생성
- 사용자 승인 전 submission CSV 생성 금지
- 1-seed 결과를 보고하기 전에 3-seed, 추가 tuning 또는 blend를 자동 실행하지 않음

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
- [x] source contract와 raw tensor 구현
- [x] source dataset unit tests
- [x] LDAPS-core 1-seed OOF
- [x] GFS-core 1-seed OOF
- [x] GEFS-mean-core 1-seed OOF
- [x] core source ensemble
- [ ] 변수군 ablation 승인
- [ ] promoted model 3-seed 확인
- [ ] current-best matched OOF 비교
- [ ] 사용자 승인 후 test submission 후보 생성

### Phase 1 완료 기록

2026-07-16 Duck full audit, commit `52e8ba6`:

```text
LDAPS train/test: 1096/365 issues, 24h, 9 channels, 4x5 mask with 16 points
GFS train/test:   1096/365 issues, 24h, 7 channels, 3x3
GEFS pressure:    1096/365 issues, 24h, 9 channels, 7x7
GEFS gust:        1096/365 issues, 24h, 1 channel, 9x9
```

- synthetic contract/scaler/fallback unit test 5개 통과
- LDAPS train, GFS train/test, GEFS mean core는 원본 결측 0
- LDAPS test는 3개 issue에서 50m max/min vector 결측이 발생해 raw+speed 기준 96 cells/issue, 총 288 cells를 같은 issue 내부 시간 보간
- 모든 tensor는 보간 후 finite이며 train/test channel/grid schema가 동일
- GEFS mean/spread publication unsafe run은 각각 5개
- GEFS mean fallback은 train 3 issues, test 2 issues에 적용
- train 3 issues와 test 1 issue는 24시간 forecast가 두 calendar year에 걸침
- cross-year issue는 삭제하거나 한 연도로 배정하지 않고 시간별 forecast year를 tensor에 보존
- 학습, OOF, blend, submission은 실행하지 않음

산출물:

```text
Duck:  /home/yunjun0914/windforecast_runs/source_experts_v1/
Local: results/source_experts_v1_contract/
```

### Phase 2 완료 기록

2026-07-16 Duck full outer-year OOF, training commit `4596063`:

| Source | Score | NMAE | FiCR |
|---|---:|---:|---:|
| LDAPS core | 0.622694 | 0.145303 | 0.390690 |
| GFS core | 0.610861 | 0.150873 | 0.372595 |
| GEFS mean core | 0.610921 | 0.152362 | 0.374204 |

- seed 42, pure 6% generation-weighted reward, h64 3-block full-context TCN
- held-out-year actual hard Score checkpoint, OOF predictions pooled once by group
- calendar year를 가로지르는 3개 issue는 train/validation 양쪽에서 issue 전체 제외
- GEFS fallback 3개 train issue는 해당 fold의 train loss에서 제외하고 validation에는 유지
- LDAPS가 pooled 및 세 group 모두 1위; GFS는 2022 fold에서만 LDAPS보다 높음
- source residual correlation은 GEFS-GFS `0.8272`, GEFS-LDAPS `0.7417`, GFS-LDAPS `0.7623`
- source별 OOF 69,747행, 중복 key 0, non-finite 0
- blend, test prediction, submission은 실행하지 않음

산출물:

```text
Duck:  /home/yunjun0914/windforecast_runs/source_experts_v1/
Local: results/source_experts_v1/
```

### Phase 3 완료 기록

2026-07-16 leave-one-year-out meta blend, code commit `4a092b7`:

```text
prediction = w_ldaps * LDAPS + w_gfs * GFS + w_gefs * GEFS
w >= 0, sum(w) = 1, intercept/floor/group split 없음
```

각 held-out year를 제외한 두 OOF year에서 pooled hard Score로 공통 source weight를 선택했다. `2.5%` coarse grid 후 각 weight 주변 `0.5%` local refinement만 수행했다.

| Held-out year | LDAPS | GFS | GEFS | Held-out Score | LDAPS delta |
|---|---:|---:|---:|---:|---:|
| 2022 | 0.635 | 0.255 | 0.110 | 0.629839 | +0.010274 |
| 2023 | 0.650 | 0.120 | 0.230 | 0.626001 | +0.008642 |
| 2024 | 0.560 | 0.255 | 0.185 | 0.646325 | +0.006540 |

Pooled nested 결과:

```text
Score = 0.630831
NMAE  = 0.137974
FiCR  = 0.399637
LDAPS standalone 대비 +0.008137
```

- group delta도 g1 `+0.006924`, g2 `+0.009758`, g3 `+0.007730`으로 모두 개선
- 69,747행, 중복 key 0, non-finite 0
- test prediction과 submission은 생성하지 않음
- 이 평가는 held-out year의 label/error를 weight 선택에서 직접 제외하지만, base model까지 meta fold 안에서 재학습한 완전한 2-level nested CV는 아님
- meta-train OOF를 만든 base fold가 meta held-out year를 학습에 포함할 수 있으므로 `+0.008137`은 강한 source diversity 신호이며 완전히 unbiased한 미래 성능 추정치로 해석하지 않음

산출물:

```text
Local: results/source_experts_v1/source_expert_convex_nested_*.csv
```

### Phase 4 pressure 단일변수 ablation

2026-07-16, code commit `0384008`. 기존 core에서 다른 설정은 고정하고 `surface_0_sp` 한 채널만 LDAPS 16개 격자와 GFS 9개 격자에 각각 추가했다. GEFS parquet에는 scalar pressure가 없어 기존 mean-core를 그대로 사용했다.

Standalone:

| Source | Core Score | +Surface pressure | Delta |
|---|---:|---:|---:|
| LDAPS | 0.622694 | 0.621934 | -0.000760 |
| GFS | 0.610861 | 0.611738 | +0.000877 |

`LDAPS-SP + GFS-SP + 기존 GEFS`를 동일 leave-one-year-out convex blend로 평가:

```text
pressure blend Score = 0.630304
pressure blend NMAE  = 0.139408
pressure blend FiCR  = 0.400015
core source blend    = 0.630831
delta                = -0.000527
```

- held-out delta는 2022 `+0.001123`, 2023 `+0.001698`, 2024 `-0.003509`
- group delta는 g1 `-0.001789`, g2 `+0.002449`, g3 `-0.002243`
- pressure는 g2 FiCR를 올렸지만 전체 NMAE를 `+0.001434` 악화시켜 pooled Score가 하락
- 69,747행, 중복 key 0, non-finite 0
- surface pressure variant는 기각하며 core source blend를 유지
- `meanSea_0_prmsl`, GEFS pressure 추가수집, LDAPS/GFS 선택적 교체는 자동 실행하지 않음
- test prediction과 submission 없음

산출물:

```text
Duck expert OOF: /home/yunjun0914/windforecast_runs/source_experts_v1/pressure_sp_v1/
Local blend:     results/source_experts_v1/pressure_sp_v1/
```

### Phase 5 GEFS raw spread S1 ablation

2026-07-16, code commit `29f50c9`. GEFS mean-core의 설정은 고정하고 아래 ensemble spread 원시 채널 7개만 추가했다.

```text
pressure 7x7: u10m_sprd, v10m_sprd, u925_sprd, v925_sprd, u850_sprd, v850_sprd
gust 9x9:     gust_sprd
```

추가 파생은 없다. component spread norm, relative spread, confidence gating, 700hPa, blend weight 변경은 실행하지 않았다. `geavg`와 `gespr` 중 하나라도 publication-safe가 아니면 mean과 spread 전체를 직전 joint-safe issue로 함께 fallback했다.

| Variant | Score | NMAE | FiCR | Delta |
|---|---:|---:|---:|---:|
| GEFS mean core | 0.610921 | 0.152362 | 0.374204 | - |
| GEFS mean + raw spread | 0.608686 | 0.158196 | 0.375568 | -0.002235 |

Group Score delta:

```text
g1 = +0.000819
g2 = -0.004238
g3 = -0.003286
```

- seed42, pure 6%, h64 3-block full-context TCN, outer-year hard Score checkpoint 등 core와 동일
- pressure tensor `9 -> 15` channels, gust tensor `1 -> 2` channels
- 전체 FiCR은 `+0.001364`였지만 NMAE가 `+0.005834` 악화
- standalone 결과만으로 채택 여부를 판정하지 않고, 최종 source ensemble에서 matched replacement를 추가 평가
- test prediction과 submission 없음

Final source ensemble comparison:

```text
baseline = LDAPS core + GFS core + GEFS mean core
variant  = LDAPS core + GFS core + GEFS mean/raw-spread core
```

| Ensemble | Score | NMAE | FiCR | Delta |
|---|---:|---:|---:|---:|
| core source convex blend | 0.630831 | 0.137974 | 0.399637 | - |
| GEFS spread replacement | 0.630390 | 0.139286 | 0.400066 | -0.000441 |

- 동일 leave-one-year-out meta hard Score, nonnegative/sum-to-one, 2.5% coarse + 0.5% local refinement
- held-out delta 2022/2023/2024 `-0.000717/+0.001290/-0.001245`
- group delta g1/g2/g3 `+0.001631/+0.000124/-0.003079`
- GEFS weight는 baseline `11/23/18.5%`에서 spread variant `8/7/13.5%`로 감소
- FiCR은 소폭 좋아졌지만 NMAE와 g3가 악화되어 최종 ensemble Score 하락
- raw spread 7채널 직접입력은 최종 source ensemble 기준으로 현재 pipeline에 미채택
- spread 정보 자체는 폐기하지 않고 component norm, relative spread, confidence gating 같은 S2 파생 후보로 보류
- S2 파생 실험은 사용자 승인 전 자동 실행하지 않음

산출물:

```text
Duck: /home/yunjun0914/windforecast_runs/source_experts_v1/gefs_spread_s1_v1_29f50c9/
Local: results/source_experts_v1/gefs_spread_s1_v1_29f50c9/
OOF:   gefs_spread_core_oof_predictions.csv
Blend: results/source_experts_v1/gefs_spread_s1_blend_29f50c9/
```

### Phase 6 GFS 10m wind S1 ablation

2026-07-16, code commit `00c54f8`. GFS core의 80/100m `u/v`, gust, speed는 유지하고 10m `u/v/speed` 3채널만 추가했다.

```text
heightAboveGround_10_10u
heightAboveGround_10_10v
wind_10m_speed
```

100-10m shear, gust factor, direction, density 등 S2 파생은 포함하지 않았다.

Standalone:

| Variant | Score | NMAE | FiCR | Delta |
|---|---:|---:|---:|---:|
| GFS core | 0.610861 | 0.150873 | 0.372595 | - |
| GFS core + 10m wind | 0.612026 | 0.149284 | 0.373336 | +0.001165 |

Final source ensemble comparison:

```text
baseline = LDAPS core + GFS core + GEFS mean core
variant  = LDAPS core + GFS 10m core + GEFS mean core
```

| Ensemble | Score | NMAE | FiCR | Delta |
|---|---:|---:|---:|---:|
| core source convex blend | 0.630831 | 0.137974 | 0.399637 | - |
| GFS 10m replacement | 0.630926 | 0.137525 | 0.399378 | +0.000095 |

- 동일 leave-one-year-out meta hard Score와 convex weight search 사용
- held-out delta 2022/2023/2024 `+0.000111/-0.001130/+0.001085`
- group delta g1/g2/g3 `-0.000509/-0.000380/+0.001173`
- standalone은 개선됐지만 final ensemble 증가는 noise 수준이며 유지 기준 `+0.001` 미달
- 사용자 결정으로 raw 10m 직접입력을 source expert baseline에 채택
- 이후 source ensemble 비교 기준은 `LDAPS core + GFS 10m core + GEFS mean core = 0.630926`
- 10m 정보는 100-10m shear와 gust factor의 부모 원천정보로도 유지하되 S2는 자동 실행하지 않음
- test prediction과 submission 없음

산출물:

```text
Duck: /home/yunjun0914/windforecast_runs/source_experts_v1/gfs_10m_s1_v1_00c54f8/
Local: results/source_experts_v1/gfs_10m_s1_v1_00c54f8/
OOF:   gfs_10m_core_oof_predictions.csv
Blend: results/source_experts_v1/gfs_10m_s1_blend_00c54f8/
```

### Phase 7 LDAPS 5m BL wind S1 ablation

2026-07-16, code commit `d040483`. LDAPS core의 50m max/min 및 10m vector는 유지하고 5m boundary-layer `X/Y/speed` 3채널만 추가했다.

```text
heightAboveGround_5_XBLWS
heightAboveGround_5_YBLWS
wind_5m_speed
```

10-5m shear, 50-10m envelope shear, BLH interaction 등 S2 파생은 포함하지 않았다.

Standalone:

| Variant | Score | NMAE | FiCR | Delta |
|---|---:|---:|---:|---:|
| LDAPS core | 0.622694 | 0.145303 | 0.390690 | - |
| LDAPS core + 5m wind | 0.623029 | 0.147393 | 0.393450 | +0.000335 |

Final source ensemble comparison은 채택된 GFS 10m baseline 위에서 수행했다.

```text
baseline = LDAPS core + GFS 10m core + GEFS mean core
variant  = LDAPS 5m core + GFS 10m core + GEFS mean core
```

| Ensemble | Score | NMAE | FiCR | Delta |
|---|---:|---:|---:|---:|
| GFS 10m accepted baseline | 0.630926 | 0.137525 | 0.399378 | - |
| LDAPS 5m replacement | 0.630987 | 0.138227 | 0.400201 | +0.000061 |

- held-out delta 2022/2023/2024 `+0.005196/-0.001377/-0.000729`
- group delta g1/g2/g3 `-0.000111/+0.003912/-0.003619`
- g2 FiCR 개선을 g3 하락이 상쇄했고 연도별 방향도 불안정
- final ensemble 이득은 noise 수준이며 유지 기준 `+0.001` 미달
- raw 5m 직접입력은 사용자 결정 전 보류하며 baseline을 자동 갱신하지 않음
- test prediction과 submission 없음

산출물:

```text
Duck: /home/yunjun0914/windforecast_runs/source_experts_v1/ldaps_5m_s1_v1_d040483/
Local: results/source_experts_v1/ldaps_5m_s1_v1_d040483/
OOF:   ldaps_5m_core_oof_predictions.csv
Blend: results/source_experts_v1/ldaps_5m_gfs_10m_s1_blend_d040483/
```

### Phase 8 LDAPS hub/BLH ratio P1 ablation

2026-07-16, code commit `7c2f344`. 사전 변수 audit에서 세 group의 단순 current-wind residual과 같은 방향의 상관을 보인 `etc_0_blh`를 아래 물리 좌표 한 채널로만 변환했다.

```text
ldaps_hub_over_blh = 117m / max(BLH, 20m)
```

raw BLH, low-BLH flag, tendency, 시간 interaction과 다른 thermo 변수는 모델 입력에 포함하지 않았다. LDAPS core의 9채널은 유지하고 ratio 한 채널만 추가했으며, 동일 seed42/pure6/h64 TCN outer-year OOF를 bear의 독립 `WindForecast` 환경에서 실행했다.

Standalone:

| Variant | Score | NMAE | FiCR | Delta |
|---|---:|---:|---:|---:|
| LDAPS core | 0.622694 | 0.145303 | 0.390690 | - |
| LDAPS core + hub/BLH ratio | 0.620155 | 0.145415 | 0.385725 | -0.002539 |

Final source ensemble comparison은 채택된 GFS 10m baseline 위에서 수행했다.

```text
baseline = LDAPS core + GFS 10m core + GEFS mean core
variant  = LDAPS BLH ratio core + GFS 10m core + GEFS mean core
```

| Ensemble | Score | NMAE | FiCR | Delta |
|---|---:|---:|---:|---:|
| GFS 10m accepted baseline | 0.630926 | 0.137525 | 0.399378 | - |
| LDAPS hub/BLH replacement | 0.630882 | 0.136816 | 0.398581 | -0.000044 |

- held-out delta 2022/2023/2024 `+0.001186/-0.000864/+0.000185`
- group delta g1/g2/g3 `-0.002625/+0.003512/-0.001019`
- ensemble NMAE는 개선됐지만 FiCR 하락이 상쇄했고 g2 이득이 g1/g3 하락과 교환됨
- standalone은 명확히 하락했고 final delta도 음수 noise 수준이므로 ratio 직접입력은 미채택
- raw BLH, flag, tendency 등 다른 표현은 자동 실험하지 않으며 baseline은 `0.630926` 유지
- OOF와 blend 모두 69,747행, duplicate 0, non-finite 0; test prediction과 submission 없음

산출물:

```text
Bear:  /home/yunjun0914/windforecast_runs/source_experts_v1/ldaps_blh_p1_v1_7c2f344/
Local: results/source_experts_v1/ldaps_blh_p1_v1_7c2f344/
OOF:   ldaps_blh_ratio_core_oof_predictions.csv
Blend: results/source_experts_v1/ldaps_blh_gfs10_blend_7c2f344/
```

### Phase 9 LDAPS 3h pressure tendency P1 ablation

2026-07-16, code commit `524c5bb`. raw surface pressure의 지형·고도 정적 성분을 제거하고 기압계 이동만 표현하기 위해 아래 한 채널만 추가했다.

```text
interior: dp3(t) = (sp[t+3] - sp[t-3]) / 6 hours
edges:    available same-issue endpoints를 사용한 one-sided Pa/hour slope
```

raw pressure, pressure anomaly, spatial gradient와 다른 파생은 포함하지 않았다. LDAPS core의 9채널은 유지하고 tendency 한 채널만 추가했으며, 동일 seed42/pure6/h64 TCN strict outer-year OOF를 bear에서 실행했다.

Standalone:

| Variant | Score | NMAE | FiCR | Delta |
|---|---:|---:|---:|---:|
| LDAPS core | 0.622694 | 0.145303 | 0.390690 | - |
| LDAPS core + 3h pressure tendency | 0.620027 | 0.147783 | 0.387837 | -0.002667 |

Final source ensemble comparison은 채택된 GFS 10m baseline 위에서 수행했다.

```text
baseline = LDAPS core + GFS 10m core + GEFS mean core
variant  = LDAPS pressure-tendency core + GFS 10m core + GEFS mean core
```

| Ensemble | Score | NMAE | FiCR | Delta |
|---|---:|---:|---:|---:|
| GFS 10m accepted baseline | 0.630926 | 0.137525 | 0.399378 | - |
| LDAPS pressure-tendency replacement | 0.629523 | 0.138404 | 0.397449 | -0.001403 |

- held-out delta 2022/2023/2024 `-0.001227/-0.002520/+0.000082`
- group delta g1/g2/g3 `-0.002438/+0.000330/-0.002103`
- standalone과 final ensemble에서 NMAE와 FiCR이 모두 악화됐으므로 tendency 직접입력은 기각
- raw pressure, anomaly, spatial pressure gradient는 이번 실험에 포함하지 않았고 자동으로 이어서 실행하지 않음
- baseline은 `0.630926` 유지; OOF와 blend 모두 69,747행, duplicate 0, non-finite 0
- test prediction과 submission 없음

산출물:

```text
Bear:  /home/yunjun0914/windforecast_runs/source_experts_v1/ldaps_pressure_tendency_p1_v1_524c5bb/
Local: results/source_experts_v1/ldaps_pressure_tendency_p1_v1_524c5bb/
OOF:   ldaps_pressure_tendency_core_oof_predictions.csv
Blend: results/source_experts_v1/ldaps_pressure_tendency_gfs10_blend_524c5bb/
```

### Phase 10 LDAPS mean sea-level pressure P1 ablation

2026-07-16, code commit `eaaa6cb`. surface pressure의 지형·고도 정적 성분을 제거하면서 절대적인 고·저기압 regime을 보존하는 raw `meanSea_0_prmsl` 한 채널만 추가했다.

사전 audit:

```text
rows/missing/non-finite = 420,864 / 0 / 0
p01/p50/p99             = 99,759 / 101,514 / 102,972 Pa
spatial range p50/p99   = 33.6 / 147.6 Pa
corr(surface pressure)  = 0.637
```

raw surface pressure, tendency, anomaly와 gradient는 포함하지 않았다. LDAPS core 9채널을 유지하고 MSLP 한 채널만 추가해 동일 seed42/pure6/h64 TCN strict outer-year OOF를 bear에서 실행했다.

Standalone:

| Variant | Score | NMAE | FiCR | Delta |
|---|---:|---:|---:|---:|
| LDAPS core | 0.622694 | 0.145303 | 0.390690 | - |
| LDAPS core + MSLP | 0.622073 | 0.146975 | 0.391120 | -0.000621 |

Final source ensemble comparison은 채택된 GFS 10m baseline 위에서 수행했다.

```text
baseline = LDAPS core + GFS 10m core + GEFS mean core
variant  = LDAPS MSLP core + GFS 10m core + GEFS mean core
```

| Ensemble | Score | NMAE | FiCR | Delta |
|---|---:|---:|---:|---:|
| GFS 10m accepted baseline | 0.630926 | 0.137525 | 0.399378 | - |
| LDAPS MSLP replacement | 0.629152 | 0.138414 | 0.396719 | -0.001774 |

- held-out delta 2022/2023/2024 `+0.000156/-0.001944/-0.002289`
- group delta g1/g2/g3 `-0.002963/+0.001360/-0.003719`
- standalone FiCR은 소폭 올랐지만 NMAE가 악화됐고, final ensemble에서는 두 metric 모두 악화
- surface pressure, 3h tendency, MSLP가 모두 final source blend를 낮췄으므로 pressure scalar 직접입력 계열은 종료
- pressure를 T/q와 결합한 air density처럼 닫힌 물리량은 별도 후보이며 이번 결론에 포함하지 않음
- baseline은 `0.630926` 유지; OOF와 blend 모두 69,747행, duplicate 0, non-finite 0
- test prediction과 submission 없음

산출물:

```text
Bear:  /home/yunjun0914/windforecast_runs/source_experts_v1/ldaps_mslp_p1_v1_eaaa6cb/
Local: results/source_experts_v1/ldaps_mslp_p1_v1_eaaa6cb/
OOF:   ldaps_mslp_core_oof_predictions.csv
Blend: results/source_experts_v1/ldaps_mslp_gfs10_blend_eaaa6cb/
```

### Phase 11 LDAPS remaining raw family ablation

2026-07-16, code commit `2c5478b`. 하나씩 보는 비용을 줄이기 위해 남은 LDAPS raw를 두 family로 묶되, 각 family는 core에 독립적으로 추가했다. 두 family를 서로 합치거나 하위 subset을 탐색하지 않았다.

Thermo/PBL, 5채널:

```text
2m temperature, dew point, relative humidity, specific humidity, raw BLH
LDAPS channels: 9 -> 14
```

Surface-regime, 14채널:

```text
radiation 4 + cloud 4 + precipitation 3 + snow 2 + terrain height 1
LDAPS channels: 9 -> 23
```

20개 후보 raw는 결측·비유한값 0이었다. `surface_0_lsm`은 전체 420,864행이 값 1인 상수라 입력에서 제외했다. pressure fields와 5m wind는 이미 별도 실험했으므로 두 family에 중복 포함하지 않았다.

Standalone:

| Variant | Score | NMAE | FiCR | Delta |
|---|---:|---:|---:|---:|
| LDAPS core | 0.622694 | 0.145303 | 0.390690 | - |
| LDAPS + Thermo/PBL | 0.620316 | 0.150344 | 0.390975 | -0.002378 |
| LDAPS + Surface-regime | 0.618141 | 0.147533 | 0.383816 | -0.004553 |

각 family의 final source ensemble은 채택된 GFS 10m baseline 위에서 독립적으로 계산했다.

| Ensemble | Score | NMAE | FiCR | Delta |
|---|---:|---:|---:|---:|
| GFS 10m accepted baseline | 0.630926 | 0.137525 | 0.399378 | - |
| LDAPS Thermo/PBL replacement | 0.631021 | 0.139480 | 0.401522 | +0.000095 |
| LDAPS Surface-regime replacement | 0.628907 | 0.137659 | 0.395474 | -0.002019 |

Thermo/PBL:

- held-out delta 2022/2023/2024 `+0.003884/-0.001615/-0.000037`
- group delta g1/g2/g3 `+0.001775/+0.000676/-0.002167`
- FiCR 개선으로 pooled는 noise 수준 양수지만 standalone, NMAE, g3, 2023이 하락해 승격 기준 `+0.001` 미달
- family raw 직접입력은 미채택하며 하위 raw subset을 자동 탐색하지 않음

Surface-regime:

- held-out delta 2022/2023/2024 `+0.001888/-0.001388/-0.004241`
- group delta g1/g2/g3 `-0.001852/+0.001029/-0.005233`
- standalone과 final ensemble 모두 명확히 하락해 family 전체 기각

공통:

- 네 prediction 파일 모두 69,747행, duplicate 0, non-finite 0
- baseline은 `0.630926` 유지, test prediction과 submission 없음
- 이 결과로 LDAPS raw family 직접입력 점검은 종료. air density나 shear 같은 파생 물리량은 별도 단계

산출물:

```text
Bear:    /home/yunjun0914/windforecast_runs/source_experts_v1/ldaps_raw_families_v1_2c5478b/
Local:   results/source_experts_v1/ldaps_raw_families_v1_2c5478b/
OOF:     ldaps_thermo_pbl_core_oof_predictions.csv
         ldaps_surface_regime_core_oof_predictions.csv
Blend 1: results/source_experts_v1/ldaps_thermo_pbl_gfs10_blend_2c5478b/
Blend 2: results/source_experts_v1/ldaps_surface_regime_gfs10_blend_2c5478b/
```

### Phase 12-A GFS remaining raw family ablation

2026-07-16, code commit `f0f7b3d`. 채택된 GFS 10m core 10채널을 공통 부모로 두고 남은 raw 28개를 세 family로 나눴다. family끼리 합치거나 하위 subset을 탐색하지 않았다.

```text
Vertical-wind    +13ch: PBL/850/700/500hPa u/v/speed + PBL VRATE  (10 -> 23)
Thermo/synoptic  +11ch: 2m T/Td/RH/q, sp/MSLP, 850/700/500 T/RH/gh (10 -> 21)
Surface-regime    +8ch: radiation 2 + precipitation 2 + cloud 4     (10 -> 18)
```

28개 raw 모두 결측·비유한값·상수 채널이 없었다. 동일 seed42/pure6/h64 TCN strict outer-year OOF를 bear에서 한 세션으로 실행했다.

Standalone:

| Variant | Score | NMAE | FiCR | Delta vs GFS10 |
|---|---:|---:|---:|---:|
| GFS 10m core | 0.612026 | 0.149284 | 0.373336 | - |
| GFS Vertical-wind | 0.613392 | 0.145799 | 0.372583 | +0.001366 |
| GFS Thermo/synoptic | 0.618382 | 0.148560 | 0.385324 | +0.006356 |
| GFS Surface-regime | 0.609827 | 0.150956 | 0.370610 | -0.002199 |

각 family는 LDAPS core와 GEFS mean을 고정한 동일 source blend에서 독립 평가했다.

| GFS replacement | Score | NMAE | FiCR | Delta vs accepted baseline |
|---|---:|---:|---:|---:|
| GFS 10m baseline | 0.630926 | 0.137525 | 0.399378 | - |
| Vertical-wind | 0.633095 | 0.135638 | 0.401827 | +0.002169 |
| Thermo/synoptic | 0.633318 | 0.137158 | 0.403795 | +0.002392 |
| Surface-regime | 0.630325 | 0.137964 | 0.398614 | -0.000601 |

Vertical-wind:

- held-out delta 2022/2023/2024 `+0.003735/-0.002270/+0.005761`
- group delta g1/g2/g3 `-0.000356/+0.004052/+0.002809`
- standalone과 final blend가 승격 기준을 통과해 사용자 결정으로 별도 GFS expert 채택

Thermo/synoptic:

- held-out delta 2022/2023/2024 `+0.005291/+0.003566/-0.000062`
- group delta g1/g2/g3 `+0.004727/+0.002755/-0.000305`
- standalone `+0.006356`, final `+0.002392`로 안정적인 개선. 사용자 결정으로 별도 GFS expert 채택

Surface-regime:

- held-out delta 2022/2023/2024 `-0.000709/+0.000086/-0.000598`
- group delta g1/g2/g3 `+0.000686/+0.000189/-0.002678`
- standalone과 final blend가 모두 하락해 family 전체 기각

### Phase 12-B GFS Vertical+Thermo feature-concat ablation

2026-07-16, code commit `ac2e8f6`. 채택된 두 family를 별도 expert로 유지하는 것과 구분하기 위해, 두 입력을 한 GFS encoder에 넣은 34채널 combined expert를 추가 검증했다. Surface-regime은 포함하지 않았다.

| GFS expert | Standalone Score | Final source blend |
|---|---:|---:|
| Vertical-wind | 0.613392 | 0.633095 |
| Thermo/synoptic | 0.618382 | 0.633318 |
| Vertical+Thermo 34ch | 0.619764 | 0.631735 |

Combined final metric:

```text
Score = 0.631735
NMAE = 0.135818
FiCR  = 0.399288
```

- combined held-out delta vs GFS10 baseline 2022/2023/2024 `+0.004901/-0.002169/+0.001776`
- combined group delta g1/g2/g3 `+0.001778/+0.001734/-0.001084`
- feature concat은 GFS standalone을 더 올렸지만 LDAPS/GEFS와의 complementarity가 줄어 final blend는 두 단독 family replacement보다 낮음
- Vertical과 Thermo family는 각각 유효한 입력군으로 채택
- 후속 사용자 결정으로, final blend 하락을 감수하고 두 family를 한 모델에 넣은 34채널 combined expert를 현재 GFS backbone으로 채택
- 이후 source feature 실험의 고정 기준은 `LDAPS core + GFS Vertical/Thermo 34ch + GEFS mean = 0.631735`
- 모든 OOF/blend prediction은 69,747행, duplicate 0, non-finite 0; test/submission 없음

산출물:

```text
Raw family OOF: results/source_experts_v1/gfs_raw_families_v1_f0f7b3d/
Vertical blend: results/source_experts_v1/gfs_vertical_wind_blend_f0f7b3d/
Thermo blend:   results/source_experts_v1/gfs_thermo_synoptic_blend_f0f7b3d/
Surface blend:  results/source_experts_v1/gfs_surface_regime_blend_f0f7b3d/
Combined OOF:   results/source_experts_v1/gfs_vertical_thermo_v1_ac2e8f6/
Combined blend: results/source_experts_v1/gfs_vertical_thermo_blend_ac2e8f6/
```

### Phase 13 GEFS raw family split ablation

2026-07-16, code commit `bbd8196`. 과거 실패한 GEFS full-spread 7채널 직접입력을 반복하지 않고, 현재 mean core에서 신규 정보의 역할에 따라 세 family를 독립적으로 구성했다.

```text
Mean700      +3ch: 700hPa mean u/v/speed
Near spread  +7ch: 10m/925hPa spread u/v/norm + gust spread
Upper spread +6ch: 850/700hPa spread u/v/norm
```

`spread norm = sqrt(u_sprd^2 + v_sprd^2)`는 실제 speed 표준편차가 아닌 component uncertainty magnitude proxy다. 세 family는 Bear RTX 4080 한 장에서 세 프로세스로 병렬 실행했다. 현재 GFS backbone은 사용자 결정에 따라 Vertical/Thermo 34ch combined를 사용했다.

Standalone:

| GEFS expert | Score | NMAE | FiCR | Delta vs mean core |
|---|---:|---:|---:|---:|
| GEFS mean core | 0.610921 | 0.152362 | 0.374204 | - |
| Mean700 | 0.610095 | 0.150794 | 0.370984 | -0.000826 |
| Near spread | 0.609201 | 0.151570 | 0.369973 | -0.001720 |
| Upper spread | 0.601498 | 0.156321 | 0.359317 | -0.009423 |

Final source blend:

| GEFS replacement | Score | NMAE | FiCR | Delta vs current baseline |
|---|---:|---:|---:|---:|
| Current GEFS mean baseline | 0.631735 | 0.135818 | 0.399288 | - |
| Mean700 | 0.630369 | 0.136822 | 0.397561 | -0.001366 |
| Near spread | 0.629984 | 0.136754 | 0.396722 | -0.001751 |
| Upper spread | 0.629894 | 0.136613 | 0.396401 | -0.001841 |

Mean700:

- held-out delta 2022/2023/2024 `-0.001901/-0.002614/+0.000614`
- group delta g1/g2/g3 `-0.000867/-0.001378/-0.001852`
- NMAE standalone은 개선됐지만 FiCR이 더 하락했고 final blend도 세 group 중 어느 곳도 개선하지 못함

Near spread:

- held-out delta 2022/2023/2024 `-0.000150/-0.003894/-0.000330`
- group delta g1/g2/g3 `-0.002650/-0.000210/-0.002393`
- 이전 full-spread보다 의미를 좁혔지만 direct feature로는 final blend를 개선하지 못함

Upper spread:

- held-out delta 2022/2023/2024 `-0.000969/-0.002766/-0.001135`
- group delta g1/g2/g3 `-0.000555/-0.002360/-0.002608`
- standalone과 모든 held-out year가 하락해 direct feature family 기각

공통:

- 세 GEFS OOF와 세 blend prediction은 각각 69,747행, duplicate 0, non-finite 0
- Mean700, Near spread, Upper spread 직접입력은 현재 backbone에 미채택
- spread 정보의 relative uncertainty 또는 source confidence gating 사용은 미실험이며 이번 결과에 포함하지 않음
- family 합본, subset, weight 후속 탐색, test prediction, submission 없음

산출물:

```text
Bear OOF:  /home/yunjun0914/windforecast_runs/source_experts_v1/gefs_raw_families_v1_bbd8196/
Local OOF: results/source_experts_v1/gefs_raw_families_v1_bbd8196/
Mean blend:  results/source_experts_v1/gefs_mean700_blend_bbd8196/
Near blend:  results/source_experts_v1/gefs_near_spread_blend_bbd8196/
Upper blend: results/source_experts_v1/gefs_upper_spread_blend_bbd8196/
```

### Phase 14 LDAPS derived family ablation

2026-07-16, code commit `84a852a`. LDAPS raw 점검을 마친 뒤 현재 core에 물리적으로 닫힌 파생 family 하나씩만 추가했다. 9개 family는 Bear RTX 4080 한 장에서 9개 프로세스로 병렬 실행했으며, family 합본·하위 subset·추가 튜닝은 하지 않았다.

50m `max/min` component를 각각 speed로 만든 뒤 빼면 전체 행의 26.6%에서 음수가 됐다. 이는 component extrema가 같은 순간의 vector가 아니기 때문이다. Envelope와 이후 50m profile은 아래처럼 component midpoint와 component range를 사용했다.

```text
mid_u = (u_max + u_min) / 2
mid_v = (v_max + v_min) / 2
envelope_norm = sqrt((u_max-u_min)^2 + (v_max-v_min)^2)
```

Family 구성:

| Family | 추가 채널 | 핵심 파생 |
|---|---:|---|
| Envelope | 7 | 50m midpoint vector/speed, component envelope, relative envelope |
| Vertical profile | 6 | 50mid-10m 및 10-5m speed/shear/alignment |
| Density/Power | 6 | virtual T, rho, density-equivalent wind, power density |
| PBL/Stability | 5 | hub/BLH, BLH tendency, envelope/shear interaction, ventilation |
| Spatial flow | 9 | grid mean/std/p90-p10, coherence, linear x/y gradient |
| Terrain interaction | 6 | relative height, terrain plane slope, wind alignment/exposure |
| Temporal trajectory | 10 | 1h ramp, vector turn, roll3, issue range/peak timing |
| Thermodynamic regime | 8 | T-Td, RH deficit, pressure gap/anomaly/gradient, T/q gradient |
| Weather regime | 9 | radiation, cloud dominance, precipitation/snow activity |

Standalone:

| LDAPS expert | Score | Delta vs core |
|---|---:|---:|
| LDAPS core | 0.622694 | - |
| Envelope | 0.621035 | -0.001659 |
| Vertical profile | 0.624016 | +0.001322 |
| Density/Power | 0.623169 | +0.000476 |
| PBL/Stability | 0.621657 | -0.001037 |
| Spatial flow | 0.623164 | +0.000471 |
| Terrain interaction | 0.622102 | -0.000591 |
| Temporal trajectory | 0.618277 | -0.004417 |
| Thermodynamic regime | 0.621419 | -0.001275 |
| Weather regime | 0.619864 | -0.002830 |

Final source blend는 사용자 결정으로 채택한 GFS Vertical/Thermo 34ch와 GEFS mean을 고정했다.

| LDAPS replacement | Score | NMAE | FiCR | Delta vs current baseline |
|---|---:|---:|---:|---:|
| Current LDAPS core baseline | 0.631735 | 0.135818 | 0.399288 | - |
| Envelope | 0.628086 | 0.140062 | 0.396233 | -0.003649 |
| Vertical profile | 0.631404 | 0.137160 | 0.399967 | -0.000331 |
| Density/Power | 0.633330 | 0.136711 | 0.403372 | +0.001595 |
| PBL/Stability | 0.630525 | 0.137111 | 0.398160 | -0.001210 |
| Spatial flow | 0.633498 | 0.136181 | 0.403177 | +0.001763 |
| Terrain interaction | 0.631330 | 0.137279 | 0.399940 | -0.000405 |
| Temporal trajectory | 0.628059 | 0.137284 | 0.393401 | -0.003677 |
| Thermodynamic regime | 0.631351 | 0.136934 | 0.399635 | -0.000384 |
| Weather regime | 0.627630 | 0.139859 | 0.395120 | -0.004105 |

Density/Power:

- held-out delta 2022/2023/2024 `+0.001087/+0.000971/+0.003094`
- group delta g1/g2/g3 `+0.002746/+0.001747/+0.000292`
- 세 연도와 세 group이 모두 개선되어 사용자 결정으로 별도 LDAPS expert 채택

Spatial flow:

- held-out delta 2022/2023/2024 `+0.002718/+0.001748/+0.001098`
- group delta g1/g2/g3 `+0.000855/+0.002659/+0.001776`
- 세 연도와 세 group이 모두 개선되고 pooled delta가 가장 커 사용자 결정으로 별도 LDAPS expert 채택

나머지:

- Vertical profile은 standalone `+0.001322`였지만 final blend `-0.000331`로 source complementarity 개선 없음
- Terrain과 Thermodynamic은 특정 연도/group 개선이 있으나 pooled는 각각 `-0.000405/-0.000384`
- Envelope, PBL, Temporal, Weather는 final blend가 명확히 하락
- 9개 OOF와 9개 blend prediction은 각각 69,747행, duplicate 0, non-finite 0
- Density와 Spatial은 각각 채택. 후속 Phase 15에서 사용자 승인으로 24ch 합본을 별도 검증
- test prediction과 submission 없음

산출물:

```text
Bear OOF:  /home/yunjun0914/windforecast_runs/source_experts_v1/ldaps_derived_families_v1_84a852a/
Local OOF: results/source_experts_v1/ldaps_derived_families_v1_84a852a/
Blend:     results/source_experts_v1/ldaps_<family>_derived_blend_84a852a/
```

### Phase 15 LDAPS Density+Spatial 24ch feature union

2026-07-16, code commit `5515d6e`. 사용자 결정으로 채택한 Density/Power와 Spatial Flow를 한 LDAPS encoder에 결합했다.

```text
LDAPS core       9ch
Density/Power   +6ch
Spatial Flow    +9ch
Combined total  24ch
```

- 두 family 사이 파생채널 중복 0
- T/q/surface pressure 같은 부모 raw는 출력에 다시 넣지 않음
- 단독 family builder가 만든 파생 tensor를 그대로 union하여 수식 변화 없음
- 추가 family, subset, 파라미터 변경 없음

| LDAPS expert | Standalone | Final source blend | Delta vs current baseline |
|---|---:|---:|---:|
| LDAPS core | 0.622694 | 0.631735 | - |
| Density/Power | 0.623169 | 0.633330 | +0.001595 |
| Spatial Flow | 0.623164 | 0.633498 | +0.001763 |
| Density+Spatial 24ch | 0.622928 | 0.631966 | +0.000231 |

Combined final metric:

```text
Score = 0.631966
NMAE = 0.137248
FiCR  = 0.401180
```

- held-out delta vs core baseline 2022/2023/2024 `+0.001483/-0.001224/+0.001616`
- group delta g1/g2/g3 `-0.000570/+0.003258/-0.001995`
- 두 family를 한 encoder에 넣으면 g2는 크게 개선되지만 g1/g3와 2023에서 상쇄
- combined는 baseline보다 소폭 높지만 Density 단독보다 `-0.001364`, Spatial 단독보다 `-0.001532`
- Density와 Spatial은 각각 별도 LDAPS expert로 채택 유지. 24ch combined는 미채택
- OOF와 blend prediction은 각각 69,747행, duplicate 0, non-finite 0
- test prediction과 submission 없음

산출물:

```text
OOF:   results/source_experts_v1/ldaps_density_spatial_v1_5515d6e/
Blend: results/source_experts_v1/ldaps_density_spatial_blend_5515d6e/
```

### Phase 16 LDAPS family-separated expert blend

2026-07-16. Density/Power와 Spatial Flow를 한 24ch encoder로 합치지 않고 각각 독립 expert로 유지한 뒤, 현재 GFS Vertical/Thermo와 GEFS mean을 함께 4-way convex blend했다.

```text
LDAPS Density expert  15ch ─┐
LDAPS Spatial expert  18ch ─┼─ nested hard-Score convex blend
GFS Vertical/Thermo        ─┤
GEFS mean                  ─┘
```

- 기존 strict outer-year OOF prediction만 재사용. 모델 재학습 없음
- held-out year를 제외한 두 OOF year의 pooled official hard Score로 weight 선택
- nonnegative, sum-to-one, intercept 없음
- coarse `0.025`, local refinement `0.005`, 공통 weight vector 1개/held-out year
- 기존 3-expert Density blend를 재실행해 세 fold weight와 최종 `0.633330`이 정확히 재현됨

| 구성 | Final source blend | Delta vs current baseline |
|---|---:|---:|
| Current LDAPS/GFS/GEFS | 0.631735 | - |
| Density expert only | 0.633330 | +0.001595 |
| Spatial expert only | 0.633498 | +0.001763 |
| Density+Spatial 24ch encoder | 0.631966 | +0.000231 |
| Density+Spatial separate experts | 0.632478 | +0.000743 |

4-expert final metric:

```text
Score = 0.632478
NMAE = 0.136106
FiCR  = 0.401062
```

| Held-out | Density | Spatial | GFS | GEFS | Score |
|---:|---:|---:|---:|---:|---:|
| 2022 | 0.470 | 0.150 | 0.205 | 0.175 | 0.636182 |
| 2023 | 0.110 | 0.370 | 0.490 | 0.030 | 0.621815 |
| 2024 | 0.245 | 0.370 | 0.200 | 0.185 | 0.651941 |

- 24ch 합본보다는 `+0.000512` 높아 family 분리 신호는 확인
- current baseline보다 `+0.000743` 높지만 Density 단독보다 `-0.000852`, Spatial 단독보다 `-0.001020`
- group delta vs current baseline g1/g2/g3 `+0.001627/+0.001554/-0.000952`
- 두 LDAPS expert는 모든 fold에서 nonzero weight를 받았지만 weight가 연도별로 크게 변함
- 두 meta-train year로 4개 hard-Score weight를 고르면 2023 held-out이 하락하여 단독 expert 이득을 상쇄
- family-separated 4-way blend는 현재 미채택. 추가 weight 조정이나 subset 탐색 없음
- OOF prediction 69,747행, duplicate 0, non-finite 0. test/submission 없음

산출물:

```text
results/source_experts_v1/ldaps_family_experts_blend_bac7bb9/
```

### Phase 17 LDAPS representation-basis family ablation

2026-07-16 사용자 승인. 기존 LDAPS core 9ch에 신규 raw를 추가하지 않고, 같은 핵심 풍장의 비선형 표현만 family별로 독립 검증한다.

| Family | Added | Total | Contract |
|---|---:|---:|---|
| Circular direction | 4 | 13 | 10m `cos/sin(theta)`, `cos/sin(2theta)`; 0.5m/s floor로 calm down-weight |
| Polynomial basis | 18 | 27 | core 9ch 각각 `x^2`, `x^3` |
| Pairwise basis | 36 | 45 | core 9ch의 unordered `x_i*x_j`, `9C2=36` |
| Spatial centroid | 4 | 13 | 10m/50m-mid speed-weighted relative grid centroid x/y |
| Time modulation | 8 | 17 | 10m/50m-mid speed x day/hour sin/cos |

구조 판단:

- 현재 time context는 lead와 day/hour sin/cos를 spatial MLP 뒤에서 concat한다.
- `raw + time`은 기존 선형층이 표현 가능해 추가하지 않는다.
- `raw * time`은 spatial compression 전 시간 조건부 풍장을 직접 제공하므로 별도 family로 본다.
- 고정 grid를 펼치는 SpatialMLP에서는 raw x/y와 `x*raw`가 cell별 linear weight에 흡수되므로 직접 좌표 채널은 추가하지 않는다.
- Spatial centroid는 좌표를 이용한 비선형 normalized moment이므로 기존 입력과 선형 중복이 아니다.

Validation:

- 각 family를 독립 strict outer-year OOF로 실행
- 동일 seed/model/loss/early stopping 유지
- 현재 GFS Vertical/Thermo와 GEFS mean을 고정한 source blend까지 평가
- family 합본, subset, weight 후속 튜닝 없음
- test prediction과 submission 없음

계획 산출물:

```text
Bear OOF: /home/yunjun0914/windforecast_runs/source_experts_v1/ldaps_representation_families_v1_<commit>/
Local:    results/source_experts_v1/ldaps_representation_families_v1_<commit>/
Blend:    results/source_experts_v1/ldaps_<family>_blend_<commit>/
```

결과 (`3b7f914`):

| Family | Standalone | Final source blend | Delta vs current baseline |
|---|---:|---:|---:|
| LDAPS core | 0.622694 | 0.631735 | - |
| Circular direction | 0.623377 | 0.632749 | +0.001014 |
| Polynomial basis | 0.624563 | 0.630377 | -0.001358 |
| Pairwise basis | 0.624447 | 0.633467 | +0.001732 |
| Spatial centroid | 0.625269 | 0.633129 | +0.001394 |
| Time modulation | 0.619855 | 0.628122 | -0.003613 |

진단:

- Pairwise held-out delta 2022/2023/2024 `+0.001304/+0.001666/+0.002927`
- Pairwise group delta g1/g2/g3 `+0.002671/+0.002717/-0.000193`; 세 연도는 모두 개선하고 g3는 사실상 중립
- Spatial centroid는 standalone delta `+0.002575`로 가장 큰 신규 정보 신호. final은 2023 `-0.000975`, g1 `-0.002704`로 일부 상쇄
- Circular는 세 연도 모두 개선했으나 g3 `-0.002161`
- Polynomial은 standalone이 `+0.001869`지만 final에서 2023/2024와 g3가 하락하여 다른 source와 오차가 겹침
- Time modulation은 standalone/final 모두 명확히 하락. 기존 TCN time context 위에 명시적 곱을 추가하는 방식은 미채택
- Pairwise/Centroid/Circular의 채택 여부는 사용자 결정 전 보류. family 합본/subset 추가 탐색 없음
- 각 OOF/blend prediction 69,747행, duplicate 0, non-finite 0. test/submission 없음

실제 산출물:

```text
OOF:   results/source_experts_v1/ldaps_representation_families_v1_3b7f914/
Blend: results/source_experts_v1/ldaps_<family>_representation_blend_3b7f914/
```

## 18. 계획 변경 규칙

다음 단계로 자동 진행하지 않는다.

- Phase 결과를 사용자에게 OOF 파일명과 함께 보고한다.
- 계획을 바꿀 때는 이 문서를 먼저 갱신한다.
- 피처나 모델을 추가할 때 목적, 기대 효과, validation, 예상 시간, 결과 파일을 먼저 설명한다.
- 새로운 아이디어가 생겨도 core source expert 결과가 나오기 전에 기존 혼합 feature 방식으로 돌아가지 않는다.
- 성능이 좋지 않아도 야금야금 튜닝하지 않고 실패 구조와 수치를 먼저 보고한다.
- 작은 사후 blend 개선은 체급 상승으로 표현하지 않는다.


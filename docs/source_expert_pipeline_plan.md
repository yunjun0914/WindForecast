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

## 13. 계획 변경 규칙

다음 단계로 자동 진행하지 않는다.

- Phase 결과를 사용자에게 OOF 파일명과 함께 보고한다.
- 계획을 바꿀 때는 이 문서를 먼저 갱신한다.
- 피처나 모델을 추가할 때 목적, 기대 효과, validation, 예상 시간, 결과 파일을 먼저 설명한다.
- 새로운 아이디어가 생겨도 core source expert 결과가 나오기 전에 기존 혼합 feature 방식으로 돌아가지 않는다.
- 성능이 좋지 않아도 야금야금 튜닝하지 않고 실패 구조와 수치를 먼저 보고한다.
- 작은 사후 blend 개선은 체급 상승으로 표현하지 않는다.


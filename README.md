# WindForecast

강원도 풍력발전단지 KPX 그룹별 시간당 발전량을 예측하는 프로젝트입니다.

이 저장소의 핵심은 두 가지 모델 계열입니다.

1. **Tree baseline**: RF + LGBM + XGB + pooled isotonic calibration
2. **PINN line**: SCADA teacher로 현장 풍속분포를 복원한 뒤, 물리식 기반 PINN으로 발전량 예측

현재 2024 time-holdout 검증에서 가장 좋은 전체 후보는 group별 recipe를 다르게 쓰는 방식입니다.

```text
group_1 = 0.70 * PINN + 0.30 * calibrated tree ensemble
group_2 = 0.85 * (0.70 * PINN + 0.30 * calibrated tree ensemble)
        + 0.15 * raw tree ensemble
group_3 = 0.90 * (0.70 * PINN(mixed UNISON/VESTAS teacher) + 0.30 * group_2/VESTAS tree transfer)
        + 0.10 * calibrated own tree ensemble
```

검증 점수는 평균 **0.6307** 후보입니다.

| 모델/조합 | group_1 | group_2 | group_3 | 평균 |
|---|---:|---:|---:|---:|
| RF+LGBM+XGB raw | 0.6027 | 0.6321 | 0.5587 | 0.5978 |
| RF+LGBM+XGB pooled isotonic | - | - | - | 0.6021 |
| PINN 단독 | 0.6341 | 0.6339 | 0.5886 | 0.6189 |
| 공통 recipe: PINN 70% + calibrated tree 30% | 0.6357 | 0.6517 | 0.5862 | 0.6245 |
| group3에 group2 tree proxy만 추가 | 0.6357 | 0.6517 | 0.5994 | 0.6289 |
| group3 teacher transfer + tree transfer | 0.6357 | 0.6517 | 0.6024 | 0.6299 |
| **group별 recipe + 추가 tree 앙상블 최고 후보** | **0.6357** | **0.6530** | **0.6034** | **0.6307** |

> 주의: 위 최고 조합은 현재 `evaluate_pinn_tree_blend.py`, `evaluate_group3_transfer_blend.py`, `evaluate_group3_pinn_teacher_transfer.py`, `evaluate_final_tree_ensemble.py`로 검증하는 실험 경로입니다. 기존 `predict.py`는 아직 tree-only 제출 파일 생성 경로입니다.

## 예측 대상

| 컬럼 | 터빈 구성 | 설비용량 |
|---|---|---:|
| `kpx_group_1` | VESTAS V126 1~6호기 | 21.6 MW |
| `kpx_group_2` | VESTAS V126 7~12호기 | 21.6 MW |
| `kpx_group_3` | UNISON U136 1~5호기 | 21.0 MW |

평가지표는 다음입니다.

```text
total_score = 0.5 * (1 - NMAE) + 0.5 * FICR
```

- `NMAE`: 설비용량 대비 평균절대오차. 실제 발전량이 설비용량 10% 이상인 시간만 평가합니다.
- `FICR`: 오차율 구간별 정산금 점수입니다. 오차율 6% 이하가 가장 좋고, 8%를 넘으면 정산금 기여가 0이 됩니다.

## 데이터 준비

`data/`는 git에 포함되어 있지 않습니다. 실행 전 아래 구조로 원본 데이터를 넣어야 합니다.

```text
data/
├── data_description.md
├── info.xlsx
├── sample_submission.csv
├── train/
│   ├── ldaps_train.csv
│   ├── gfs_train.csv
│   ├── train_labels.csv
│   ├── scada_vestas_train.csv
│   └── scada_unison_train.csv
└── test/
    ├── ldaps_test.csv
    └── gfs_test.csv
```

컬럼 설명은 `docs/데이터 변수 구조.md`, 실제 사용 변수와 실험 기록은 `docs/exp_logs.md`를 참고하면 됩니다.

## 환경 설치

권장 conda 환경 이름은 `WindForecast`입니다.

```bash
conda env create -f environment.yml
conda activate WindForecast
```

이미 환경이 있다면 바로 아래처럼 실행하면 됩니다.

```bash
conda run -n WindForecast python --version
```

`torch` 설치가 CUDA 버전 문제로 실패하면 아래 중 하나를 사용합니다.

```bash
# CUDA 12.4 사용 가능 시
conda run -n WindForecast pip install torch --index-url https://download.pytorch.org/whl/cu124

# GPU/CUDA를 안 쓸 때
conda run -n WindForecast pip install torch
```

## 처음 보는 사람용 실행 순서

### 1. 기존 tree-only 제출 파일 만들기

가장 안정적인 기존 제출 경로입니다. 실제 리더보드 점수는 0.6087이었습니다.

```bash
conda run -n WindForecast python predict.py
```

출력:

```text
results/submission.csv
```

이 경로는 다음 순서로 동작합니다.

```text
LDAPS/GFS 예보
→ 바람 피처 생성
→ SCADA 기반 경험적 power curve 피처 추가
→ group별 RF/LGBM/XGB 학습
→ 세 모델 단순평균
→ 3개 group OOF 예측을 pooled isotonic으로 보정
→ 0~설비용량 범위로 clamp
→ submission.csv 저장
```

### 2. 현재 최고 검증 조합 재현하기

먼저 공통 recipe인 **PINN 70% + calibrated tree 30%**를 재현합니다.

```bash
# 1) PINN checkpoint 학습
conda run -n WindForecast python train_pinn.py

# 2) PINN과 tree ensemble을 2024 holdout에서 블렌드 평가
conda run -n WindForecast python evaluate_pinn_tree_blend.py
```

출력:

```text
results/pinn_vestas_stage1.pt
results/pinn_vestas_stage2.pt
results/pinn_unison_stage1.pt
results/pinn_unison_stage2.pt
results/pinn_kpx_group_1_bias.pt
results/pinn_kpx_group_2_bias.pt
results/pinn_kpx_group_3_bias.pt
results/pinn_tree_blend_scores.csv
```

`evaluate_pinn_tree_blend.py`는 weight를 0.00~1.00까지 0.05 간격으로 바꿔가며 평가합니다.

현재 최고:

```text
tree_variant = calibrated
pinn_weight  = 0.70
tree_weight  = 0.30
mean score   = 0.6245
```

그 다음 group_3만 VESTAS/group_2 transfer 후보와 섞습니다.

```bash
conda run -n WindForecast python evaluate_group3_transfer_blend.py
conda run -n WindForecast python evaluate_group3_pinn_teacher_transfer.py
conda run -n WindForecast python evaluate_final_tree_ensemble.py
```

현재 group별 최고:

```text
group_1 score = 0.6357
group_2 score = 0.6530  # current group2 recipe 85% + raw tree 15%
group_3 score = 0.6034  # current group3 recipe 90% + own calibrated tree 10%
mean          = 0.6307
```

### 3. 주요 검증 스크립트

| 스크립트 | 목적 | 보통 언제 실행하나 |
|---|---|---|
| `evaluate_time_holdout.py` | 2022~2023 train, 2024 validation으로 tree 모델 기본 성능 확인 | 피처 변경 후 baseline 확인 |
| `calibrate_and_evaluate.py` | tree ensemble + isotonic 보정 효과 확인 | 후보정 변경 시 |
| `train_pinn.py` | SCADA teacher + PINN checkpoint 학습 | PINN 구조/하이퍼파라미터 변경 시 |
| `evaluate_pinn_tree_blend.py` | PINN과 tree ensemble의 최적 블렌드 weight 확인 | 최고 조합 확인 시 |
| `evaluate_group3_transfer_blend.py` | group_3에 VESTAS/group_2 transfer를 섞는 전용 실험 | group_3 병목 개선 시 |
| `evaluate_group3_pinn_teacher_transfer.py` | group_3 PINN teacher 자체를 UNISON/VESTAS/mixed로 바꾸는 실험 | group_3 PINN 입력 개선 시 |
| `evaluate_final_tree_ensemble.py` | 현재 최고 recipe에 tree계열을 추가로 섞는 실험 | 마지막 앙상블 weight 확인 시 |
| `evaluate_scada_teacher_time_holdout.py` | tree 모델에 SCADA teacher 피처를 직접 추가한 실험 | SCADA teacher feature 실험 시 |
| `calibrate_scada_teacher_time_holdout.py` | SCADA teacher feature tree 모델의 isotonic 보정 확인 | 위 실험의 보정 확인 시 |
| `diagnose_pinn_bias.py` | PINN bias가 실제 잔차 규모를 흡수하는지 확인 | bias 구조 변경 시 |
| `diagnose_pinn_physics.py` | 물리 출력, 음수 예측, clamp 필요성 확인 | 물리식 변경 시 |

## 작동 방식

### Tree baseline

Tree baseline은 예보 데이터를 표 형태 피처로 만들고, group별로 세 모델을 학습합니다.

사용 모델:

```python
RandomForestRegressor(random_state=42, n_jobs=-1)
LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
XGBRegressor(random_state=42, n_jobs=-1)
```

핵심 피처:

- LDAPS/GFS 바람 관련 변수
- 풍속 파생 피처
- 시간 주기 인코딩
- SCADA 풍속-발전량으로 만든 group별 경험적 power curve 피처

검증과 보정:

- 검증은 `2024-01-01 01:00:00` 이후를 validation으로 둡니다.
- random K-fold는 미래 연도 일반화를 잘 못 봐서 최종 판단 기준에서 제외했습니다.
- isotonic 보정은 세 group의 OOF 예측을 설비용량 비율로 정규화해서 한 번에 학습합니다.

### PINN

PINN은 발전량을 바로 맞추는 순수 블랙박스가 아니라, 먼저 현장 풍속분포를 복원하고 물리식으로 발전량을 계산합니다.

전체 흐름:

```text
LDAPS/GFS forecast
→ SCADA teacher가 현장 풍속분포 예측
   - scada_ws_mean
   - scada_ws_std
   - scada_ws_p10/p50/p90
→ PINN 입력 v, v_std 구성
→ P = 0.5 * rho * A * v^3 * C_eff(v, day, month)
→ 풍속분포 적분
→ 터빈 응답 지연 EMA 적용
→ group별 HOD bias 적용
→ 평가/제출에서만 0~capacity clamp
```

중요한 점:

- SCADA를 test에 직접 넣는 것이 아닙니다.
- train SCADA로 `forecast → site wind distribution` teacher를 학습합니다.
- honest 검증에서는 teacher도 2024 SCADA를 보지 않습니다.
- 현재 성능 상승의 핵심은 물리식 자체보다 `forecast → 실제 현장 풍속분포 복원`입니다.

### SCADA teacher

SCADA teacher는 forecast weather feature로 SCADA 풍속 통계를 예측하는 보조 모델입니다.

teacher target:

```text
scada_ws_mean
scada_ws_std
scada_ws_p10
scada_ws_p50
scada_ws_p90
```

teacher model:

```python
MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
)
```

PINN 입력에서 사용하는 값:

```text
v     = predicted scada_ws_mean
v_std = 0.5 * predicted scada_ws_std
      + 0.5 * (predicted scada_ws_p90 - predicted scada_ws_p10) / 2.563
```

## 하이퍼파라미터

### Tree baseline

현재 tree 계열은 **튜닝하지 않은 기본값**을 사용합니다.

```python
RandomForestRegressor(random_state=42, n_jobs=-1)
LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)
XGBRegressor(random_state=42, n_jobs=-1)
```

OOF/isotonic용 CV:

```python
KFold(n_splits=3, shuffle=True, random_state=42)
IsotonicRegression(out_of_bounds="clip")
```

왜 기본값을 쓰는가:

- `tune.py`의 random K-fold 튜닝은 CV 점수는 좋아졌지만 실제 time-holdout/리더보드에서는 이득이 없었습니다.
- 이 프로젝트는 시간 일반화가 중요해서, 튜닝 점수보다 2024 holdout을 더 믿습니다.

### PINN training

현재 `train_pinn.py`의 주요 설정:

```python
VAL_START = "2024-01-01 01:00:00"

STAGE1_EPOCHS = 500
STAGE2_EPOCHS = 2000
LR = 1e-3
BIAS_LR = 1e-3

N_COLLOCATION = 2000
GAMMA = 0.039709016191988696

USE_SCADA_WIND_TEACHER = True
HONEST_SCADA_TEACHER_HOLDOUT = True

USE_WIND_DISTRIBUTION = {
    "vestas": True,
    "unison": True,
}

USE_MOY_BIAS = False
USE_TRAIN_ONLY_HOUR_BIAS = False
USE_TRAIN_ONLY_YEAR_BIAS = False
```

PINN loss weight:

```python
LAMBDA = {
    "betz": 2.026253819683706,
    "bc": 0.004598334508698543,
    "flat": 0.2930203005161716,
    "smooth": 0.004331969897949324,
    "hod": 0.001,
    "moy": 0.001,
    "hour": 0.01,
    "year": 0.01,
}
```

학습은 2단계입니다.

1. Stage 1: 제조사별 물리 backbone 학습
   - VESTAS와 UNISON 각각 하나의 `PowerCurvePINN`을 학습합니다.
   - `C_eff(v, doy, moy)`, 풍속분포 적분, EMA response를 학습합니다.

2. Stage 2: 물리 backbone freeze 후 group별 bias만 학습
   - 현재 기본은 hour-of-day bias만 사용합니다.
   - bias는 kWh 직접값이 아니라 설비용량 비율로 parameterize합니다.
   - `moy`, train-only hour, train-only year bias는 검증에서 악화되어 꺼져 있습니다.

### PINN 물리 파라미터

모델 구조:

```text
C_eff(v, doy, moy) = C_all(v) + g_doy(doy) + g_moy(moy)
P_phys             = 0.5 * rho * A * v^3 * C_eff
P_output           = EMA(P_phys, tau)
```

기본 구조:

- `C_all`: hidden size 32의 작은 MLP
- `g_doy`: 365일 주기 harmonic correction
- `g_moy`: 12개월 주기 harmonic correction
- `tau`: softplus로 양수 제약을 둔 학습 가능한 응답 시간
- `EMA_WINDOW = 24`
- wind distribution 적분은 5-point Gauss-Hermite quadrature 사용

## Clamp 정책

학습 중에는 예측값을 clamp하지 않습니다.

이유:

- 학습 중 clamp를 걸면 음수/초과 구간에서 gradient가 죽을 수 있습니다.
- 특히 bias가 붙은 뒤에는 낮은 풍속에서 음수 예측이 생길 수 있으므로, 모델이 그 문제를 loss로 보게 두는 편이 낫습니다.

대신 평가/제출에서만 clamp합니다.

```text
prediction = clip(prediction, 0, group_capacity)
```

## 결과 파일

| 파일 | 의미 |
|---|---|
| `results/submission.csv` | `predict.py`가 만드는 기존 tree-only 제출 파일 |
| `results/calibration_comparison.csv` | tree isotonic 보정 검증 결과 |
| `results/pinn_*_stage1.pt` | PINN stage 1 checkpoint |
| `results/pinn_*_stage2.pt` | PINN stage 2 checkpoint |
| `results/pinn_kpx_group_*_bias.pt` | group별 PINN bias checkpoint |
| `results/pinn_tree_blend_scores.csv` | PINN/tree weight별 2024 holdout 점수 |

## 현재 결론

- 단순 tree ensemble은 안정적이지만 피크 발전량에서 평균으로 수렴하는 경향이 있습니다.
- PINN은 SCADA teacher가 만든 현장 풍속분포를 입력으로 쓰면서 큰 폭으로 개선됐습니다.
- SCADA teacher feature를 tree 모델에 그냥 추가하는 방식은 기대보다 약했습니다.
- group_1/group_2는 **PINN을 메인으로 두고 calibrated tree ensemble을 30% 섞는 것**이 강합니다.
- group_3는 own tree보다 **VESTAS/group_2 teacher를 PINN 입력에 일부 섞고, group_2 tree transfer를 후단에 한 번 더 섞는 것**이 더 강합니다.

다음 개발 우선순위:

1. `predict.py` 또는 별도 `predict_pinn_tree_blend.py`에 group별 recipe 제출 경로 구현
2. SCADA teacher ensemble/target 확장으로 forecast → site wind reconstruction 강화
3. group_3 adaptive VESTAS prior: 시간대/풍속/계절별로 UNISON/VESTAS teacher mixing weight 조정
4. LDAPS/GFS 격자 선택, terrain-aware wind feature, SCADA 품질 필터링 재검토
5. 단순 평균 대신 residual stacking 또는 group별 weight 탐색

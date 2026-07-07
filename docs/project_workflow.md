# Project Workflow

작성일: 2026-07-08 00:46:07 +09:00

## 현재 모델 구조

현재 메인 모델은 세 블록만 유지한다.

```text
PINN prediction
TREE prediction
-> PINN50:TREE50 blend
-> submission
```

### 1. PINN

목적:

- 물리식 기반 예측.
- 피크/FICR 쪽 보완.

주요 파일:

- `predict_pinn_effective_grid_g1_year_bagging.py`
- `train_pinn.py`
- `utils/pinn_effective_pipeline.py`

출력:

- `results/submission_pinn_effective_grid_g1_year_bagging.csv`

### 2. TREE

목적:

- compact weather, meteo, physics feature 기반 안정적 예측.
- 현재 다음 개선 우선순위는 tree hyperparameter optimization.

주요 파일:

- `predict_tree_compact_v2_metric_valid_lgbm_mean.py`
- `predict_tree_compact_physics_v2.py`
- `utils/compact_physics_features.py`
- `utils/meteo_features.py`

출력:

- `results/submission_tree_compact_v2_metric_valid_lgbm_mean.csv`

### 3. PINN50:TREE50

목적:

- PINN과 TREE를 동일 비율로 섞어 최종 안정화.
- 당분간 일관성을 위해 blend ratio는 기본적으로 `50:50`으로 유지한다.

주요 파일:

- `blend_submission_files.py`

출력:

- `results/submission.csv`

## 작업 원칙

### 1. 실험 파일 관리

불필요한 실험 파일은 메인 루트에 두지 않는다.

원칙:

- 메인 파이프라인 파일은 루트 또는 `utils/`에 둔다.
- 검증용 핵심 스크립트만 `experiments/`에 둔다.
- 일회성/진단/실패 실험은 남기지 않는다.
- 꼭 보존해야 하는 경우 별도 archive 폴더나 문서 요약으로만 남긴다.
- 사용하지 않는 결과 CSV/PNG/PT는 삭제한다.

### 2. Test Submission 생성 지양

큰 변화가 없으면 test submission을 만들지 않는다.

test 파일을 만들어도 되는 경우:

- validation에서 의미 있는 개선이 확인됨.
- 대략 `+0.01` 이상 개선 후보.
- 유저가 명시적으로 test/submission 생성을 요청함.

그 외에는 validation 결과와 로그만 남긴다.

### 3. 유저와 상호작용 우선

큰 실험은 독단적으로 오래 돌리지 않는다.

원칙:

- 코드 작성 전 파이프라인을 먼저 설명한다.
- 실험 의도, 입력, 출력, 검증 방식을 유저에게 직관적으로 공유한다.
- test submission 생성, 대규모 sweep, 구조 변경은 유저 확인 후 진행한다.
- 단순 문법/import 체크처럼 짧고 안전한 검증은 바로 수행해도 된다.

### 4. Exp Logs 작성 방식

`docs/exp_logs.md`는 짧고 읽기 쉬워야 한다.

필수 형식:

```text
작성일: YYYY-MM-DD HH:mm:ss +09:00
실험명:
목적:
결과:
판단:
다음 액션:
```

작성 원칙:

- 장문 설명 금지.
- 핵심 숫자만 표로 정리.
- 실패 실험은 실패 이유와 결론만 남긴다.
- 이미 버린 실험의 세부 구현 설명은 남기지 않는다.
- 현재 메인 모델과 무관한 내용은 archive/summary 수준으로만 기록한다.

### 5. 코드 작성 전 파이프라인 설명

코드를 짜기 전에 먼저 아래를 정리한다.

```text
1. 무엇을 바꾸는가
2. 어떤 데이터/피처를 쓰는가
3. 학습 구조는 어떻게 되는가
4. validation은 어떻게 보는가
5. 결과 파일은 무엇을 만들 것인가
6. test submission을 만들지 여부
```

설명은 처음 보는 사람도 바로 이해할 수 있게 한다.

나쁜 설명:

```text
teacher 붙이고 residual calibration 돌림
```

좋은 설명:

```text
TREE 모델의 LGBM 하이퍼파라미터만 바꾼다.
입력 feature는 기존 compact v2 그대로 둔다.
2022/2023 -> 2024, 2022/2024 -> 2023, 2023/2024 -> 2022 year-fold로 본다.
submission은 만들지 않고 validation summary만 저장한다.
```


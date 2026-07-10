# WindForecast Agent Instructions

이 파일은 이 레포에서 작업하는 Codex/에이전트가 먼저 읽어야 하는 기본 지침이다. 상세 인수인계는 `.agents/windforecast_agent_context.md`와 `docs/best_model_usage.md`를 기준으로 한다.

## Must Read First

1. `.agents/windforecast_agent_context.md`
2. `docs/best_model_usage.md`
3. `docs/rules.md`
4. `docs/exp_logs.md` 최근 항목

## Current Best Model

현재 최고 public 제출은 아래 파일이다.

```text
results/submission_pinnfloor350_pinn25_tree20_tcn55_weightedl1_finalfloor10_v1.csv
```

Public score:

```text
score  = 0.6386205415
1-nMAE = 0.8682636645
FiCR   = 0.4089774184
time   = 2026-07-11 01:21:21 KST
id     = 1484025
```

구조:

```text
TCN_family = 0.30 * TCN_W24 + 0.40 * TCN_W72 + 0.30 * TCN_W168
PINN_floor = clip(PINN, 0.35 * capacity, capacity)
final_raw  = 0.25 * PINN_floor + 0.20 * TREE + 0.55 * TCN_family
final      = clip(final_raw, 0.10 * capacity, capacity)
```

Branch files:

```text
PINN    = results/submission_pinn_lgbm_teacher_year_bagging.csv
TREE    = results/submission_tree_lgbm_best_v2_l1.csv
TCN W24 = results/submission_seqnn_short_tcn_w24_v1.csv
TCN W72 = results/submission_seqnn_mid_tcn_w72_v1.csv
TCN W168= results/submission_seqnn_long_tcn_w168_v1.csv
```

주의:

- `results/submission.csv`는 임시 파일이다. 최고 모델로 간주하지 않는다.
- 현재 최고는 구버전 `PINN50 + TREE50`이 아니다.
- 현재 최고는 이전 `PINN25 + TREE40 + TCN35 + group3 pseudo2022`도 아니다.
- 현재 최고의 TREE branch는 `submission_tree_lgbm_best_v2_l1.csv`이다. group3 pseudo2022 TREE가 아니다.
- 최종 weight, PINN floor, final floor는 사용자가 허락하지 않으면 임의로 바꾸지 않는다.
- 상세 재현 명령과 시간별 best timeline은 `docs/best_model_usage.md`를 따른다.

## Collaboration Rules

- 실험이나 큰 코드 변경 전에 목적, 파이프라인, 기대 효과, 수정 파일, validation 방식, 예상 실행 시간, 결과 파일명을 사용자에게 먼저 설명한다.
- 사용자가 명시하지 않으면 test submission을 만들지 않는다.
- 작은 OOF 개선만으로 제출 후보를 만들지 않는다. 큰 개선 또는 사용자 명시 요청이 필요하다.
- 결과를 말할 때는 OOF인지 public인지, 파일명이 무엇인지 명확히 말한다.
- exp log는 짧고 직관적으로 남긴다. 긴 구조 설명은 docs로 분리한다.
- 임의로 실험을 계속 이어서 결론짓지 않는다. 다음 실험으로 넘어가기 전에 사용자와 확인한다.
- 기존 파일을 정리하거나 삭제하기 전에는 사용자 의도를 확인한다.
- teacher-style feature에서 LGBM teacher를 다시 쓰지 않는다. 사용자가 명시적으로 허락하지 않으면 RF OOB, empirical table, 단순 회귀 같은 가벼운 방식만 쓴다.

## Competition Rules

- 예측값에는 해당 행의 예측기준시점 이전에 생성/공개/확정된 정보만 사용할 수 있다.
- 예측기준시점 이후 관측값, same-time AWS, 사후 보정자료, 재분석자료를 final/test 입력으로 쓰면 안 된다.
- 평가 데이터셋을 학습 데이터로 쓰는 test-time adaptation/pseudo-labeling은 금지 소지가 있으므로 하지 않는다.
- 외부 데이터는 공개 데이터, 라이선스, 수집 시점, 재현 가능성을 소명할 수 있어야 한다.
- 규칙 판단은 항상 `docs/rules.md`를 우선한다.

## Environment

권장 실행 환경:

```powershell
conda run -n WindForecast python --version
```

긴 학습/추론을 실행하기 전에는 어떤 branch(PINN/TREE/TCN)를 건드리는지 사용자에게 먼저 말한다.

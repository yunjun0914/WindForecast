# WindForecast Agent Instructions

이 파일은 이 레포에서 작업하는 Codex/에이전트가 먼저 읽어야 하는 기본 지침이다. 상세 인수인계는 `.agents/windforecast_agent_context.md`를 기준으로 한다.

## Must Read First

1. `.agents/windforecast_agent_context.md`
2. `docs/best_model_usage.md` (과거 모델 기록)
3. `docs/rules.md`
4. `docs/exp_logs.md` 최근 항목
5. `docs/source_expert_pipeline_plan.md` (현재 활성 계획)

## Current Best Model

현재 최고 public 제출은 아래 파일이다.

```text
results/submission_jointmix_p50_t5_c45_pb50_cb25_v1.csv
```

Public score:

```text
score = 0.63999 (user-reported public score)
time  = 2026-07-14 KST
```

구조:

```text
PINN_base_share = 0.50
TCN_base_share  = 0.25
final_raw       = 0.50 * PINN + 0.05 * TREE + 0.45 * TCN
PINN_floor      = 0.20 * capacity
final_floor     = 0.10 * capacity
```

주의:

- `results/`는 실험 산출물 디렉터리이므로 Git에서 기본적으로 무시한다.
- 사용자 승인을 받은 최종 제출 CSV만 `git add -f <path>`로 명시적으로 추가한다.
- OOF prediction, score, summary, diagnostics, log, archive, cache는 Git에 올리지 않는다.
- `submission_share50_v1.csv`의 public score는 `0.639938`로 현재 최고보다 낮다.
- 최종 weight, PINN floor, final floor는 사용자가 허락하지 않으면 임의로 바꾸지 않는다.

## Collaboration Rules

- 실험이나 큰 코드 변경 전에 목적, 파이프라인, 기대 효과, 수정 파일, validation 방식, 예상 실행 시간, 결과 파일명을 사용자에게 먼저 설명한다.
- 사용자가 승인한 실험 구성까지만 실행하고 결과가 나오면 즉시 멈춰 보고한다. 결과가 나쁘다는 이유로 weight, alpha, feature subset, loss, model size, seed, horizon, floor, clip, 후처리 또는 추가 앙상블을 임의 탐색하지 않는다.
- 성능이 나오지 않으면 실패한 정확한 구조, 사용 데이터와 피처, 학습/validation 방식, OOF 결과 파일과 metric, 실패 해석을 그대로 보고한다. 작은 사후 튜닝으로 개선처럼 포장하지 않는다.
- 다음 실험과 blend/grid search는 계획 문서에 이미 있더라도 실행 직전에 구체적인 탐색 범위와 비용을 설명하고 사용자 승인을 다시 받는다.
- 사용자가 명시하지 않으면 test submission을 만들지 않는다.
- 작은 OOF 개선만으로 제출 후보를 만들지 않는다. 큰 개선 또는 사용자 명시 요청이 필요하다.
- 결과를 말할 때는 OOF인지 public인지, 파일명이 무엇인지 명확히 말한다.
- OOF 모델 선택 점수는 outer-fold 예측을 모두 이어 붙인 뒤 group별 metric을 한 번만 계산하고, 3개 group을 동일 가중한다. 연도별 평균은 안정성 진단에만 쓴다.
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

# WindForecast 에이전트 지침

이 파일은 이 레포에서 작업하는 Codex/에이전트가 먼저 읽어야 하는 기본 지침이다.

## 먼저 읽을 문서

1. `docs/rules.md`
2. `docs/project_workflow.md`

## 모델 및 제출 기록

- 현재 최고 모델, public 점수, 제출 파일, 앙상블 비율을 이 문서나 일반 문서에서 고정된 사실로 관리하지 않는다.
- 파일명, 과거 문서, 과거 실험 로그만 보고 현재 최고 모델을 추정하지 않는다.
- 최고 기록이나 제출 구조가 필요한 작업은 사용자에게 확인하거나 사용자가 제공한 리더보드 기록과 실제 파일을 직접 대조한다.
- 사용자가 명시적으로 기록을 허락한 경우에만 허락받은 범위의 제출 정보를 문서에 남긴다.
- `results/`는 실험 산출물 디렉터리이므로 Git에서 기본적으로 무시한다.
- 사용자 승인을 받은 최종 제출 CSV만 `git add -f <path>`로 명시적으로 추가한다.
- OOF prediction, score, summary, diagnostics, log, archive, cache는 Git에 올리지 않는다.

## 협업 규칙

- 실험이나 큰 코드 변경 전에 목적, 파이프라인, 기대 효과, 수정 파일, validation 방식, 예상 실행 시간, 결과 파일명을 사용자에게 먼저 설명한다.
- 사용자가 명시하지 않으면 test submission을 만들지 않는다.
- 작은 OOF 개선만으로 제출 후보를 만들지 않는다. 큰 개선 또는 사용자 명시 요청이 필요하다.
- 결과를 말할 때는 OOF인지 public인지, 파일명이 무엇인지 명확히 말한다.
- OOF 모델 선택 점수는 outer-fold 예측을 모두 이어 붙인 뒤 group별 metric을 한 번만 계산하고, 3개 group을 동일 가중한다. 연도별 평균은 안정성 진단에만 쓴다.
- 임의로 실험을 계속 이어서 결론짓지 않는다. 다음 실험으로 넘어가기 전에 사용자와 확인한다.
- 기존 파일을 정리하거나 삭제하기 전에는 사용자 의도를 확인한다.
- teacher-style feature에서 LGBM teacher를 다시 쓰지 않는다. 사용자가 명시적으로 허락하지 않으면 RF OOB, empirical table, 단순 회귀 같은 가벼운 방식만 쓴다.
- 모든 문서 작성과 수정은 한국어로 한다.

## 실험 로그 규칙

- 사용자가 명시적으로 허락했을 때만 experiment log를 생성하거나 수정한다.
- 사용자가 허락한 실험과 허락한 내용만 기록한다.
- 사용자가 요청하지 않은 해석, 다음 실험 제안, 최고모델 선언, 장황한 구현 설명을 log에 추가하지 않는다.
- log 파일이 삭제되었거나 존재하지 않으면 사용자 지시 없이 다시 만들지 않는다.
- 단순히 실험을 실행했다는 이유만으로 자동 기록하지 않는다.

## 검증 규칙

- Validation으로 모든 fold가 공유할 하나의 하이퍼파라미터 세트를 선택한다.
- 선택된 하이퍼파라미터로 각 fold의 best epoch를 측정하고 그 중앙값을 최종 공통 epoch로 고정한다.
- 중앙값 epoch가 정해지면 모든 outer fold를 동일한 하이퍼파라미터와 동일한 epoch로 다시 학습해 fixed-epoch OOF를 계산한다.
- fold별 best checkpoint를 최종 OOF 예측이나 submission에 직접 사용하지 않는다.
- Submission은 모든 사용 가능한 학습 연도를 합쳐 공통 하이퍼파라미터와 중앙값 epoch로 새 모델을 학습해 만든다.

## 대회 규칙

- 예측값에는 해당 행의 예측기준시점 이전에 생성/공개/확정된 정보만 사용할 수 있다.
- 예측기준시점 이후 관측값, same-time AWS, 사후 보정자료, 재분석자료를 final/test 입력으로 쓰면 안 된다.
- 평가 데이터셋을 학습 데이터로 쓰는 test-time adaptation/pseudo-labeling은 금지 소지가 있으므로 하지 않는다.
- 외부 데이터는 공개 데이터, 라이선스, 수집 시점, 재현 가능성을 소명할 수 있어야 한다.
- 규칙 판단은 항상 `docs/rules.md`를 우선한다.

## 실행 환경

권장 실행 환경:

```powershell
conda run -n WindForecast python --version
```

`bear` 서버의 재현 가능한 실행 환경:

```bash
/home/yunjun0914/WindForecast_env/bin/python --version
```

- Python `3.10.12`, PyTorch `2.6.0+cu124`, CUDA 사용 가능
- `WindForecast_env`는 독립 venv이며 과거 micromamba 경로를 사용하지 않는다.
- 서버 실험은 `/home/yunjun0914/windforecast_runs/<run_name>` 아래 격리 run directory에서 실행한다.

긴 학습/추론을 실행하기 전에는 어떤 branch(PINN/TREE/TCN)를 건드리는지 사용자에게 먼저 말한다.

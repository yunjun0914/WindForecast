# WindForecast Agent Context

작성일: 2026-07-10 KST

이 문서는 새 세션/다른 에이전트가 현재 프로젝트 상태를 빠르게 이해하기 위한 기본 컨텍스트다. 작업 시작 전에 `AGENTS.md`, 이 문서, `docs/best_model_usage.md`, `docs/rules.md`를 읽는다.

## 1. Current Best

현재 최고 public 제출:

| Item | Value |
|---|---|
| File | `results/submission_pinn25_tree40_tcn35_tree_g3_vestas_pseudo2022_w010.csv` |
| Public score | `0.6370788926` |
| Public 1-nMAE | `0.8701764551` |
| Public FiCR | `0.4039813302` |
| Memo | `pinn25_tree40_tcn35_tree_g3_vestas_pseudo2022` |

`results/submission.csv`는 작업 중 계속 덮어쓰는 임시 파일이다. 최고 성능 파일로 말하면 안 된다.

## 2. Model Pipeline

현재 최고 구조:

```text
final = 0.25 * PINN + 0.40 * TREE + 0.35 * TCN_family
TCN_family = 0.30 * TCN_W24 + 0.40 * TCN_W72 + 0.30 * TCN_W168
```

Branch별 기준 파일:

| Branch | File |
|---|---|
| PINN | `results/submission_pinn_lgbm_teacher_year_bagging_stage2_es.csv` |
| TREE | `results/submission_tree_lgbm_best_v2_l1_aggressive_minimal_rolling_v1_g3_vestas_pseudo2022_w010.csv` |
| TCN W24 | `results/submission_seqnn_short_tcn_w24_v1.csv` |
| TCN W72 | `results/submission_seqnn_mid_tcn_w72_v1.csv` |
| TCN W168 | `results/submission_seqnn_long_tcn_w168_v1.csv` |

현재 최고 public에서는 `soft_metric TCN`이 아니라 기존 `weighted_l1` TCN family를 사용한다.

## 3. Branch Notes

PINN:

- SCADA teacher/effective wind 기반 물리 모델이다.
- `lgbm_time_oof` teacher backend가 기준이다.
- year-bagging 구조이며 test는 leave-one-year 모델들의 평균이다.
- raw SCADA를 test처럼 직접 넣지 않는다. weather 기반 teacher 예측값을 사용한다.
- early stopping은 기본 활성화다.

TREE:

- tuned LGBM tabular model이다.
- feature profile 기준은 `aggressive_minimal_rolling_v1`이다.
- 하이퍼파라미터 기준 파일은 `results/power_lgbm_hyperparams_v2_l1_20_best.csv`다.
- group3는 VESTAS transfer + pseudo2022 weight `0.10`으로 교체한 버전이 현재 최고 public에 쓰였다.

TCN:

- W24/W72/W168 세 window를 쓴다.
- TCN family 내부 weight는 `0.30 / 0.40 / 0.30`이다.
- 기본 loss는 `weighted_l1`, weight policy는 `actual_sqrt`다.
- soft metric loss는 OOF에서 좋아 보였지만 public 최고를 넘지 못해 현재 최고 재현에는 쓰지 않는다.

## 4. Rebuild Command

이미 branch 파일이 있으면 아래 명령으로 현재 최고 구조를 재조합한다.

```powershell
conda run -n WindForecast python experiments\blend_three_branch_submission.py `
  --pinn results\submission_pinn_lgbm_teacher_year_bagging_stage2_es.csv `
  --tree results\submission_tree_lgbm_best_v2_l1_aggressive_minimal_rolling_v1_g3_vestas_pseudo2022_w010.csv `
  --tcn24 results\submission_seqnn_short_tcn_w24_v1.csv `
  --tcn72 results\submission_seqnn_mid_tcn_w72_v1.csv `
  --tcn168 results\submission_seqnn_long_tcn_w168_v1.csv `
  --pinn-weight 0.25 `
  --tree-weight 0.40 `
  --tcn-family-weight 0.35 `
  --tcn24-weight 0.30 `
  --tcn72-weight 0.40 `
  --tcn168-weight 0.30 `
  --output results\submission_pinn25_tree40_tcn35_tree_g3_vestas_pseudo2022_w010_rebuild.csv
```

`--also-update-submission-csv`는 사용자가 명시했을 때만 붙인다.

## 5. Validation Philosophy

기본 검증은 leave-one-year-out OOF다.

| Fold | Train years | Predict year |
|---|---|---|
| 1 | `2022, 2023` | `2024` |
| 2 | `2022, 2024` | `2023` |
| 3 | `2023, 2024` | `2022` |

원칙:

- 검증 연도 raw SCADA를 teacher 입력으로 직접 쓰지 않는다.
- teacher feature는 train row도 OOF/crossfit 예측값으로 만든다.
- test는 각 leave-one-year model의 예측을 평균한다.
- 최종 예측은 group capacity 범위로 clamp한다.
- OOF 상승이 public 상승으로 항상 이어지지 않았다. 작은 OOF 개선만으로 test submission을 만들지 않는다.

## 6. Competition Rule Summary

대회 규칙은 `docs/rules.md`가 원본 기준이다. 핵심은 아래와 같다.

- 각 예측값에는 그 행의 예측기준시점 이전에 실제로 사용 가능했던 정보만 사용할 수 있다.
- 정보 사용 가능 여부는 대상 시각이 아니라 생성/공개/확정 시각 기준이다.
- 예측기준시점 이후 관측값, same-time AWS, 사후 보정자료, 재분석자료는 final/test 입력으로 금지다.
- 평가 데이터셋은 제출 파일 생성을 위한 추론 목적으로만 사용한다.
- 2025 test weather를 학습 데이터로 넣는 test-time adaptation/pseudo-labeling은 금지 소지가 있어 하지 않는다.
- 외부 데이터는 공개 데이터, 라이선스, 수집 시점, 사용 기간, 전처리 코드를 소명할 수 있어야 한다.

## 7. Behavior Rules

사용자가 가장 중요하게 보는 작업 방식:

1. 실험 전 목적, 파이프라인, 기대 효과를 먼저 설명한다.
2. 사용자가 허락하지 않으면 최종 ensemble weight를 바꾸지 않는다.
3. 실험을 여러 개 연속으로 혼자 돌리고 결론짓지 않는다.
4. 제출 파일은 큰 검증 개선이 있거나 사용자가 명시했을 때만 만든다.
5. public 결과와 OOF 결과를 섞어 말하지 않는다.
6. 결과 파일명, 점수, 사용 구조를 함께 말한다.
7. exp log는 짧게 남긴다.
8. 필요 없는 실험 파일/결과는 사용자 동의 없이 삭제하지 않는다.

## 8. Good Next-Step Areas

현재 gap은 상위권 대비 FiCR이 특히 크다. 큰 상승 후보는 아래 계열이다.

- data/time alignment audit: `forecast_kst_dtm`, `data_available_kst_dtm`, lead time 정렬 재확인.
- weather -> turbine/site wind 복원: 풍향, 공간장, forecast lead, hub-height proxy 개선.
- group3 전용 개선: 2022 target 부재와 제조사 차이 보완.
- FiCR-oriented modeling: public에서 깨지지 않는 방식만 신중히 검토.
- 외부 데이터: `docs/rules.md`를 지키는 base-time 관측/공개 데이터만 사용.

실험으로 들어가기 전 반드시 사용자에게 어떤 branch와 어떤 데이터 흐름을 바꾸는지 설명한다.


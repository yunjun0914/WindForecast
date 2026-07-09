# SeqNN Short GRU W24 V1

작성일: 2026-07-09 03:28:12 +09:00

목적: TREE가 rolling feature로 간접적으로 보는 시간 문맥을 GRU가 최근 24시간 weather sequence로 직접 볼 때, 단독 성능과 TREE와의 blend diversity가 있는지 확인한다.

## 고정점

- 기존 PINN/TREE 제출 경로는 변경하지 않음
- test submission 생성하지 않음
- validation: leave-one-year-out OOF
- target history 사용하지 않음
- raw SCADA 직접 feature 사용하지 않음
- early stopping 사용

## 모델

| 항목 | 값 |
|---|---|
| model name | `seqnn_short_gru_w24_v1` |
| family | `seqnn` |
| window | `24` |
| input features | `34` |
| model | GRU, hidden `64`, layer `1` |
| loss | capacity-normalized weighted L1 |
| weight policy | `actual_sqrt` |
| early stopping | validation official score |
| device | CUDA 사용 확인 |

## 입력 Feature Family

- calendar: `sin/cos doy`, `sin/cos hod`
- core wind: GFS 100m/850hPa/10m, LDAPS 50m/10m
- gust/ramp proxy: gust, gust factor, gust minus ws10
- physics: `rho * ws^3`, shear
- spatial compact: GFS 850/100 grid, LDAPS 50m grid
- forecast cycle: lead hour, lead 24h sin/cos, available hour sin/cos
- direction compact: selected normalized direction sin/cos

## 단독 OOF 결과

| Model | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---:|---:|---:|---:|
| TREE rolling baseline | `0.62396` | `0.12809` | `0.37602` | `0.60563` |
| SeqNN short GRU W24 | `0.61323` | `0.13700` | `0.36347` | `0.59955` |

단독 모델은 TREE보다 낮다. 단독 제출 후보는 아니다.

## TREE + SeqNN OOF Blend

base는 `aggressive_minimal_rolling_v1` TREE, extra는 `seqnn_short_gru_w24_v1`이다.

| Variant | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---:|---:|---:|---:|
| TREE base | `0.62396` | `0.12809` | `0.37602` | `0.60563` |
| TREE 70% + SeqNN 30% | `0.62768` | `0.12571` | `0.38107` | `0.60973` |
| SeqNN only | `0.61323` | `0.13700` | `0.36347` | `0.59955` |

OOF blend gain:

- score: `+0.00372`
- nMAE: `0.12809 -> 0.12571`
- FICR: `0.37602 -> 0.38107`
- worst fold: `0.60563 -> 0.60973`

## Group 평균

| Variant | Group | Score | nMAE | FICR |
|---|---|---:|---:|---:|
| TREE base | group1 | `0.62184` | `0.12461` | `0.36830` |
| TREE base | group2 | `0.65099` | `0.12198` | `0.42397` |
| TREE base | group3 | `0.57888` | `0.14407` | `0.30182` |
| TREE 70% + SeqNN 30% | group1 | `0.62747` | `0.12109` | `0.37602` |
| TREE 70% + SeqNN 30% | group2 | `0.65433` | `0.12077` | `0.42943` |
| TREE 70% + SeqNN 30% | group3 | `0.58128` | `0.14151` | `0.30407` |

모든 group에서 소폭 개선했다.

## Residual Correlation

| Scope | Corr |
|---|---:|
| overall | `0.86338` |
| group1 | `0.84270` |
| group2 | `0.86142` |
| group3 | `0.89459` |

상관은 낮지 않다. 즉 완전히 다른 모델이라기보다 TREE와 비슷한 방향의 예측을 하면서, 일부 smooth/temporal averaging 효과로 blend가 개선된 것으로 보인다.

## 생성 파일

- `utils/seq_dataset.py`
- `models/seqnn.py`
- `experiments/run_seqnn_oof.py`
- `experiments/evaluate_oof_branch_comparison.py`
- `results/oof_seqnn_short_gru_w24_v1.csv`
- `results/scores_seqnn_short_gru_w24_v1.csv`
- `results/summary_seqnn_short_gru_w24_v1.csv`
- `results/oof_compare_tree_rolling_seqnn_short_gru_w24_v1_scores.csv`
- `results/oof_compare_tree_rolling_seqnn_short_gru_w24_v1_summary.csv`
- `results/oof_compare_tree_rolling_seqnn_short_gru_w24_v1_residual_corr.csv`

## 판단

`seqnn_short_gru_w24_v1`은 단독 모델로는 약하지만, TREE와 섞을 때 OOF 기준 승격 후보 조건인 `+0.003`을 만족했다. 다만 residual correlation이 높으므로 바로 test submission으로 가기보다는 다음을 먼저 확인한다.

1. PINN/TREE/SeqNN 3-branch OOF blend에서 같은 이득이 유지되는가?
2. seed bagging 또는 longer window가 SeqNN 단독 품질을 올리는가?
3. long window DLinear/GRU가 더 낮은 residual correlation을 만드는가?

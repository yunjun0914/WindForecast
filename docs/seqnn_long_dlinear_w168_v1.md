# SeqNN Long DLinear W168 V1

작성일: 2026-07-09 03:35:28 +09:00

목적: 최근 168시간 weather sequence를 보는 long-window model이 TREE와 다른 long-term weather regime 신호를 제공하는지 확인한다.

## 고정점

- 기존 PINN/TREE 제출 경로 변경 없음
- test submission 생성 없음
- validation: leave-one-year-out OOF
- target history 사용하지 않음
- raw SCADA 직접 feature 사용하지 않음
- early stopping 사용

## 모델

| 항목 | 값 |
|---|---|
| model name | `seqnn_long_dlinear_w168_v1` |
| family | `seqnn` |
| window | `168` |
| input features | `34` |
| model | DLinear-like trend/residual head |
| loss | capacity-normalized weighted L1 |
| weight policy | `actual_sqrt` |
| early stopping | validation official score |
| device | CUDA 사용 확인 |

## 단독 OOF 결과

| Model | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---:|---:|---:|---:|
| TREE rolling baseline | `0.62396` | `0.12809` | `0.37602` | `0.60563` |
| SeqNN long DLinear W168 | `0.54953` | `0.17689` | `0.27595` | `0.53806` |

단독 성능이 너무 낮다.

## TREE + DLinear OOF Blend

| Variant | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---:|---:|---:|---:|
| TREE base | `0.62396` | `0.12809` | `0.37602` | `0.60563` |
| TREE 95% + DLinear 5% | `0.62387` | `0.12798` | `0.37572` | `0.60573` |
| DLinear only | `0.54953` | `0.17689` | `0.27595` | `0.53806` |

5%만 섞어도 score가 하락한다. blend 후보가 아니다.

## Residual Correlation

| Scope | Corr |
|---|---:|
| overall | `0.68020` |
| group1 | `0.64812` |
| group2 | `0.68016` |
| group3 | `0.72796` |

short GRU보다 residual correlation은 낮지만, 예측 품질이 너무 낮아 blend diversity로 활용되지 못했다.

## 생성 파일

- `results/oof_seqnn_long_dlinear_w168_v1.csv`
- `results/scores_seqnn_long_dlinear_w168_v1.csv`
- `results/summary_seqnn_long_dlinear_w168_v1.csv`
- `results/oof_compare_tree_rolling_seqnn_long_dlinear_w168_v1_scores.csv`
- `results/oof_compare_tree_rolling_seqnn_long_dlinear_w168_v1_summary.csv`
- `results/oof_compare_tree_rolling_seqnn_long_dlinear_w168_v1_residual_corr.csv`

## 판단

`seqnn_long_dlinear_w168_v1`은 기각한다.

다만 long-window idea 자체를 버리지는 않는다. 다음에 long branch를 다시 본다면:

1. DLinear 대신 GRU/TCN long window
2. 더 작은 learning rate와 stronger weight decay
3. feature set 축소
4. group-shared model + group embedding

위 방향이 더 낫다.

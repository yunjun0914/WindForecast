# SeqNN Mid TCN W72 V1

작성일: 2026-07-09 03:48:27 +09:00

목적: 최근 72시간 weather sequence를 dilated causal convolution으로 보면서, short GRU보다 더 넓은 weather pattern/ramp/FICR 신호를 잡는지 확인한다.

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
| model name | `seqnn_mid_tcn_w72_v1` |
| family | `seqnn` |
| window | `72` |
| input features | `34` |
| model | TCN, causal/dilated conv |
| layers | `4`, dilation `1/2/4/8` |
| hidden size | `64` |
| loss | capacity-normalized weighted L1 |
| weight policy | `actual_sqrt` |
| early stopping | validation official score |
| device | CUDA 사용 확인 |

## 단독 OOF 결과

| Model | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---:|---:|---:|---:|
| TREE rolling baseline | `0.62396` | `0.12809` | `0.37602` | `0.60563` |
| SeqNN short GRU W24 | `0.61323` | `0.13700` | `0.36347` | `0.59955` |
| SeqNN mid TCN W72 | `0.61521` | `0.14078` | `0.37120` | `0.60283` |

TCN 단독은 TREE보다 낮지만 short GRU보다 높고, SeqNN 승격 후보 기준인 `0.615`를 넘었다.

## TREE + TCN OOF Blend

base는 `aggressive_minimal_rolling_v1` TREE, extra는 `seqnn_mid_tcn_w72_v1`이다.

| Variant | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---:|---:|---:|---:|
| TREE base | `0.62396` | `0.12809` | `0.37602` | `0.60563` |
| TREE 60% + TCN 40% | `0.63021` | `0.12714` | `0.38756` | `0.61593` |
| TCN only | `0.61521` | `0.14078` | `0.37120` | `0.60283` |

OOF blend gain:

- score: `+0.00624`
- nMAE: `0.12809 -> 0.12714`
- FICR: `0.37602 -> 0.38756`
- worst fold: `0.60563 -> 0.61593`

## Group 평균

| Variant | Group | Score | nMAE | FICR |
|---|---|---:|---:|---:|
| TREE base | group1 | `0.62184` | `0.12461` | `0.36830` |
| TREE base | group2 | `0.65099` | `0.12198` | `0.42397` |
| TREE base | group3 | `0.57888` | `0.14407` | `0.30182` |
| TREE 60% + TCN 40% | group1 | `0.63137` | `0.12234` | `0.38507` |
| TREE 60% + TCN 40% | group2 | `0.65720` | `0.12240` | `0.43681` |
| TREE 60% + TCN 40% | group3 | `0.58367` | `0.14275` | `0.31009` |

모든 group에서 개선했다. 특히 FICR 개선이 크다.

## Residual Correlation

| Scope | Corr |
|---|---:|
| overall | `0.84561` |
| group1 | `0.81419` |
| group2 | `0.86193` |
| group3 | `0.86520` |

상관은 높지만 short GRU와 비슷한 수준이며, blend gain은 더 크다.

## 생성 파일

- `results/oof_seqnn_mid_tcn_w72_v1.csv`
- `results/scores_seqnn_mid_tcn_w72_v1.csv`
- `results/summary_seqnn_mid_tcn_w72_v1.csv`
- `results/oof_compare_tree_rolling_seqnn_mid_tcn_w72_v1_scores.csv`
- `results/oof_compare_tree_rolling_seqnn_mid_tcn_w72_v1_summary.csv`
- `results/oof_compare_tree_rolling_seqnn_mid_tcn_w72_v1_residual_corr.csv`

## 판단

`seqnn_mid_tcn_w72_v1`은 현재 SeqNN 후보 중 가장 유망하다.

다음 확인 후보:

1. TCN seed bagging
2. TCN hyperparameter light tuning
3. PINN/TREE/TCN 3-branch OOF blend
4. TCN feature set 축소/확장 ablation

test submission은 아직 만들지 않는다.

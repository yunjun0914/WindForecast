# SeqNN TCN Window Comparison

작성일: 2026-07-09 03:58:29 +09:00

목적: TCN branch가 window 길이에 대해 일관적으로 유효한지 확인한다. 비교 대상은 24h, 72h, 168h이다.

## 고정점

- model: TCN, causal/dilated conv
- layers: `4`
- dilation: `1/2/4/8`
- hidden size: `64`
- input features: `34`
- validation: leave-one-year-out OOF
- TREE base: `aggressive_minimal_rolling_v1`
- test submission 생성 없음

## 단독 OOF

| Model | Window | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---:|---:|---:|---:|---:|
| TCN W24 | 24 | `0.61415` | `0.14121` | `0.36950` | `0.60455` |
| TCN W72 | 72 | `0.61521` | `0.14078` | `0.37120` | `0.60283` |
| TCN W168 | 168 | `0.61220` | `0.14268` | `0.36708` | `0.60402` |

단독 성능은 W72가 가장 높다.

## TREE + TCN Blend

| Extra Model | Best extra weight | Mean score | Mean nMAE | Mean FICR | Worst fold |
|---|---:|---:|---:|---:|---:|
| none, TREE base | `0.00` | `0.62396` | `0.12809` | `0.37602` | `0.60563` |
| TCN W24 | `0.30` | `0.62775` | `0.12663` | `0.38213` | `0.61124` |
| TCN W72 | `0.40` | `0.63021` | `0.12714` | `0.38756` | `0.61593` |
| TCN W168 | `0.30` | `0.62876` | `0.12689` | `0.38441` | `0.61462` |

모든 TCN window가 TREE blend에서 양수 개선이다. 최적은 W72다.

## Residual Correlation

| Model | Overall corr |
|---|---:|
| TCN W24 | `0.85071` |
| TCN W72 | `0.84561` |
| TCN W168 | `0.84367` |

window가 길어질수록 residual correlation은 조금 낮아진다. 하지만 W168은 단독 품질이 낮아 W72보다 최종 blend가 낮다.

## 판단

TCN family는 일관적으로 유효하다.

현재 우선순위:

1. `seqnn_mid_tcn_w72_v1`: best SeqNN branch
2. `seqnn_long_tcn_w168_v1`: second, lower correlation but weaker quality
3. `seqnn_short_tcn_w24_v1`: useful but W72/GRU 대비 우선순위 낮음

다음 후보:

- W72 TCN seed bagging
- W72 TCN light hyperparameter tuning
- W72/W168 TCN끼리 ensemble 후 TREE blend 확인
- 나중에 PINN/TREE/TCN 3-branch OOF blend

test submission은 아직 만들지 않는다.

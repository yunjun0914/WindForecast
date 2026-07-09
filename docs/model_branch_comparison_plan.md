# Model Branch Comparison Plan

작성일: 2026-07-09 02:02:30 +09:00

목적: TREE만 강화하는 흐름에서 벗어나, 서로 다른 시간 해상도를 보는 예측 branch를 같은 OOF 기준으로 비교한다. 이 문서는 바로 실험을 돌리는 명령서가 아니라, 다음 구현/검증의 기준이다.

## Current Phase

현재 위치는 Phase 4B다.

| Phase | 내용 | 상태 |
|---|---|---|
| Phase 3 | feature family/profile 정리 | 진행 중, `aggressive_minimal_rolling_v1` 임시 기준 |
| Phase 4A | TREE baseline 강화 | 진행 중, tuned LGBM + rolling feature |
| Phase 4B | 모델 branch 비교 | 현재 계획 단계 |
| Phase 4C | OOF 기반 blend/stack 후보 판단 | 이후 |

## Core Question

현재 TREE는 tabular feature로 시간 문맥을 간접 반영한다. 그러나 외부 대회/연구 코드는 짧은 window와 긴 window를 별도 branch로 보는 경우가 많다.

따라서 다음 질문을 확인한다.

1. 짧은 window 모델이 ramp/FICR을 TREE보다 잘 잡는가?
2. 긴 window 모델이 NMAE와 generalization을 안정화하는가?
3. SeqNN branch가 PINN/TREE와 다른 오차 구조를 가지는가?
4. group3 데이터 부족 문제에 shared/global sequence branch가 도움이 되는가?

## Branch Candidates

| Branch | Input Shape | Main Model | 목적 | 기대 |
|---|---|---|---|---|
| `tree_lgbm_rolling` | current row + engineered time features | LGBM | 안정적 tabular 기준선 | NMAE 기준 |
| `seqnn_short_gru` | recent 24h sequence | GRU or TCN | ramp, 빠른 변화, FICR 근처 보정 | FICR diversity |
| `seqnn_mid_tcn` | recent 72h sequence | TCN/WaveNet-lite | 날씨 변화 패턴과 local regime | short/long 중간 |
| `seqnn_long_dlinear` | recent 168h sequence | DLinear or GRU | 큰 날씨 흐름, 주간/계절 맥락 | NMAE/generalization |
| `pinn_effective` | weather + SCADA teacher | PINN | 물리/피크/FICR | 기존 PINN diversity |

## Fixed Baselines

기본 비교 기준:

| Baseline | File/Result | 기준 |
|---|---|---|
| TREE full control | `full_v2` | score `0.62361` |
| TREE minimal rolling | `aggressive_minimal_rolling_v1` | score `0.62396` |
| PINN current | `pinn_effective`, LGBM teacher | score about `0.61259` |
| Stable blend | PINN 50 : TREE 50 | 사용자가 바꾸기 전까지 기본 |

현재 TREE feature 기준은 `aggressive_minimal_rolling_v1`로 둔다. `aggressive_minimal_context_v1`은 NMAE는 개선했지만 FICR을 깎았으므로 branch comparison 기준으로 쓰지 않는다.

## SeqNN Input Feature Set

SeqNN에는 511개 feature를 넣지 않는다. sequence 모델은 feature 수보다 시간 구조가 중요하므로 작은 입력으로 시작한다.

### SeqNN v1 Feature 후보

| Family | Feature 예시 | 이유 |
|---|---|---|
| calendar | `sin_doy`, `cos_doy`, `sin_hod`, `cos_hod` | 시간 주기 |
| core wind | `gfs_ws100_speed`, `gfs_ws850_speed`, `ldaps_ws50_max_speed`, `ldaps_ws10_speed` | 발전량 주 신호 |
| gust/ramp | `gfs_surface_0_gust`, `phys_gfs_gust_factor`, `lead1_minus_lag1`류 | 튀는 구간 |
| physics | `rho * ws^3`, shear | 물리적 풍력 proxy |
| spatial compact | `phys_gfs_ws850_grid_max`, `phys_gfs_ws850_grid_p90`, `phys_ldaps_ws50max_grid_max` | 위치/풍속장 proxy |
| forecast cycle | `forecast_lead_hours`, lead 24h sin/cos | 예보 품질/lead-time |

목표 feature 수:

- v1: 20-40개
- 60개 이상은 이유가 있을 때만 허용
- raw SCADA feature 직접 사용 금지

## Window Design

| Window | 이름 | 목적 | 우선 모델 |
|---:|---|---|---|
| 24 | short | 급변/ramp/FICR | GRU, TCN |
| 72 | mid | weather regime 변화 | TCN |
| 168 | long | 주간 흐름/generalization | DLinear, GRU |

초기 구현은 한 번에 세 모델을 모두 크게 돌리지 않는다.

권장 순서:

1. `seqnn_short_gru_w24`
2. `seqnn_long_dlinear_w168`
3. short/long 둘 중 신호가 있으면 `seqnn_mid_tcn_w72`

## OOF Validation Contract

모든 branch는 동일한 leave-one-year-out OOF를 사용한다.

| Fold | Train years | Predict year |
|---|---|---|
| 1 | `2023,2024` | `2022` |
| 2 | `2022,2024` | `2023` |
| 3 | `2022,2023` | `2024` |

주의:

- validation year raw SCADA 직접 사용 금지
- target history 직접 사용 금지
- train row teacher를 쓰는 경우 OOF/crossfit만 허용
- sequence window가 과거 row를 볼 때도 target은 포함하지 않는다
- test submission은 만들지 않는다

## Output Schema

새 branch는 처음부터 standard OOF long format을 출력한다.

필수 출력:

- `results/oof_seqnn_short_gru_w24_v1.csv`
- `results/scores_seqnn_short_gru_w24_v1.csv`
- `results/summary_seqnn_short_gru_w24_v1.csv`

OOF 필수 컬럼:

- `forecast_kst_dtm`
- `pred_year`
- `train_years`
- `model_family`
- `model_name`
- `group`
- `actual`
- `pred`
- `is_clipped`

## Comparison Metrics

단독 점수만 보지 않는다.

| 평가 | 내용 |
|---|---|
| mean score | 전체 평균 |
| group score | 특히 group3 |
| nMAE/FICR split | NMAE 개선인지 FICR 개선인지 분리 |
| worst fold | 과적합/일반화 위험 |
| residual correlation | TREE/PINN과 다른 오차인지 |
| blend simulation | OOF에서만 50:50 또는 동일가중 평균 확인 |

## Promotion Criteria

SeqNN 단독이 TREE를 이길 필요는 없다. diversity가 핵심이다.

승격 후보:

- 단독 mean score가 `0.615` 이상
- 또는 group3가 현재 TREE보다 명확히 개선
- 또는 TREE/PINN residual correlation이 낮고 blend OOF가 `+0.003` 이상
- worst fold가 크게 악화되지 않음

강한 승격:

- OOF blend 기준 `+0.005` 이상
- group3 개선 동반
- FICR 개선이 NMAE를 크게 훼손하지 않음

test submission 기준:

- OOF `+0.01`급 개선
- 또는 사용자가 명시적으로 요청

## Implementation Order

1. `utils/seq_dataset.py`
   - feature table -> sequence tensor
   - group별 target merge
   - year-fold split

2. `models/seqnn.py`
   - GRU baseline
   - DLinear baseline
   - TCN은 v2

3. `experiments/run_seqnn_oof.py`
   - branch/model/window/config 선택
   - standard OOF output
   - early stopping 필수

4. `experiments/evaluate_oof_branch_blends.py`
   - TREE/PINN/SeqNN OOF 비교
   - residual correlation
   - OOF-only blend

## Risks

| Risk | 대응 |
|---|---|
| 학습 느림 | small feature set, early stopping, window 24부터 |
| group3 과적합 | shared model + group embedding 후보는 v2로 분리 |
| FICR 개선 없이 NMAE 악화 | 단독 제출 후보 아님, diversity만 확인 |
| target history leakage | target lag 사용 금지 |
| 코드 복잡도 증가 | standard OOF schema부터 맞춤 |

## Current Recommendation

다음 구현은 `seqnn_short_gru_w24_v1`부터 시작한다.

이유:

- 가장 빠르게 검증 가능
- rolling feature가 이미 유효했으므로 sequence도 단기 weather context에서 신호가 날 가능성이 높다
- FICR/ramp 쪽에서 TREE와 다른 오차 구조가 기대된다

`seqnn_short_gru_w24_v1`이 단독으로 약하더라도 residual correlation이 낮으면 blend 후보로 남긴다.

## First Implementation Result

작성일: 2026-07-09 03:28:12 +09:00

`seqnn_short_gru_w24_v1`을 구현하고 OOF 검증했다.

결과:

| Model | Mean score | Mean nMAE | Mean FICR | 판단 |
|---|---:|---:|---:|---|
| TREE rolling baseline | `0.62396` | `0.12809` | `0.37602` | base |
| SeqNN short GRU W24 | `0.61323` | `0.13700` | `0.36347` | 단독 약함 |
| TREE 70% + SeqNN 30% | `0.62768` | `0.12571` | `0.38107` | OOF blend 후보 |

해석:

- 단독 SeqNN은 TREE보다 낮다.
- 그러나 TREE에 30% 섞으면 OOF score가 `+0.00372` 오른다.
- group1/2/3 모두 소폭 개선했다.
- residual correlation은 overall `0.86338`로 높아, 완전히 독립적인 diversity라기보다는 temporal smoothing/sequence 효과로 보인다.

다음 판단:

- 바로 test submission을 만들지 않는다.
- PINN/TREE/SeqNN 3-branch OOF blend 확인이 다음 후보.
- long-window DLinear/GRU도 여전히 가치가 있다.

## Long DLinear W168 Result

작성일: 2026-07-09 03:35:28 +09:00

`seqnn_long_dlinear_w168_v1`을 구현하고 OOF 검증했다.

결과:

| Model | Mean score | Mean nMAE | Mean FICR | 판단 |
|---|---:|---:|---:|---|
| TREE rolling baseline | `0.62396` | `0.12809` | `0.37602` | base |
| SeqNN long DLinear W168 | `0.54953` | `0.17689` | `0.27595` | 기각 |
| TREE 95% + DLinear 5% | `0.62387` | `0.12798` | `0.37572` | base보다 낮음 |

해석:

- residual correlation은 overall `0.68020`으로 short GRU보다 낮다.
- 그러나 단독 예측 품질이 너무 낮아 blend에 도움이 되지 않는다.
- long-window 아이디어는 유지하되, DLinear W168 v1은 기각한다.

다음 long 후보:

- long GRU
- mid/long TCN
- group-shared sequence model

## Mid TCN W72 Result

작성일: 2026-07-09 03:48:27 +09:00

`seqnn_mid_tcn_w72_v1`을 구현하고 OOF 검증했다.

결과:

| Model | Mean score | Mean nMAE | Mean FICR | 판단 |
|---|---:|---:|---:|---|
| TREE rolling baseline | `0.62396` | `0.12809` | `0.37602` | base |
| SeqNN mid TCN W72 | `0.61521` | `0.14078` | `0.37120` | SeqNN 단독 후보 |
| TREE 60% + TCN 40% | `0.63021` | `0.12714` | `0.38756` | 강한 OOF blend 후보 |

해석:

- TCN 단독은 TREE보다 낮지만 short GRU보다 높다.
- TREE blend에서 score `+0.00624`, worst fold `+0.01030`.
- group1/2/3 모두 개선했고, FICR 개선이 크다.
- 현재 SeqNN 후보 중 가장 유망하다.

다음 판단:

- test submission은 아직 만들지 않는다.
- TCN seed bagging/light tuning 또는 PINN/TREE/TCN 3-branch OOF blend를 다음 후보로 둔다.

## TCN Window Comparison Result

작성일: 2026-07-09 03:58:29 +09:00

TCN을 window `24`, `72`, `168`로 비교했다.

| Model | Single score | Best TREE+TCN score | Best extra weight | 판단 |
|---|---:|---:|---:|---|
| TCN W24 | `0.61415` | `0.62775` | `0.30` | 유효 |
| TCN W72 | `0.61521` | `0.63021` | `0.40` | best |
| TCN W168 | `0.61220` | `0.62876` | `0.30` | 유효하지만 W72보다 낮음 |

해석:

- TCN family는 window 길이에 대해 일관적으로 TREE blend 개선을 만든다.
- W72가 단독/블렌드 기준 모두 가장 좋다.
- W168은 residual correlation이 가장 낮지만 단독 품질이 약해 W72를 넘지 못했다.
- W24는 GRU W24와 비슷한 역할이나, W72가 더 좋으므로 우선순위는 낮다.

다음 후보:

- W72 TCN seed bagging
- W72 TCN light tuning
- W72/W168 TCN ensemble
- PINN/TREE/TCN OOF blend는 마지막 성능 끌어올릴 때 수행

## PINN + TREE + TCN Family Blend Result

작성일: 2026-07-09 04:09:49 +09:00

사용자 지정 TCN family를 먼저 만들었다.

| TCN component | Weight |
|---|---:|
| W24 | `0.30` |
| W72 | `0.40` |
| W168 | `0.30` |

그 다음 PINN/TREE/TCN-family branch weight를 OOF에서 `0.05` 간격으로 비교했다.

| Variant | Mean score | Mean nMAE | Mean FICR | Worst fold | 판단 |
|---|---:|---:|---:|---:|---|
| TREE only | `0.62396` | `0.12809` | `0.37602` | `0.60563` | base |
| TCN family only | `0.62084` | `0.13797` | `0.37966` | `0.61141` | 단독은 낮지만 FICR 보조 |
| PINN only | `0.61293` | `0.14197` | `0.36783` | `0.60596` | 단독 약함 |
| PINN `0.25` + TREE `0.40` + TCN family `0.35` | `0.63088` | `0.12760` | `0.38937` | `0.61667` | best |

해석:

- 세 branch를 섞는 방향은 맞다.
- TCN family는 TREE의 NMAE를 크게 망치지 않으면서 FICR을 올린다.
- PINN은 단독 점수는 낮지만 20-30% 정도 섞일 때 best 근처에 자주 등장한다.
- group3는 여전히 병목이다. 모델 branch 다양화만으로는 0.65급 점프가 부족하므로, 다음 큰 후보는 group3/weather-teacher/feature 쪽이다.

출력:

- `results/oof_pinn_tree_tcnfamily_w24_030_w72_040_w168_030_best_oof.csv`
- `results/oof_pinn_tree_tcnfamily_w24_030_w72_040_w168_030_summary.csv`
- `docs/pinn_tree_tcn_family_oof_blend.md`

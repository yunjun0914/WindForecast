# External Wind Forecasting Code Reference

작성일: 2026-07-09 KST

목적: 외부 공개 코드와 현업 라이브러리가 실제로 어떤 데이터 처리, 모델 구조, 후처리를 쓰는지 정리한다. 이 문서는 "바로 실험할 목록"이 아니라, 큰 방향을 정할 때 반복해서 참고하는 자료다.

## 핵심 결론

1. 상위권 코드는 피쳐를 무작정 많이 만들기보다, **SCADA/운영 이상치 처리**, **풍향/공간 구조**, **시간 시퀀스**, **물리 기반 wind feature**를 강하게 다룬다.
2. 풍향은 단순 raw column이 아니라, wake, yaw, 단지축, 입사각과 연결되는 구조적 변수다.
3. SCADA는 그대로 test에 쓸 수 없으므로, "SCADA 예측값을 teacher feature로 추가"보다 더 큰 목표는 **weather 자체를 turbine/site effective weather로 보정**하는 것이다.
4. PINN 계열에 NN/RNN을 추가할 때는 residual 보정이 아니라, 기존 PINN과 같은 입력/출력 구조에서 **network backbone만 다양화한 병렬 모델**로 보는 게 맞다.
5. Tree 성능을 더 크게 올리려면 500개 feature를 더 늘리기보다, 낮은 가치 feature family를 줄이고 물리적으로 강한 family를 재구성하는 쪽이 우선이다.

## 참고한 코드와 자료

| 구분 | 링크 | 확인한 핵심 |
|---|---|---|
| KDD 2022 3rd | https://github.com/LongxingTan/KDDCup2022-WPF | Transformer/BERT 계열, wind speed/direction 중심, lag/rolling, daily pattern postprocess |
| KDD 2022 6th | https://github.com/shaido987/KDD_wind_power_forecast | MDLinear + XGTN, invalid mask, cluster/location feature, WaveNet + GCN |
| KDD 2022 11th | https://github.com/BUAABIGSCity/KDDCUP2022 | AGCRN/MTGNN, geo/DTW graph, FilterHuberLoss, turbine-level normalization |
| Deep Spatio-Temporal | https://github.com/jiangyuan-li/Deep-Spatio-Temporal | GRU/LSTM seq2seq, turbine embedding, optional attention |
| OpenOA | https://github.com/NREL/OpenOA | data quality filters, air density, shear, veer, turbulence, power curve fitting |
| FLORIS | https://github.com/NREL/floris | wake simulation, wind speed/direction, turbulence, air density, yaw, power setpoint |
| PyWake | https://gitlab.windenergy.dtu.dk/TOPFARM/PyWake | wake deficit, blockage, turbulence, power/CT curve, rotor averaging |
| HEFTCom2024 | https://arxiv.org/abs/2505.10367 | NWP source별 sister forecast, spatial stats, quantile LightGBM, stacking |

## 외부 코드에서 반복되는 패턴

### 1. SCADA/운영 이상치 처리

KDD 코드들은 발전량을 그냥 모두 학습하지 않는다. 다음 같은 조건을 mask로 빼거나 보정한다.

- `Patv < 0`
- `Wspd > 2.5`인데 `Patv == 0`
- blade pitch 계열 `Pab1/Pab2/Pab3 > 89`
- `Wdir`, `Ndir` 범위 이상
- NaN row

우리에게 주는 의미:

- train-only hour bias는 이상치를 흡수하는 장치로는 유용하지만, 지금보다 더 명시적인 invalid/availability mask가 필요할 수 있다.
- target residual 기반 이상치 탐지는 모델 오차를 다시 학습할 위험이 있다.
- SCADA 원본 기반의 availability, valid turbine ratio, curtailment proxy를 sample weight나 loss weight로 쓰는 방향이 더 자연스럽다.

우선순위 후보:

1. SCADA 기반 invalid/availability mask 정의
2. mask를 label 제거가 아니라 sample weight로 적용
3. group/year/fold별로 mask 비율과 점수 영향 확인

### 2. 풍향은 raw column보다 구조가 중요

FLORIS/PyWake는 풍향을 단순 feature로 보지 않는다. 모델 입력 자체가 다음 구조를 가진다.

- wind speed
- wind direction
- turbulence intensity
- air density
- yaw angle
- wake deficit
- blockage
- power/CT curve
- turbine layout

우리에게 주는 의미:

- WD를 teacher target에 넣는 것은 좋은 출발이다.
- 다만 더 큰 후보는 **direction-conditioned effective wind**다.
- 예: `ws * cos(direction - learned_axis)`, direction bin별 power curve, upwind/downwind weighted grid, wake-loss proxy.

주의:

- 360개 방향별 자유 파라미터처럼 너무 자유도가 높은 correction은 과적합 위험이 크다.
- Fourier basis, coarse direction bins, ridge-axis basis처럼 낮은 자유도부터 보는 게 낫다.

### 3. 공간 구조를 모델에 넣는다

KDD GNN 계열은 터빈 위치 또는 출력 유사도로 graph를 만든다.

- geo graph: 터빈 좌표 거리 기반
- DTW graph: historical power pattern 유사도 기반
- cosine similarity adjacency
- WaveNet/GCN, AGCRN, MTGNN 계열

우리에게 주는 의미:

- 우리는 turbine 좌표가 없지만, weather grid 좌표와 group output은 있다.
- 따라서 full turbine graph 대신 다음이 현실적이다.

후보:

1. group별 effective grid weight 학습
2. wind direction별 upwind/downwind grid aggregation
3. grid 간 gradient, max-min, p75-p25, p90-p10
4. group3 데이터 부족 보완을 위한 global/shared spatial encoder

### 4. 시계열 모델은 direct forecast로 둔다

외부 NN 계열은 보통 다음처럼 원본 시계열 구조를 직접 본다.

- DLinear: moving average로 trend/seasonal decomposition
- XGTN: WaveNet temporal block + GCN spatial block
- Deep Spatio-Temporal: GRU/LSTM seq2seq + turbine embedding + attention
- KDD 3rd: historical wind speed/direction + calendar decoder + postprocess

우리에게 주는 의미:

- RNN/GRU/TCN을 넣는다면 residual model부터 하지 않는다.
- 기존 PINN과 같은 목적의 병렬 후보로 둔다.
- 즉 `weather sequence -> power` 또는 `weather sequence -> PINN latent/effective wind -> power`가 맞다.

우리 기준 설계 원칙:

- 기본 blend 비율은 사용자가 명시하기 전까지 `PINN 50 : TREE 50`을 유지한다.
- 새 NN 후보는 먼저 OOF valid에서 단독 성능과 PINN/TREE 상관관계를 확인한다.
- residual correction은 마지막 후보정 단계로 둔다.

### 5. 물리 기반 feature family

OpenOA/FLORIS/PyWake에서 반복되는 물리 변수:

- air density
- air-density-adjusted wind speed
- wind speed cubed
- `rho * ws^3`
- vertical shear
- wind veer
- turbulence intensity
- wind direction sin/cos
- power curve
- wake loss
- availability/curtailment loss

우리에게 주는 의미:

- 500개 feature를 단순 확장하는 것보다, 물리적으로 설명 가능한 family를 중심으로 재정리해야 한다.
- 중요도가 낮은 family는 제거하고, 남은 자리에 위 물리 family를 더 정교하게 넣는 편이 낫다.

우선 유지 가치가 큰 family:

- wind speed / wind power
- direction / directional component
- vertical shear / veer
- spatial gradient / effective grid
- calendar seasonality
- SCADA teacher-derived effective weather

우선 제거 검토 family:

- 일관적으로 importance가 낮은 precipitation 계열
- cloud 계열
- 설명력이 낮은 중복 polynomial
- 과도하게 세분화된 low-importance raw meteo

## 우리 모델에 바로 연결되는 큰 후보

### A. SCADA 기반 availability weighting

목표:

- 날씨로 설명하기 어려운 운영정지/센서오류/curtailment를 loss에서 과도하게 학습하지 않도록 한다.

핵심:

- target residual이 아니라 SCADA 원본에서 mask/weight를 만든다.
- hard delete보다 sample weight가 안전하다.

검증:

- OOF year-fold에서 group별 NMAE/FICR 확인
- 특히 group3 악화 여부 확인

### B. Direction-conditioned effective wind

목표:

- 풍향을 raw WD로 넣는 수준을 넘어, 단지에 실제로 들어오는 바람 성분으로 변환한다.

후보 feature:

- `ws * cos(wd - axis)`
- `ws * sin(wd - axis)`
- direction bin별 effective ws
- direction별 grid weight
- upwind weighted ws
- wake-loss proxy

검증:

- Tree only OOF
- PINN teacher target으로 WD 포함/미포함 비교
- feature importance에서 WD family가 살아나는지 확인

### C. Spatial weather representation

목표:

- group별 가장 유효한 weather grid/height/location을 학습한다.

후보:

- nearest/mean/max/min 대신 learned weighted grid
- wind direction별 upwind/downwind aggregation
- 850hPa, 100m, 10m 사이 shear/ratio/difference

검증:

- group3 개선 여부를 최우선으로 본다.
- group1/2가 약간 떨어져도 group3가 크게 오르면 따로 ensemble 후보 가능.

### D. Direct sequence NN as PINN sibling

목표:

- Tree/PINN과 다른 오차 구조를 가진 모델을 추가한다.

후보:

- DLinear
- GRU/LSTM seq2seq
- TCN/WaveNet
- lightweight Transformer

금지:

- 처음부터 residual model로 만들지 않는다.
- 검증 없이 test submission 생성하지 않는다.
- 50:50 기본 blend를 임의로 바꾸지 않는다.

### E. Feature family pruning and rebuild

목표:

- 511개 feature를 설명 가능한 family 단위로 줄이고, 강한 물리 feature를 넣는다.

방법:

- 개별 feature importance만 보고 삭제하지 않는다.
- family 단위로 낮은 계열을 제거한다.
- wind family 내부의 일부 low importance feature는 보존할 수 있다.

검증:

- full feature vs pruned feature
- pruned + new physics family
- group별 score와 feature importance 동시 확인

## 작업 규칙

이 문서에서 나온 후보는 바로 실험하지 않는다. 실험 전에는 다음을 먼저 사용자에게 설명한다.

1. 목적
2. 변경점
3. 고정할 구조
4. 사용할 데이터/feature
5. validation 방식
6. 기대 효과
7. 위험
8. 생성 파일

큰 폭 개선 후보가 아닌 경우 test submission은 만들지 않는다. 기본 앙상블 비율은 사용자가 바꾸라고 하기 전까지 `PINN 50 : TREE 50`으로 둔다.

## 다음에 다시 볼 코드 포인트

필요할 때만 아래 파일을 추가 확인한다.

- KDD 6th `methods/xtgn/model/layers_version_1.py`: WaveNet/GCN 내부 구조
- KDD 6th `methods/xtgn/model/engine.py`: masked loss와 optimizer 세부
- KDD 11th `loss.py`: FilterHuberLoss 조건 전체
- OpenOA `filters.py`: bin/window/unresponsive filter 구현
- OpenOA `power_curve/functions.py`: IEC/GAM/GAM3 power curve
- PyWake `power_ct_functions.py`: yaw/density/power curve wrapping
- FLORIS `floris_model.py`: wind direction, turbulence, air density, yaw, disable turbine 흐름

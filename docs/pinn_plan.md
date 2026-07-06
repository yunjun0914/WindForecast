# PINN 모델링 계획

기존 트리 앙상블(RF/LGBM/XGB) 파이프라인과 별도로 진행하는 물리 기반(PINN) 모델링 계획. 트리 계열의 근본적 약점(과적합, 평균수렴 편향)을 물리식 구조로 보완하는 게 목표.

## 1. 지배방정식

$$P_{phys}(v,t) = \frac{1}{2}\rho(t)\,A\,v(t)^3\cdot C_{eff}(v,t)$$

$$C_{eff}(v,t) = C_{all}(v) + g_{doy}(doy(t)) + g_{moy}(moy(t))$$

- $C_{eff} \equiv C_p\cdot\eta$ — 파워계수와 발전효율을 분리 불가능한 하나의 값으로 통합 (SCADA에 RPM·피치각이 없어서 $C_p(\lambda,\beta)$를 따로 못 구함)
- $C_{all}(v)$: 신경망이 학습하는 기본 성분 (물리적 스케일 그대로, 대략 0.3~0.4대)
- $g_{doy}, g_{moy}$: 연주기·일주기(day-of-year, month-of-year) 1차 조화항. 0 근처 작은 보정, L2 수축. **합(sum)이지 평균 아님**

## 2. 데이터 요구사항 (항목별)

| 물리량 | 조달 방법 | 데이터 소스 | 상태 |
|---|---|---|---|
| $\rho$ (공기밀도) | 이상기체법칙 $\rho=p/(R_{specific}T)$ | LDAPS `heightAboveGround_2_t`+`surface_0_sp`, GFS `heightAboveGround_2_2t`+`surface_0_sp` | **복원 필요** — 바람전용 피처정리 때 뺐던 기온·기압을 이 항 계산용으로만 되살림 |
| $A$ (로터면적) | $\pi(D/2)^2$ | `info.xlsx` 로터직경 — VESTAS V126=126m→A≈12,469m², UNISON U136=136m→A≈14,527m² | 그룹별(제작사별) 상수, 이미 확보됨 |
| $v$ (허브높이 근사풍속) | LDAPS 10m 풍속을 시어 지수(power-law, $\alpha=0.14$)로 허브높이(117m)까지 외삽 + 제작사별 SCADA 기반 선형보정 | `ldaps_train.csv`의 `heightAboveGround_10_10u/10v` 16격자 평균 → `utils/pinn_data.py::build_pinn_weather`/`fit_wind_speed_correction` | **완료** — 최초엔 GFS 100m 풍속(`gfs_ws100_speed`)을 썼으나 SCADA 대비 약 2배 낮게 나오는 버그 발견(상관 0.70, 중앙값 비율 1.98) → LDAPS로 전환(상관 0.80, 비율≈1.01). 아래 "구현 중 겪은 함정" 참고 |
| $C_{max}$ (Betz 실효상한) | SCADA 역산 + 95%분위수 | SCADA(`scada_vestas_train.csv`,`scada_unison_train.csv`)의 (v,P)쌍으로 $C_{eff,emp}=P/(0.5\rho Av^3)$ 계산, 매칭 시각의 기온·기압으로 $\rho$ 추정 | **완료** (`estimate_cp_max.py`) — VESTAS≈0.42, UNISON≈0.49 (95%분위수, $v\ge9$m/s 필터) |
| $doy(t), moy(t)$ | 이미 만든 시간피처 로직 재사용 | `utils/preprocessing.py`의 cyclical 인코딩 방식 그대로 | 이미 확보됨 |
| 라벨 $y_t$ | 그룹별 발전량 | `train_labels.csv` | 이미 확보됨 |

## 3. 시간축: ODE와 정확해

$$\tau\frac{dP}{dt}+P=P_{phys}(v(t))$$

선형 ODE라 **정확해(컨볼루션)를 그대로 사용** (Option A 채택, $P_\theta$라는 별도 네트워크 불필요 — 자세한 논의는 아래 "결정 배경" 참고):

$$P_t = \sum_{k=0}^{K}w_k\,P_{phys}(v_{t-k}),\quad w_k=\alpha(1-\alpha)^k,\quad \alpha=1-e^{-\Delta t/\tau}$$

- $\tau$(또는 $\alpha$)는 **제작사별 스칼라 학습 상수 2개** (VESTAS 공유, UNISON 별도) — 풍속·시간에 따라 변하는 함수가 아니라 그냥 최적화되는 숫자 2개. 제약 없는 자유파라미터 $\theta_\tau$에서 $\tau=\text{softplus}(\theta_\tau)$, $\alpha=1-e^{-\Delta t/\tau}$로 변환해서 항상 $\tau>0$, $\alpha\in(0,1)$ 보장
- **윈도우 $K=24$**(하루치)로 고정. $K$ 자체는 물리량이 아니라 무한합 $\sum_{k=0}^\infty$를 어디서 끊어도 오차가 무시할 만한지 정하는 공학적 절단점 — $\alpha$가 웬만큼 작지 않은 이상 24시간이면 가중치가 거의 다 소진됨

## 4. 잔차 흡수 계층 (latent factor / hierarchical bias)

$$\hat P_t = P_t + b_{hour}(t) + b_{day}(doy_t) + b_{month}(moy_t)$$

| 항 | 인덱스 | Test 적용 | 이유 |
|---|---|---|---|
| $b_{hour}$ | 정확한 타임스탬프(row당 1개) | ❌ | 학습 전용 잔차흡수(정비·고장 등 물리로 설명 안 되는 노이즈). Test의 정확한 시각은 학습에 없던 새 키라 대응값 없음 |
| $b_{day}$ | day-of-year (1~366) | ✅ | 주기적으로 반복되는 값이라 test에도 대응 가능 |
| $b_{month}$ | month-of-year (1~12) | ✅ | 마찬가지로 주기 반복. 원래 "연도별" 아이디어였으나 실제 연도는 반복 안 되니 월 단위로 근사 |

손실에 L2 수축 적용 (세밀한 수준일수록 더 강하게):
$$\lambda_h\|b_{hour}\|^2 + \lambda_d\|b_{day}\|^2 + \lambda_m\|b_{month}\|^2$$

**주의**: 검증(2024 홀드아웃)·실제 test 예측 시 $b_{hour}=0$으로 두고, $b_{day}, b_{month}$는 학습된 값을 그대로 사용.

## 5. $v$축 물리 제약 (collocation, PINN 잔차손실)

시간축 ODE는 정확해로 풀렸지만, $C_{all}(v)$가 풍속에 대해 만족해야 할 물리 제약은 collocation point(라벨 불필요, 학습+테스트 전체 풍속분포에서 샘플링)로 강제:

| 손실 | 수식 | 적용범위 |
|---|---|---|
| $L_{Betz}$ | $\max(0,C_{all}(v_c)-C_{max})^2$ | 전체 $v$, $C_{max}$=제작사별 실측 95%분위수 |
| $L_{bc}$ | $C_{all}(v_{cutin})^2+C_{all}(v_{cutout})^2$ | 경계점(컷인/컷아웃) |
| $L_{flat}$ | $(dP_{phys}/dv)^2$ | 정격~컷아웃 구간만 |
| $L_{smooth}$ | $(d^2C_{all}/dv^2)^2$ | 전체 $v$ |

## 6. 전체 손실함수

$$L = L_{data} + \lambda_{Betz}L_{Betz}+\lambda_{bc}L_{bc}+\lambda_{flat}L_{flat}+\lambda_{smooth}L_{smooth} + \lambda_h\|b_{hour}\|^2+\lambda_d\|b_{day}\|^2+\lambda_m\|b_{month}\|^2$$

$L_{data}$는 가중 MSE 대신 **실제 채점식(NMAE+FICR)을 직접 손실로 사용**합니다. $valid$ = 실제발전량이 설비용량 10% 이상인 시간대 (평가 기준과 동일), $e_i=|y_i-\hat P_i|/\text{capacity}$:

$$L_{NMAE} = \frac{1}{N_v}\sum_{i\in valid} e_i$$

FICR의 오차구간별 단가(계단함수, 6%=4.0/8%=3.0/초과=0.0)는 미분 불가능(거의 모든 곳에서 기울기 0)이라 sigmoid로 부드럽게 근사:

$$\text{price}_{soft}(e) = 4 - \sigma\!\left(\frac{e-0.06}{\gamma}\right) - 3\sigma\!\left(\frac{e-0.08}{\gamma}\right), \quad \gamma=0.01\text{ (고정 상수, 학습 안 함)}$$

$$L_{FICR} = 1-\frac{\sum_{i\in valid} y_i\cdot\text{price}_{soft}(e_i)}{\sum_{i\in valid} y_i\cdot4.0}$$

$$L_{data} = 0.5\,L_{NMAE} + 0.5\,L_{FICR}$$

$\lambda_{NMAE}=\lambda_{FICR}=0.5$는 임의로 고른 게 아니라 **대회 채점식(`0.5(1-NMAE)+0.5FICR`)을 최대화하는 것과 상수차만 나고 수학적으로 동일** — 대회 점수 자체를 목적함수로 직접 최적화하는 것과 같음. $\gamma$는 **학습 파라미터로 두지 않고 0.01로 고정** — γ까지 학습되면 손실 지형이 학습 중 계속 바뀌어 불안정해질 위험이 커서, 상수로 고정하는 게 안전하다고 판단

## 7. 아키텍처

- **$C_{all}$은 제작사별 공유**: VESTAS용 하나(group_1,2 공유), UNISON용 하나(group_3) — 그룹별로 완전히 따로 두지 않음. (group_2 모델 전이 실험에서 물리관계 공유가 그룹별 개별학습보다 잘 일반화됨을 확인한 것과 일치)
- 그룹별 차이(정비 이력, 위치별 노이즈)는 bias 계층이 흡수
- **2단계 커리큘럼 학습**: 1단계 물리항($C_{all}$)만으로 먼저 수렴 → 2단계 bias 항 추가 (bias가 그래디언트를 초반에 가로채는 것 방지)

## 8. 검증

기존과 동일하게 **시간기준 홀드아웃**(2022~2023 학습 / 2024 검증) 재사용. 검증 시 $b_{hour}$ 미사용(0), $b_{day}/b_{month}$는 사용.

## 9. 결정 배경 요약 (왜 이 구조인지)

- **$P_\theta$(collocation 잔차 기반 시간축 근사)를 안 쓰는 이유**: ODE가 선형이라 정확해가 이미 있음. 정확해를 놔두고 신경망이 같은 걸 근사로 재발견하게 하면, $C_{all}$의 부정확함 + 시간동역학 근사오차가 이중으로 쌓여서 손해. $C_{all}$(우리가 모르는 부분)에만 신경망의 근사 예산을 쓰는 게 맞음
- **"test기간 노출" 효과는 $v$축 collocation으로 대체 확보**: 시간축에서 못 얻는 준지도학습 이점을, $C_{all}(v)$에 대한 Betz/경계조건/평탄성/매끄러움 제약을 test기간 풍속분포에서도 계산하는 방식으로 동일하게 얻음
- **$C_{max}$를 이론값(16/27)이 아니라 실측 기반으로 바꾼 이유**: 16/27은 상수라 애초에 "신뢰구간"을 적용할 통계적 대상이 아님. 실측 SCADA 기반 분포는 진짜 변동성이 있는 데이터라 95%분위수가 통계적으로 의미 있음

### $C_{max}$ 실측 산정 시 겪은 함정 (`estimate_cp_max.py`)

1. **SCADA `power_kw10m` 단위 오해**: 평균전력(kW)이 아니라 **10분간 누적 에너지(kWh)**였음. `train_labels.csv`와 대조해서 확인 — 6개(10분×6) 값을 **합산**해야 라벨과 일치(비율 0.93~1.01), 평균을 내면 5.6~6배 차이남. 시간당 평균전력 환산 시 **6을 곱해야** 함
2. **저풍속 구간 난류편향**: $v<9$m/s 근처에서 $C_{eff,emp}$가 비정상적으로 높게 나옴(UNISON 일부 3.3까지). 원인은 Jensen 부등식 — SCADA 풍속은 10분 평균인데 발전량은 $v^3$ 비례라, $E[v^3]\ge(E[v])^3$이라서 순간 돌풍이 있으면 평균풍속 기준 역산 $C_p$가 과대추정됨. 저풍속일수록 난류강도가 상대적으로 커서 이 왜곡도 심해짐. **$v\ge9$m/s로 필터링**해서 해결
3. UNISON이 VESTAS보다 심하게 오염된 건 부지(태백가덕산/태백원동) 차이 때문일 가능성을 확인했으나, 같은 부지(1호기)도 동일하게 나타나서 **부지 문제는 아니었고 제작사(터빈 자체 특성) 차이**로 결론

## 10. 미정 사항 (구현 전 확정 필요)

- $\lambda_{Betz},\lambda_{bc},\lambda_{flat},\lambda_{smooth},\lambda_h,\lambda_d,\lambda_m$ 각각의 크기 — 직접 실험하며 파악
- PyTorch 환경 세팅 (`environment.yml` 갱신)

## 11. 구현 중 겪은 함정

### 11.1 풍속 소스 오류 (GFS→LDAPS 전환)

최초 구현은 GFS 100m 풍속을 그대로 사용 — 검증 결과 예측 이용률이 실제(29%)의 1/3 수준(10.7%)으로 나와 발견. 원인 진단 과정: (1) 학습된 $C_{all}(v)$ 곡선 자체는 물리적으로 타당(cut-in, rated 근처 피크, rated 이후 $1/v^3$ 하강) → 모델 구조는 문제 없음. (2) 검증기간 예측에 쓰인 $v$의 평균이 3.89m/s로, 관측된 29% 이용률에 비해 터무니없이 낮음. (3) GFS $v$와 SCADA 실측 풍속을 직접 상관분석 → 상관 0.70, 중앙값 비율 1.98배(GFS가 SCADA의 절반). GFS는 0.25°(~28km) 해상도라 능선 지형의 가속 효과를 못 잡는 것으로 결론.
**해결**: LDAPS(1.5km) 10m 풍속을 시어 지수 $\alpha=0.14$로 허브높이(117m) 외삽 → 상관 0.80, 비율≈1.01로 크게 개선. 그래도 남는 편차는 제작사별 SCADA 선형회귀(`fit_wind_speed_correction`, 학습기간 SCADA만 사용해 리키지 없음)로 추가 보정.

### 11.2 bias(hour/day/month)가 전혀 안 움직이던 문제 — 두 개의 독립된 버그

`bias_l2`가 500 epoch 내내 정확히 0.0000으로 찍혀서 발견. `TurbineGroupBias`는 물리식(`PowerCurvePINN.forward`)에는 전혀 등장하지 않고, `group_prediction()`에서 물리식 예측 결과에 **사후적으로 더해지는** 순수 잔차보정 항 — 그런데도 전혀 안 움직인 건 명백히 버그였음.

1. **Adam `eps`가 gradient를 삼킴**: `data_loss`는 NMAE 항을 `capacity`(21600)로 나누고 `N_valid`(~11000개) 샘플에 대해 평균낸다. 샘플 하나에 대응하는 파라미터(특히 `hour_bias`는 샘플당 1개)가 받는 gradient는 $O(1/(\text{capacity}\times N_{valid}))\approx10^{-9}$ 스케일인데, Adam 기본 `eps=1e-8`이 이보다 커서 적응적 정규화의 분모를 gradient 분산이 아니라 eps 자체가 지배 → 사실상 스텝 크기가 붕괴. **해결**: bias 파라미터를 별도 optimizer param group으로 분리해 `eps=1e-12`, `lr=1e-2`(`BIAS_LR`/`BIAS_EPS`, `train_pinn.py`) 적용.
2. **`bias_l2`(L2 정규화)의 mean-reduction이 만드는 과도한 복원력**: `bias_l2`는 `weight.pow(2).mean()`, 즉 전체 파라미터 개수 $n$으로 나눈다. 파라미터 하나 $w_i$가 받는 정규화 gradient는 $2\lambda w_i/n$ — `hour_bias`는 $n\approx17422$(학습 row 수)라서, 데이터 gradient(~$10^{-9}$)와 균형점을 계산하면 $w_i^*\approx \text{data\_grad}\times n/(2\lambda)$. $\lambda_{hour}=1.0$이면 균형점이 $\approx9\times10^{-6}$으로 사실상 0에 고정됨 — eps를 고쳐도 이 정규화가 즉시 다시 눌러버림. `day_bias`($n=366$), `month_bias`($n=12$)도 같은 계산으로 현재 $\lambda=0.1$이 필요한 값보다 4~5자리 수 크다는 게 확인됨.
   **결론**: $\lambda_{hour},\lambda_d,\lambda_m$의 실효 탐색범위는 기존에 가정했던 "적당한" 구간(0.001~10)이 아니라 **1e-8~1e-2 수준까지 훨씬 낮게** 잡아야 함 (`sweep_pinn.py`에 반영).

### 11.3 하이퍼파라미터 sweep 설계

8개 파라미터($\gamma$, $\lambda_{Betz,bc,flat,smooth,hour,day,month}$) 전체 grid search는 비현실적(3값씩만 잡아도 $3^8$). Optuna(TPE) 기반 랜덤 sweep으로 전환, 목적함수는 time-holdout 3그룹 평균 `total_score`. 탐색 속도를 위해 sweep 중엔 epoch을 500+500→200+200으로 줄이고, 최적값을 찾은 후 전체 epoch으로 재학습해서 최종 검증. 현재 기본값(수기 설정값)을 `study.enqueue_trial`로 첫 trial에 강제 포함해 baseline 대비 개선 여부를 항상 비교 가능하게 함.

`train_pinn.py`의 `LAMBDA`/`GAMMA` 기본값은 이 sweep에서 찾은 best trial(time-holdout score 0.548/0.550/0.505) 값으로 고정되어 있음 — `STAGE2_EPOCHS=2000`도 이 조합에서 bias가 의미있는 크기까지 자라는 데 필요한 epoch 수.

### 11.4 물리 자유도 축소 재설계 시도 — 채택 안 함 (기록용)

기존 구조(전 구간 $C_{all}$ MLP 32x2 + $g_{doy}/g_{moy}$ + NMAE·FICR soft-relaxed loss)에서 "물리쪽 자유도가 너무 높다"는 문제의식으로 아래 재설계를 시도:

- $C_{all}$을 cut-in~rated 구간 전용 소형 MLP(`RampNet`, hidden=8~16)로 축소, rated~cutout은 닫힌식 $P_{rated}/(0.5\rho A v^3)$으로 대체
- $g_{doy}$/$g_{moy}$(중복되는 연주기 두 개) 삭제 → bias 계층(`day_bias`/`month_bias`/신규 `hod_bias`)으로 전부 이관
- 손실함수를 NMAE+FICR soft-relaxation에서 표준 MSE로 교체 (step-function 민감도로 인한 노이즈 회피 목적)
- 그룹 집계 시 "터빈 전체가 동시에 정격을 낸다"는 가정이 만드는 과대예측을 보정하기 위한 `GroupScale`(그룹당 스칼라 배율) 추가

**결과: 기존 대비 성능이 계속 낮게 나옴**(0.51~0.53 vs 기존 0.548/0.550/0.505). 원인 진단 과정에서 나온 핵심 발견들:

1. **풍속 구간별 오차 패턴**: rated 근처에서 큰 과대예측(+12~+23%p), 5~9m/s에서 과소예측(-4~-10%p) — 개별 터빈의 "완만히 증가→rated에서 꺾여 평평"한 kinked 파워커브를, 터빈간 풍속 편차를 무시한 채 대표풍속 하나로 평가해서 생기는 **Jensen 부등식 갭**(꺾이는 지점에서 최대, 멀수록 감소)으로 추정. 상수 배율(`GroupScale`) 하나로는 이 v-의존적 갭을 못 잡음 — 제대로 하려면 풍속 분포(평균+학습가능한 표준편차)에 대해 개별 커브를 적분(가우시안 quadrature)해야 함 (미구현, 다음 시도 후보)
2. **bias 계층 재검증**: train(2022-2023)에서 학습한 bias를 val(2024)에 적용했을 때 잔차 분산 설명력을 직접 측정 — `day_bias`/`month_bias`는 **-17%~-39%, -1%~-2%** (오히려 악화, 순수 train기간 과적합)인 반면 `hod_bias`(시간대)는 **+7~+9%**로 유일하게 진짜 일반화되는 신호였음. `day`/`month`는 "매년 반복되는 계절성"이 아니라 "그 해 한정 날씨지속성(autocorrelation)"을 external-year 검증 없이(같은 연도 내 groupby) 측정하면 44%까지 과대평가되는 착시가 있었음 — 반드시 train/val 분리해서 재현성을 검증해야 함
3. bias 크기가 작은 건 정규화(weight_decay) 문제가 아니었음 — weight_decay=0으로 꺼도 크기 거의 불변, `BIAS_LR`을 5배 올려도 (day/month가 섞여있는 상태에서는) train은 좋아지고 val은 그대로/악화 — 전형적 과적합 신호

**결론**: 이 재설계 자체(닫힌식, hod-only bias, MSE loss)의 방향성은 타당해 보이지만 아직 기존 구조의 성능을 못 넘었고, 특히 GroupScale의 v-의존성 부족이 병목으로 보임. `models/pinn.py`/`train_pinn.py`는 일단 기존 최고 성능 구성으로 복원(revert)했고, 위 발견들(Jensen 갭 보정, hod_bias 전용 강화)은 추후 별도로 재시도할 후보로 남겨둠.

### 11.5 현재 채택 상태 — hod-only bias만 부분 채택, 나머지는 원복

11.4의 재설계 3요소(닫힌식, MSE loss, hod-only bias) 중 **hod-only bias만 따로 떼서** 기존(전 구간 $C_{all}$ + $g_{doy}/g_{moy}$ + NMAE·FICR soft loss) 위에 얹어봄:

| 구성 | group_1 | group_2 | group_3 |
|---|---|---|---|
| 기존 최고 (hour+day+month bias, 378개 파라미터) | 0.5482 | 0.5516 | 0.5051 |
| 닫힌식+MSE+hod-only (11.4 재설계 전체) | 0.5382 | 0.5185 | 0.4988 |
| **기존 물리+NMAE·FICR loss + hod-only bias (24개 파라미터)** | 0.5475 | 0.5491 | 0.5049 |

세 번째 조합이 **기존 최고와 사실상 동급**(런간 노이즈 수준 차이, 0.001~0.002)이면서 bias 파라미터가 378개→24개로 줄었음 — day/month bias가 검증 성능에 실질적으로 기여한 게 없었다는 11.4의 발견이 그대로 재확인됨. **현재 코드(`models/pinn.py`, `train_pinn.py`, `utils/pinn_losses.py`)는 이 조합(전체 물리 백본 + NMAE·FICR soft loss + hod-only bias)으로 맞춰져 있음.**

MSE loss와 닫힌식(GroupScale 포함)은 각각 독립적으로 손해였음이 확인됐으니 채택 안 함 — 닫힌식/Jensen's gap 보정은 여전히 유효한 다음 시도 후보로 남지만, 붙이려면 MSE가 아니라 원래 NMAE·FICR loss 위에서 다시 시도해야 함.

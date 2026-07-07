# Weather Calibration Plan

## 목적

SCADA를 test feature처럼 예측해서 붙이는 것이 아니라, train 기간 SCADA를 supervision으로 사용해 LDAPS/GFS weather 자체를 터빈이 받은 weather에 가깝게 보정한다.

핵심 목표:

```text
raw LDAPS/GFS weather field
-> SCADA-supervised weather calibrator
-> corrected turbine/site weather
-> power model or PINN
```

## 이번 트랙에서 제외할 구조

다음 구조는 teacher feature 추가 실험으로 분류하고, weather calibration 트랙에서는 제외한다.

```text
raw weather 500 features
+ predicted SCADA/teacher features
-> tree model
```

이 방식이 항상 나쁘다는 뜻은 아니다. 기존 앙상블, gating, diagnostic, baseline 비교에는 계속 쓸 수 있다. 다만 이번 목표인 "SCADA로 weather 자체를 보정한다"와는 다른 실험으로 분리한다.

특히 아래 방식은 weather calibration 본실험으로 보지 않는다.

- `recon_*` feature를 기존 `all_meteo_compact_v2` 옆에 붙이는 방식.
- SCADA availability, response ratio, site wind prediction을 raw tree feature로 그대로 추가하는 방식.
- downstream model이 raw weather shortcut을 그대로 타는 방식.

## Calibrator

Calibrator는 발전량이 아니라 SCADA turbine weather를 맞춘다.

입력 후보:

- LDAPS/GFS grid-level wind vector.
- height별 u/v: GFS 10m, 80m, 100m, 850hPa, LDAPS 10m, 50m.
- spatial 후보: nearest, IDW, upwind/downwind, grid gradient.
- 최소한의 시간/계절 보정.

타겟:

- turbine별 SCADA `ws`.
- 가능하면 turbine별 SCADA `wd`.

출력:

- `corr_turbine_XX_u`
- `corr_turbine_XX_v`
- `corr_turbine_XX_ws`
- `corr_turbine_XX_wd`
- `corr_site_ws_mean/p50/p90/std`
- `corr_site_ws_cubic_mean`
- `corr_density_x_ws3`
- `corr_power_curve_sum`

## Validation

outer year-fold를 반드시 유지한다.

```text
2022,2023 -> 2024
2022,2024 -> 2023
2023,2024 -> 2022
```

각 fold에서:

- train years SCADA만 calibrator 학습에 사용.
- holdout year SCADA는 calibrator 학습에 사용하지 않음.
- train rows는 가능하면 inner OOF corrected weather로 생성.
- 최종 평가는 teacher R2가 아니라 power score 기준.

## Power Model

첫 실험은 raw weather를 제거하고 corrected weather 중심으로만 간다.

1. corrected weather + empirical power curve baseline.
2. corrected weather + small residual tree.
3. corrected weather 기반 PINN.

성공 기준:

- validation mean score `+0.01` 이상이면 submission 후보.
- `+0.001~0.002` 수준은 로그만 남기고 보류.

# PINN Data Strategy Notes

## Current State

The original PINN used a very narrow weather input:

- LDAPS 10m wind averaged over 16 grids
- power-law extrapolation to 117m hub height
- manufacturer-level linear correction to SCADA wind speed
- LDAPS temperature and surface pressure for air density
- calendar features for small harmonic/bias terms

That was physically clean but under-informed. Time-holdout scores stayed around:

| Version | group_1 | group_2 | group_3 | Avg |
|---|---:|---:|---:|---:|
| Original HOD-only bias | 0.5656 | 0.5657 | 0.5459 | 0.5590 |

## Fixes Already Made

### EMA Tail Preservation

The finite EMA convolution now folds the infinite tail mass into the oldest lag. This preserves steady-state gain:

```text
w_k = alpha(1-alpha)^k, k < K
w_K = (1-alpha)^K
```

This was mathematically correct but only moved score slightly because learned `tau` is near 2 hours.

### Capacity-Normalized Bias

The original `hod_bias` was learned directly in kWh. With Adam and `lr=1e-2`, it only reached about `20 kWh`, while true HOD residual patterns were around `1,000-2,000 kWh`.

Biases are now parameterized as capacity ratios:

```text
prediction += capacity * bias_ratio
```

This made HOD bias actually absorb the repeating diurnal residual.

### Evaluation Clamp

Training still uses raw predictions to preserve gradients. Evaluation clamps only for metric calculation:

```text
pred_eval = clip(pred_raw, 0, capacity)
```

Submission generation should also always clamp.

## Data Improvements That Worked

### GFS 850 hPa Wind

`gfs_ws850_speed` had high time-holdout feature importance in the tree models, especially for group_2. Adding it to the forecast-to-SCADA wind correction improved PINN substantially.

| Version | group_1 | group_2 | group_3 | Avg |
|---|---:|---:|---:|---:|
| HOD-only baseline | 0.5656 | 0.5657 | 0.5459 | 0.5590 |
| + GFS 850 wind correction | 0.5841 | 0.5871 | 0.5613 | 0.5775 |

### Expanded Forecast-to-SCADA Wind Correction

The current best PINN wind correction uses standardized ridge regression from forecast variables to SCADA wind speed:

- LDAPS hub-proxy wind `v`
- LDAPS 50m max/min wind speed
- LDAPS boundary-layer wind speed
- LDAPS boundary-layer height
- GFS 80m wind speed
- GFS 100m wind speed
- GFS 850 hPa wind speed
- GFS 850 hPa `u/v`
- GFS surface gust

Current score:

| Version | group_1 | group_2 | group_3 | Avg |
|---|---:|---:|---:|---:|
| Expanded wind correction | 0.5961 | 0.6050 | 0.5644 | 0.5885 |

This is the strongest evidence so far: better physical inputs matter much more than tweaking the neural power curve.

## Data Improvements With Mixed Results

### LDAPS Grid Spread as Wind Distribution

We added `v_std` from the 16 LDAPS grid wind-speed standard deviation and integrated power over a Gaussian wind distribution:

```text
P_group = n_turbines * E[P(v_mean + sigma Z)]
sigma = floor + scale * v_std
```

This helped UNISON but hurt VESTAS when applied globally. Current default:

```text
VESTAS wind distribution: off
UNISON wind distribution: on
```

## Data Improvements That Did Not Help Yet

### Month Bias and Train-Year Bias

Capacity-normalized `moy_bias` and train-only `year_bias` were tested. They reduced validation score. Likely reason:

- month signal is already partly represented by `g_moy`
- month bias learns train-period weather persistence rather than truly repeating seasonality
- train-only year bias helps train residuals but does not apply to validation/test

Current default keeps only HOD bias.

### Train-Only Per-Hour Bias

Train-row-only bias strongly reduces training loss but does not improve validation score. It can steal some signal from HOD bias. Current default disables it.

## Next Direction: SCADA as Teacher Signal

SCADA should not be used directly at test time, but it can teach the model how forecast variables map to actual site conditions.

Recommended next experiment:

1. Build hourly SCADA teacher targets per group:
   - `scada_ws_mean`
   - `scada_ws_std`
   - `scada_ws_p10`
   - `scada_ws_p50`
   - `scada_ws_p90`
   - optional wind direction sin/cos means
2. Train forecast-to-SCADA converters using only forecast variables available in train/test.
3. Feed predicted SCADA-like wind distribution into PINN:
   - `v_mean = predicted scada_ws_mean`
   - `sigma = predicted scada_ws_std` or quantile-derived spread
4. Keep the physics model focused on power generation, not on correcting bad wind inputs.

This is test-safe because SCADA is used only as a training target for a forecast-to-site-condition mapper.

## SCADA Teacher Result

Implemented a first version of the teacher:

```text
forecast variables -> group-level SCADA wind distribution
PINN v      = predicted scada_ws_mean
PINN v_std  = 0.5 * predicted scada_ws_std + 0.5 * (p90 - p10) / 2.563
```

Two validation modes were checked:

| Mode | group_1 | group_2 | group_3 | Avg | Note |
|---|---:|---:|---:|---:|---|
| Teacher fit on all SCADA | 0.7056 | 0.6869 | 0.6349 | 0.6758 | Upper bound; leaks 2024 SCADA into 2024 holdout |
| Teacher fit only before 2024 | 0.6341 | 0.6339 | 0.5886 | 0.6189 | Honest 2022-2023 -> 2024 validation |

This confirms the working hypothesis: the main bottleneck is site wind reconstruction.
Even without 2024 SCADA leakage, the SCADA teacher lifted the PINN from about `0.5885`
to about `0.6189` on the time holdout.

## Working Hypothesis

The main bottleneck is not the governing equation. It is site wind reconstruction.

```text
forecast fields -> actual group/turbine wind distribution -> physical power
```

The closer we get the middle term, the more useful the PINN becomes.

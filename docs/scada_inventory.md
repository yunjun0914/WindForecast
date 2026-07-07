# SCADA Inventory

## Raw SCADA Files

### `data/train/scada_vestas_train.csv`

- shape: `157819 x 37`
- time range: `2022-01-01 01:00:00` to `2025-01-01 00:00:00`
- turbines: `vestas_wtg01` to `vestas_wtg12`
- column families:
  - `*_power_kw10m`: 12 columns
  - `*_ws`: 12 columns
  - `*_wd`: 12 columns

### `data/train/scada_unison_train.csv`

- shape: `105264 x 16`
- time range: `2023-01-01 00:10:00` to `2025-01-01 00:00:00`
- turbines: `unison_wtg01` to `unison_wtg05`
- column families:
  - `*_power_kw10m`: 5 columns
  - `*_ws`: 5 columns
  - `*_wd`: 5 columns

## Group Mapping

Defined in `utils/power_curve.py`.

| group | SCADA turbines | raw file |
|---|---|---|
| `kpx_group_1` | `vestas_wtg01` to `vestas_wtg06` | `scada_vestas_train.csv` |
| `kpx_group_2` | `vestas_wtg07` to `vestas_wtg12` | `scada_vestas_train.csv` |
| `kpx_group_3` | `unison_wtg01` to `unison_wtg05` | `scada_unison_train.csv` |

## Currently Used By Canonical SCADA-Teacher PINN

Canonical entrypoints:

- Historical scripts `predict_pinn_scada_teacher.py` and `predict_pinn_scada_teacher_year_bagging.py` were removed during cleanup.
- Current PINN entrypoint: `predict_pinn_effective_grid_g1_year_bagging.py`

The canonical PINN path uses SCADA only to build teacher targets. It does not use raw SCADA at test time.

Used raw SCADA columns:

```text
*_ws
```

Teacher targets built from hourly turbine wind-speed distributions:

```text
scada_ws_mean
scada_ws_std
scada_ws_p10
scada_ws_p50
scada_ws_p75
scada_ws_p90
scada_ws_max
scada_ws_cubic
scada_ws_ramp
```

Current group recipe:

| group | teacher source | PINN wind input |
|---|---|---|
| `kpx_group_1` | VESTAS group1 SCADA `ws` | `scada_ws_cubic` |
| `kpx_group_2` | VESTAS group2 SCADA `ws` | `scada_ws_p90` |
| `kpx_group_3` | 30% UNISON group3 `p90` + 70% VESTAS group2 `p90` | blended `p90` |

`v_std` is derived from predicted SCADA wind-speed spread:

```text
0.5 * scada_ws_std + 0.5 * (scada_ws_p90 - scada_ws_p10) / 2.563
```

## Used Elsewhere

### Tree / Power-Curve Features

`utils/power_curve.py` uses:

```text
*_ws
*_power_kw10m
```

It fits empirical turbine power curves from pooled SCADA `(wind speed, power)` samples.

### Direction Experiments

`utils/scada_direction.py` uses:

```text
*_wd
```

It builds direction targets:

```text
scada_wd_sin
scada_wd_cos
scada_wd_concentration
scada_wd_sin_std
scada_wd_cos_std
```

This is not part of the current canonical SCADA-teacher PINN.

## Quick Data Notes

VESTAS:

- `ws` missing: `0.0%`
- `wd` missing: `0.0%`
- `power_kw10m` missing: `0.0%`
- `power_kw10m` has outliers: `868 / 1,893,828` values have `abs(power) > 3600`
- `power_kw10m` has negative values: `317,221`

UNISON:

- `ws` missing: about `0.97%`
- `wd` missing: about `0.42%`
- `power_kw10m` missing: about `0.42%`
- no `abs(power) > 4200` values observed

## Candidate Next Checks

- SCADA `wd` is available but not used in the current canonical path.
- Plain wind direction features were previously weak, but terrain-aware directional features may still be worth testing:
  - southwest wind component
  - group ridge-axis along/cross wind components
  - direction x wind-speed interactions
- VESTAS `power_kw10m` outliers should be handled before relying heavily on tree power-curve features.

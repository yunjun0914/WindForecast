import numpy as np
import pandas as pd

from utils.pinn_physics import SINGLE_TURBINE_CAPACITY_W

GROUP_TURBINE_PREFIXES = {
    "kpx_group_1": [f"vestas_wtg{i:02d}" for i in range(1, 7)],
    "kpx_group_2": [f"vestas_wtg{i:02d}" for i in range(7, 13)],
    "kpx_group_3": [f"unison_wtg{i:02d}" for i in range(1, 6)],
}

GROUP_N_TURBINES = {group: len(prefixes) for group, prefixes in GROUP_TURBINE_PREFIXES.items()}

GROUP_MANUFACTURER = {
    "kpx_group_1": "vestas",
    "kpx_group_2": "vestas",
    "kpx_group_3": "unison",
}

POWER_CURVE_CAP_TOL = 1.02


def _power_10m_capacity_kwh(group):
    manufacturer = GROUP_MANUFACTURER[group]
    return SINGLE_TURBINE_CAPACITY_W[manufacturer] / 1000.0 / 6.0


def fit_power_curve(wind_speed, power, bin_width=0.5, max_speed=30.0):
    """Bin-averaged empirical power curve. A binned lookup (rather than a monotonic
    regression) is used because real curves are non-monotonic: power rises with speed
    up to rated, then drops to zero past cutout."""
    bins = np.arange(0, max_speed + bin_width, bin_width)
    bin_idx = np.clip(np.digitize(wind_speed, bins), 1, len(bins) - 1)
    bin_centers = bins[:-1] + bin_width / 2

    curve = np.full(len(bin_centers), np.nan)
    for i in range(1, len(bins)):
        mask = bin_idx == i
        if mask.any():
            curve[i - 1] = power[mask].mean()
    curve = pd.Series(curve).ffill().bfill().to_numpy()

    def curve_fn(ws):
        ws = np.clip(np.asarray(ws, dtype=float), bin_centers[0], bin_centers[-1])
        return np.interp(ws, bin_centers, curve)

    return curve_fn


def fit_group_power_curve(scada_df, group, clean=False):
    """Fit one power curve per KPX group by pooling (wind_speed, power) samples
    across that group's turbines from the training-period SCADA data."""
    ws_parts, power_parts = [], []
    for prefix in GROUP_TURBINE_PREFIXES[group]:
        ws_col, power_col = f"{prefix}_ws", f"{prefix}_power_kw10m"
        ws_parts.append(scada_df[ws_col].to_numpy())
        power_parts.append(scada_df[power_col].to_numpy())
    ws = np.concatenate(ws_parts)
    power = np.concatenate(power_parts)
    valid = ~(np.isnan(ws) | np.isnan(power))
    if clean:
        cap_10m = _power_10m_capacity_kwh(group) * POWER_CURVE_CAP_TOL
        valid &= (ws >= 0) & (ws <= 35) & (power >= 0) & (power <= cap_10m)
    return fit_power_curve(ws[valid], power[valid])


def add_power_curve_feature(df, wind_speed_col, curve_fn, n_turbines, out_col="power_curve_est"):
    """Apply a fitted (train-only) curve to forecast wind speed -- safe for both
    train and test periods since the curve itself never sees test-period data."""
    df = df.copy()
    df[out_col] = curve_fn(df[wind_speed_col].to_numpy()) * n_turbines
    return df

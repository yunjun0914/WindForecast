import numpy as np
import pandas as pd


TIME_CONTEXT_BASE_COLS = [
    "gfs_ws10_speed",
    "gfs_ws100_speed",
    "gfs_ws850_speed",
    "gfs_surface_0_gust",
    "ldaps_ws10_speed",
    "ldaps_ws50_max_speed",
    "phys_gfs_ws850_grid_max",
    "phys_gfs_ws850_grid_p90",
    "phys_ldaps_ws50max_grid_max",
]


def weather_time_feature_names(base_cols=TIME_CONTEXT_BASE_COLS):
    names = []
    for col in base_cols:
        names.extend(
            [
                f"{col}_lag1",
                f"{col}_lag3",
                f"{col}_lead1",
                f"{col}_lead3",
                f"{col}_roll3_mean",
                f"{col}_roll3_std",
                f"{col}_roll6_mean",
                f"{col}_roll6_std",
                f"{col}_roll6_max",
                f"{col}_lead1_minus_lag1",
            ]
        )
    return names


def add_weather_time_features(df, base_cols=TIME_CONTEXT_BASE_COLS):
    """Add compact NWP time-context features without using target history."""
    out = df.sort_values("forecast_kst_dtm").copy()
    issue_col = (
        "data_available_kst_dtm"
        if "data_available_kst_dtm" in out.columns
        else None
    )

    for col in base_cols:
        if col not in out.columns:
            continue

        current = out[col].astype(float)
        if issue_col is None:
            lag1 = current.shift(1).fillna(current)
            lag3 = current.shift(3).fillna(current)
            lead1 = current.shift(-1).fillna(current)
            lead3 = current.shift(-3).fillna(current)
            roll3_mean = current.rolling(window=3, min_periods=1).mean()
            roll3_std = current.rolling(window=3, min_periods=1).std(ddof=0)
            roll6_mean = current.rolling(window=6, min_periods=1).mean()
            roll6_std = current.rolling(window=6, min_periods=1).std(ddof=0)
            roll6_max = current.rolling(window=6, min_periods=1).max()
        else:
            grouped = out.groupby(issue_col, sort=False)[col]
            lag1 = grouped.shift(1).fillna(current)
            lag3 = grouped.shift(3).fillna(current)
            lead1 = grouped.shift(-1).fillna(current)
            lead3 = grouped.shift(-3).fillna(current)
            roll3_mean = grouped.transform(
                lambda values: values.rolling(window=3, min_periods=1).mean()
            )
            roll3_std = grouped.transform(
                lambda values: values.rolling(window=3, min_periods=1).std(ddof=0)
            )
            roll6_mean = grouped.transform(
                lambda values: values.rolling(window=6, min_periods=1).mean()
            )
            roll6_std = grouped.transform(
                lambda values: values.rolling(window=6, min_periods=1).std(ddof=0)
            )
            roll6_max = grouped.transform(
                lambda values: values.rolling(window=6, min_periods=1).max()
            )

        out[f"{col}_lag1"] = lag1
        out[f"{col}_lag3"] = lag3
        out[f"{col}_lead1"] = lead1
        out[f"{col}_lead3"] = lead3
        out[f"{col}_roll3_mean"] = roll3_mean
        out[f"{col}_roll3_std"] = roll3_std.replace(np.nan, 0.0)
        out[f"{col}_roll6_mean"] = roll6_mean
        out[f"{col}_roll6_std"] = roll6_std.replace(np.nan, 0.0)
        out[f"{col}_roll6_max"] = roll6_max
        out[f"{col}_lead1_minus_lag1"] = lead1 - lag1

    return out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0).reset_index(drop=True)

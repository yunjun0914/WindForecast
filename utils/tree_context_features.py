import numpy as np
import pandas as pd

from utils.preprocessing import TIME_KEY_COLS


FORECAST_CYCLE_FEATURES = [
    "forecast_lead_hours",
    "forecast_lead_mod24_sin",
    "forecast_lead_mod24_cos",
    "data_available_hod_sin",
    "data_available_hod_cos",
    "data_available_dow_sin",
    "data_available_dow_cos",
]

DIRECTION_VECTOR_PAIRS = [
    ("gfs_heightAboveGround_10_10u", "gfs_heightAboveGround_10_10v", "gfs_ws10"),
    ("gfs_heightAboveGround_100_100u", "gfs_heightAboveGround_100_100v", "gfs_ws100"),
    ("gfs_isobaricInhPa_850_u", "gfs_isobaricInhPa_850_v", "gfs_ws850"),
    ("ldaps_heightAboveGround_10_10u", "ldaps_heightAboveGround_10_10v", "ldaps_ws10"),
    ("ldaps_heightAboveGround_50_50MUmax", "ldaps_heightAboveGround_50_50MVmax", "ldaps_ws50max"),
]

DIRECTION_VEER_FEATURES = [
    "phys_gfs_veer_ws100_ws10_sin",
    "phys_gfs_veer_ws100_ws10_cos",
    "phys_gfs_veer_ws100_ws10_abs",
    "phys_gfs_veer_ws850_ws100_sin",
    "phys_gfs_veer_ws850_ws100_cos",
    "phys_gfs_veer_ws850_ws100_abs",
    "phys_ldaps_veer_ws50max_ws10_sin",
    "phys_ldaps_veer_ws50max_ws10_cos",
    "phys_ldaps_veer_ws50max_ws10_abs",
]

SPATIAL_Q_LEVELS = [
    ("gfs", "heightAboveGround_10_10u", "heightAboveGround_10_10v", "ws10"),
    ("gfs", "heightAboveGround_100_100u", "heightAboveGround_100_100v", "ws100"),
    ("gfs", "isobaricInhPa_850_u", "isobaricInhPa_850_v", "ws850"),
    ("ldaps", "heightAboveGround_10_10u", "heightAboveGround_10_10v", "ws10"),
    ("ldaps", "heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax", "ws50max"),
]

SPATIAL_EXISTING_FEATURES = [
    "phys_gfs_ws10_grid_std",
    "phys_gfs_ws10_near_minus_grid_mean",
    "phys_gfs_ws100_grid_std",
    "phys_gfs_ws100_near_minus_grid_mean",
    "phys_gfs_ws850_grid_std",
    "phys_gfs_ws850_near_minus_grid_mean",
    "phys_ldaps_ws10_grid_std",
    "phys_ldaps_ws10_near_minus_grid_mean",
    "phys_ldaps_ws50max_grid_std",
    "phys_ldaps_ws50max_near_minus_grid_mean",
]


def forecast_cycle_feature_names():
    return list(FORECAST_CYCLE_FEATURES)


def direction_context_feature_names():
    names = []
    for _, _, prefix in DIRECTION_VECTOR_PAIRS:
        names.extend([f"{prefix}_dir_cos", f"{prefix}_dir_sin"])
    names.extend(DIRECTION_VEER_FEATURES)
    return names


def spatial_context_feature_names():
    names = []
    for source, _, _, level in SPATIAL_Q_LEVELS:
        names.extend(
            [
                f"phys_{source}_{level}_grid_q25",
                f"phys_{source}_{level}_grid_q75",
                f"phys_{source}_{level}_grid_iqr",
            ]
        )
    names.extend(SPATIAL_EXISTING_FEATURES)
    return names


def add_forecast_cycle_features(df):
    out = df.copy()
    forecast = pd.to_datetime(out["forecast_kst_dtm"])
    available = pd.to_datetime(out["data_available_kst_dtm"])
    lead_hours = (forecast - available).dt.total_seconds() / 3600.0
    lead_mod24 = np.mod(lead_hours, 24.0)

    out["forecast_lead_hours"] = lead_hours
    out["forecast_lead_mod24_sin"] = np.sin(2.0 * np.pi * lead_mod24 / 24.0)
    out["forecast_lead_mod24_cos"] = np.cos(2.0 * np.pi * lead_mod24 / 24.0)

    available_hod = available.dt.hour
    available_dow = available.dt.dayofweek
    out["data_available_hod_sin"] = np.sin(2.0 * np.pi * available_hod / 24.0)
    out["data_available_hod_cos"] = np.cos(2.0 * np.pi * available_hod / 24.0)
    out["data_available_dow_sin"] = np.sin(2.0 * np.pi * available_dow / 7.0)
    out["data_available_dow_cos"] = np.cos(2.0 * np.pi * available_dow / 7.0)
    return out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)


def add_direction_context_features(df):
    out = df.copy()
    eps = 1e-6
    for u_col, v_col, prefix in DIRECTION_VECTOR_PAIRS:
        if u_col not in out.columns or v_col not in out.columns:
            continue
        u = out[u_col].astype(float)
        v = out[v_col].astype(float)
        speed = np.sqrt(u**2 + v**2).clip(lower=eps)
        out[f"{prefix}_dir_cos"] = u / speed
        out[f"{prefix}_dir_sin"] = v / speed
    return out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)


def _source_spatial_quantiles(df, source):
    work = df.copy()
    work["forecast_kst_dtm"] = pd.to_datetime(work["forecast_kst_dtm"])
    frames = []

    for level_source, u_col, v_col, level in SPATIAL_Q_LEVELS:
        if level_source != source or u_col not in work.columns or v_col not in work.columns:
            continue
        speed_col = f"_{source}_{level}_speed"
        work[speed_col] = np.sqrt(work[u_col].astype(float) ** 2 + work[v_col].astype(float) ** 2)
        q = work.groupby("forecast_kst_dtm")[speed_col].quantile([0.25, 0.75]).unstack().reset_index()
        q = q.rename(
            columns={
                0.25: f"phys_{source}_{level}_grid_q25",
                0.75: f"phys_{source}_{level}_grid_q75",
            }
        )
        q[f"phys_{source}_{level}_grid_iqr"] = (
            q[f"phys_{source}_{level}_grid_q75"] - q[f"phys_{source}_{level}_grid_q25"]
        )
        frames.append(q)

    if not frames:
        return pd.DataFrame(columns=["forecast_kst_dtm"])

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="forecast_kst_dtm", how="outer")
    return out


def add_spatial_context_features(df, ldaps, gfs):
    out = df.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    quantiles = _source_spatial_quantiles(gfs, "gfs").merge(
        _source_spatial_quantiles(ldaps, "ldaps"), on="forecast_kst_dtm", how="outer"
    )
    out = out.merge(quantiles, on="forecast_kst_dtm", how="left")
    return out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)


def add_tree_context_features(df, ldaps, gfs):
    out = add_forecast_cycle_features(df)
    out = add_direction_context_features(out)
    out = add_spatial_context_features(out, ldaps, gfs)
    return out


def tree_context_feature_names():
    return forecast_cycle_feature_names() + direction_context_feature_names() + spatial_context_feature_names()

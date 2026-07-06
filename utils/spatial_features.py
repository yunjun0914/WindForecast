import numpy as np
import pandas as pd

from utils.preprocessing import TIME_KEY_COLS, haversine_km
from utils.site_metadata import group_site_summary


SPATIAL_LDAPS_LEVELS = [
    ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "ws10"),
    ("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax", "ws50max"),
    ("heightAboveGround_50_50MUmin", "heightAboveGround_50_50MVmin", "ws50min"),
    ("heightAboveGround_5_XBLWS", "heightAboveGround_5_YBLWS", "ws5bl"),
]

SPATIAL_GFS_LEVELS = [
    ("heightAboveGround_80_u", "heightAboveGround_80_v", "ws80"),
    ("heightAboveGround_100_100u", "heightAboveGround_100_100v", "ws100"),
    ("isobaricInhPa_850_u", "isobaricInhPa_850_v", "ws850"),
    ("isobaricInhPa_700_u", "isobaricInhPa_700_v", "ws700"),
]


def _xy_offsets_km(df, target_lat, target_lon):
    dx = (df["longitude"].to_numpy(float) - target_lon) * 111.32 * np.cos(np.radians(target_lat))
    dy = (df["latitude"].to_numpy(float) - target_lat) * 110.57
    return dx, dy


def _weighted_mean(df, value_col, weight_col):
    tmp = df[[*TIME_KEY_COLS, value_col, weight_col]].copy()
    tmp["_wx"] = tmp[value_col] * tmp[weight_col]
    grouped = tmp.groupby(TIME_KEY_COLS, as_index=False).agg(_wx=("_wx", "sum"), _w=(weight_col, "sum"))
    return grouped["_wx"] / grouped["_w"].replace(0, np.nan)


def _nearest_grid_id(df, target_lat, target_lon):
    grids = df[["grid_id", "latitude", "longitude"]].drop_duplicates()
    dist = haversine_km(target_lat, target_lon, grids["latitude"], grids["longitude"])
    return grids.loc[dist.idxmin(), "grid_id"]


def build_source_spatial_features(df, source, group, mode="all"):
    sites = group_site_summary()
    site = sites[group]
    target_lat, target_lon = site["latitude"], site["longitude"]
    axis_x, axis_y = site["axis_x"], site["axis_y"]
    levels = SPATIAL_LDAPS_LEVELS if source == "ldaps" else SPATIAL_GFS_LEVELS

    work = df.copy()
    dx, dy = _xy_offsets_km(work, target_lat, target_lon)
    dist = np.sqrt(dx**2 + dy**2)
    work["_dx"] = dx
    work["_dy"] = dy
    work["_dist"] = np.maximum(dist, 0.05)
    work["_inv_dist_w"] = 1.0 / (work["_dist"] + 0.3) ** 2

    feature_frames = []
    nearest_id = _nearest_grid_id(work, target_lat, target_lon)
    nearest = work[work["grid_id"] == nearest_id].copy()

    static_cols = []
    for col in ["surface_0_h", "surface_0_lsm"]:
        if col in work.columns:
            static_cols.append(col)
    if static_cols:
        stat = nearest[TIME_KEY_COLS + static_cols].rename(columns={c: f"{source}_{group}_nearest_{c}" for c in static_cols})
        feature_frames.append(stat)

    for u_col, v_col, prefix in levels:
        if u_col not in work.columns or v_col not in work.columns:
            continue
        speed_col = f"_{prefix}_speed"
        along_axis_col = f"_{prefix}_axis_along"
        cross_axis_col = f"_{prefix}_axis_cross"
        upwind_w_col = f"_{prefix}_upwind_w"

        work[speed_col] = np.sqrt(work[u_col] ** 2 + work[v_col] ** 2)
        work[along_axis_col] = work[u_col] * axis_x + work[v_col] * axis_y
        work[cross_axis_col] = -work[u_col] * axis_y + work[v_col] * axis_x

        if mode in {"nearest", "all"}:
            near_cols = nearest[TIME_KEY_COLS + [u_col, v_col]].copy()
            near_cols[f"{source}_{group}_near_{prefix}_speed"] = np.sqrt(near_cols[u_col] ** 2 + near_cols[v_col] ** 2)
            near_cols = near_cols.drop(columns=[u_col, v_col])
            feature_frames.append(near_cols)

        if mode in {"weighted", "all"}:
            out = work[TIME_KEY_COLS].drop_duplicates().sort_values("forecast_kst_dtm").reset_index(drop=True)
            out[f"{source}_{group}_idw_{prefix}_speed"] = _weighted_mean(work, speed_col, "_inv_dist_w")
            out[f"{source}_{group}_idw_{prefix}_axis_along"] = _weighted_mean(work, along_axis_col, "_inv_dist_w")
            out[f"{source}_{group}_idw_{prefix}_axis_cross"] = _weighted_mean(work, cross_axis_col, "_inv_dist_w")
            feature_frames.append(out)

        if mode in {"upwind", "all"}:
            safe_speed = work[speed_col].replace(0, np.nan)
            ux = work[u_col] / safe_speed
            uy = work[v_col] / safe_speed
            along_upwind = -(work["_dx"] * ux + work["_dy"] * uy)
            cross_upwind = np.abs(-work["_dx"] * uy + work["_dy"] * ux)
            gate = 1.0 / (1.0 + np.exp(-along_upwind / 2.0))
            work[upwind_w_col] = np.exp(-((cross_upwind / 2.0) ** 2)) * gate * work["_inv_dist_w"]

            out = work[TIME_KEY_COLS].drop_duplicates().sort_values("forecast_kst_dtm").reset_index(drop=True)
            out[f"{source}_{group}_upwind_{prefix}_speed"] = _weighted_mean(work, speed_col, upwind_w_col)
            out[f"{source}_{group}_upwind_{prefix}_axis_along"] = _weighted_mean(work, along_axis_col, upwind_w_col)
            out[f"{source}_{group}_upwind_{prefix}_axis_cross"] = _weighted_mean(work, cross_axis_col, upwind_w_col)
            feature_frames.append(out)

    merged = feature_frames[0]
    for frame in feature_frames[1:]:
        merged = merged.merge(frame, on=TIME_KEY_COLS, how="outer")
    merged["forecast_kst_dtm"] = pd.to_datetime(merged["forecast_kst_dtm"])
    return merged.sort_values("forecast_kst_dtm").ffill().fillna(0).reset_index(drop=True)


def add_group_spatial_features(weather, ldaps, gfs, group, mode="all"):
    out = weather.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    if mode in {"ldaps", "nearest", "weighted", "upwind", "all"}:
        ldaps_mode = mode if mode in {"nearest", "weighted", "upwind"} else "all"
        ld = build_source_spatial_features(ldaps, "ldaps", group, ldaps_mode)
        out = out.merge(ld.drop(columns=["data_available_kst_dtm"], errors="ignore"), on="forecast_kst_dtm", how="left")
    if mode in {"gfs", "all"}:
        gf = build_source_spatial_features(gfs, "gfs", group, "all")
        out = out.merge(gf.drop(columns=["data_available_kst_dtm"], errors="ignore"), on="forecast_kst_dtm", how="left")
    return out.sort_values("forecast_kst_dtm").ffill().fillna(0).reset_index(drop=True)

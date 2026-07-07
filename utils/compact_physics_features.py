import numpy as np
import pandas as pd

from utils.preprocessing import FARM_CENTROID, TIME_KEY_COLS, haversine_km
from utils.site_metadata import group_site_summary


LDAPS_LEVELS = [
    ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "ws10"),
    ("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax", "ws50max"),
    ("heightAboveGround_50_50MUmin", "heightAboveGround_50_50MVmin", "ws50min"),
    ("heightAboveGround_5_XBLWS", "heightAboveGround_5_YBLWS", "ws5bl"),
]

GFS_LEVELS = [
    ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "ws10"),
    ("heightAboveGround_80_u", "heightAboveGround_80_v", "ws80"),
    ("heightAboveGround_100_100u", "heightAboveGround_100_100v", "ws100"),
    ("planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v", "wspbl"),
    ("isobaricInhPa_850_u", "isobaricInhPa_850_v", "ws850"),
    ("isobaricInhPa_700_u", "isobaricInhPa_700_v", "ws700"),
    ("isobaricInhPa_500_u", "isobaricInhPa_500_v", "ws500"),
]


def _speed(u, v):
    return np.sqrt(u**2 + v**2)


def _target_for_group(group):
    if group is None:
        return FARM_CENTROID[0], FARM_CENTROID[1], 1.0, 0.0
    site = group_site_summary()[group]
    return site["latitude"], site["longitude"], site["axis_x"], site["axis_y"]


def _nearest_grid_id(df, target_lat, target_lon):
    grids = df[["grid_id", "latitude", "longitude"]].drop_duplicates()
    dist = haversine_km(target_lat, target_lon, grids["latitude"], grids["longitude"])
    return grids.loc[dist.idxmin(), "grid_id"]


def _grid_offsets(df, target_lat, target_lon):
    dx = (df["longitude"].to_numpy(float) - target_lon) * 111.32 * np.cos(np.radians(target_lat))
    dy = (df["latitude"].to_numpy(float) - target_lat) * 110.57
    dist = np.sqrt(dx**2 + dy**2)
    return dx, dy, np.maximum(dist, 0.05)


def _weighted_time_mean(df, value_col, weight_col):
    tmp = df[[*TIME_KEY_COLS, value_col, weight_col]].copy()
    tmp["_weighted"] = tmp[value_col] * tmp[weight_col]
    grouped = tmp.groupby(TIME_KEY_COLS, as_index=False).agg(_weighted=("_weighted", "sum"), _w=(weight_col, "sum"))
    grouped[value_col] = grouped["_weighted"] / grouped["_w"].replace(0, np.nan)
    return grouped[[*TIME_KEY_COLS, value_col]]


def _time_slope(df, value_col, coord_col, out_col):
    tmp = df[[*TIME_KEY_COLS, value_col, coord_col]].copy()
    means = tmp.groupby(TIME_KEY_COLS, as_index=False).agg(_coord_mean=(coord_col, "mean"), _value_mean=(value_col, "mean"))
    tmp = tmp.merge(means, on=TIME_KEY_COLS, how="left")
    tmp["_num"] = (tmp[coord_col] - tmp["_coord_mean"]) * (tmp[value_col] - tmp["_value_mean"])
    tmp["_den"] = (tmp[coord_col] - tmp["_coord_mean"]) ** 2
    out = tmp.groupby(TIME_KEY_COLS, as_index=False).agg(_num=("_num", "sum"), _den=("_den", "sum"))
    out[out_col] = out["_num"] / out["_den"].replace(0, np.nan)
    return out[[*TIME_KEY_COLS, out_col]]


def _angular_diff(u_high, v_high, u_low, v_low):
    high = np.arctan2(v_high, u_high)
    low = np.arctan2(v_low, u_low)
    return np.arctan2(np.sin(high - low), np.cos(high - low))


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _temp_kelvin(value):
    value = pd.Series(value, copy=False).astype(float)
    return np.where(value < 200.0, value + 273.15, value)


def _pressure_pa(value):
    value = pd.Series(value, copy=False).astype(float)
    return np.where(value < 2000.0, value * 100.0, value)


def _relative_humidity_fraction(value):
    value = pd.Series(value, copy=False).astype(float)
    return np.clip(np.where(value > 1.5, value / 100.0, value), 0.0, 1.0)


def _air_density(temp, pressure, relative_humidity=None, specific_humidity=None):
    temp_k = _temp_kelvin(temp)
    pressure_pa = _pressure_pa(pressure)
    if specific_humidity is not None:
        q = np.clip(pd.Series(specific_humidity, copy=False).astype(float).to_numpy(), 0.0, 0.05)
        virtual_temp = temp_k * (1.0 + 0.61 * q)
    elif relative_humidity is not None:
        rh = _relative_humidity_fraction(relative_humidity)
        temp_c = temp_k - 273.15
        vapor_pressure = rh * 611.2 * np.exp((17.67 * temp_c) / (temp_c + 243.5))
        virtual_temp = temp_k / np.maximum(1.0 - 0.378 * vapor_pressure / pressure_pa, 0.7)
    else:
        virtual_temp = temp_k
    return pressure_pa / (287.05 * virtual_temp)


def _aggregate_thermo(ldaps, gfs, target_lat, target_lon):
    ldaps = ldaps.copy()
    gfs = gfs.copy()
    ldaps["forecast_kst_dtm"] = pd.to_datetime(ldaps["forecast_kst_dtm"])
    gfs["forecast_kst_dtm"] = pd.to_datetime(gfs["forecast_kst_dtm"])
    frames = []

    ld_cols = [
        "heightAboveGround_2_t",
        "heightAboveGround_2_r",
        "heightAboveGround_2_q",
        "surface_0_sp",
        "meanSea_0_prmsl",
        "etc_0_blh",
        "surface_0_NDNSW",
        "surface_0_NDNLW",
        "surface_0_avg_lsprate",
        "surface_0_lssrate",
        "surface_0_ncpcp",
        "etc_0_hcc",
        "etc_0_mcc",
        "etc_0_lcc",
    ]
    ld_cols = [c for c in ld_cols if c in ldaps.columns]
    if ld_cols:
        ld = ldaps.groupby(TIME_KEY_COLS, as_index=False)[ld_cols].mean()
        ld["phys_ldaps_air_density"] = _air_density(
            ld["heightAboveGround_2_t"],
            ld["surface_0_sp"],
            relative_humidity=ld.get("heightAboveGround_2_r"),
            specific_humidity=ld.get("heightAboveGround_2_q"),
        )
        cloud_cols = [c for c in ["etc_0_hcc", "etc_0_mcc", "etc_0_lcc"] if c in ld.columns]
        precip_cols = [c for c in ["surface_0_avg_lsprate", "surface_0_lssrate", "surface_0_ncpcp"] if c in ld.columns]
        if cloud_cols:
            ld["phys_ldaps_cloud_sum"] = ld[cloud_cols].sum(axis=1)
        if precip_cols:
            ld["phys_ldaps_precip_sum"] = ld[precip_cols].sum(axis=1)
        rename = {
            "etc_0_blh": "phys_ldaps_blh",
            "surface_0_NDNSW": "phys_ldaps_shortwave",
            "surface_0_NDNLW": "phys_ldaps_longwave",
            "surface_0_sp": "phys_ldaps_surface_pressure",
        }
        keep = [*TIME_KEY_COLS, "phys_ldaps_air_density"]
        keep += [v for k, v in rename.items() if k in ld.columns]
        keep += [c for c in ["phys_ldaps_cloud_sum", "phys_ldaps_precip_sum"] if c in ld.columns]
        frames.append(ld.rename(columns=rename)[keep])

    grid_id = _nearest_grid_id(gfs, target_lat, target_lon)
    gf_cols = [
        "heightAboveGround_2_2t",
        "heightAboveGround_2_2r",
        "heightAboveGround_2_2sh",
        "surface_0_sp",
        "meanSea_0_prmsl",
        "planetaryBoundaryLayer_0_VRATE",
        "surface_0_gust",
        "surface_0_dswrf",
        "surface_0_dlwrf",
        "surface_0_prate",
        "surface_0_tp",
        "lowCloudLayer_0_lcc",
        "middleCloudLayer_0_mcc",
        "highCloudLayer_0_hcc",
        "atmosphere_0_tcc",
        "isobaricInhPa_850_t",
        "isobaricInhPa_700_t",
        "isobaricInhPa_500_t",
    ]
    gf_cols = [c for c in gf_cols if c in gfs.columns]
    if gf_cols:
        gf = gfs.loc[gfs["grid_id"] == grid_id, TIME_KEY_COLS + gf_cols].copy()
        gf["phys_gfs_air_density"] = _air_density(
            gf["heightAboveGround_2_2t"],
            gf["surface_0_sp"],
            relative_humidity=gf.get("heightAboveGround_2_2r"),
            specific_humidity=gf.get("heightAboveGround_2_2sh"),
        )
        cloud_cols = [c for c in ["lowCloudLayer_0_lcc", "middleCloudLayer_0_mcc", "highCloudLayer_0_hcc", "atmosphere_0_tcc"] if c in gf.columns]
        precip_cols = [c for c in ["surface_0_prate", "surface_0_tp"] if c in gf.columns]
        if cloud_cols:
            gf["phys_gfs_cloud_sum"] = gf[cloud_cols].sum(axis=1)
        if precip_cols:
            gf["phys_gfs_precip_sum"] = gf[precip_cols].sum(axis=1)
        if {"isobaricInhPa_850_t", "isobaricInhPa_500_t"}.issubset(gf.columns):
            gf["phys_gfs_lapse_850_500"] = _temp_kelvin(gf["isobaricInhPa_850_t"]) - _temp_kelvin(gf["isobaricInhPa_500_t"])
        rename = {
            "surface_0_sp": "phys_gfs_surface_pressure",
            "planetaryBoundaryLayer_0_VRATE": "phys_gfs_pbl_vrate",
            "surface_0_gust": "phys_gfs_gust",
            "surface_0_dswrf": "phys_gfs_shortwave",
            "surface_0_dlwrf": "phys_gfs_longwave",
        }
        keep = [*TIME_KEY_COLS, "phys_gfs_air_density"]
        keep += [v for k, v in rename.items() if k in gf.columns]
        keep += [c for c in ["phys_gfs_cloud_sum", "phys_gfs_precip_sum", "phys_gfs_lapse_850_500"] if c in gf.columns]
        frames.append(gf.rename(columns=rename)[keep])

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="forecast_kst_dtm", how="inner", suffixes=("", "_dup"))
        out = out.drop(columns=[c for c in out.columns if c.endswith("_dup")])
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    return out


def _source_spatial_stats(df, source, levels, target_lat, target_lon, axis_x, axis_y, include_advanced=False):
    work = df.copy()
    work["forecast_kst_dtm"] = pd.to_datetime(work["forecast_kst_dtm"])
    dx, dy, dist = _grid_offsets(work, target_lat, target_lon)
    work["_dx"] = dx
    work["_dy"] = dy
    work["_dist"] = dist
    work["_idw"] = 1.0 / (work["_dist"] + 0.3) ** 2
    nearest_id = _nearest_grid_id(work, target_lat, target_lon)
    frames = []

    for u_col, v_col, level in levels:
        if u_col not in work.columns or v_col not in work.columns:
            continue
        speed_col = f"_{level}_speed"
        along_col = f"_{level}_along_axis"
        cross_col = f"_{level}_cross_axis"
        sw_col = f"_{level}_southwest"
        work[speed_col] = _speed(work[u_col], work[v_col])
        work[along_col] = work[u_col] * axis_x + work[v_col] * axis_y
        work[cross_col] = -work[u_col] * axis_y + work[v_col] * axis_x
        work[sw_col] = -(work[u_col] + work[v_col]) / np.sqrt(2.0)

        grouped = work.groupby(TIME_KEY_COLS)[speed_col]
        stats = grouped.agg(["mean", "std", "min", "max"]).reset_index()
        stats = stats.rename(
            columns={
                "mean": f"phys_{source}_{level}_grid_mean",
                "std": f"phys_{source}_{level}_grid_std",
                "min": f"phys_{source}_{level}_grid_min",
                "max": f"phys_{source}_{level}_grid_max",
            }
        )
        stats[f"phys_{source}_{level}_grid_range"] = (
            stats[f"phys_{source}_{level}_grid_max"] - stats[f"phys_{source}_{level}_grid_min"]
        )
        if include_advanced:
            q = grouped.quantile([0.1, 0.9]).unstack().reset_index()
            q = q.rename(columns={0.1: f"phys_{source}_{level}_grid_p10", 0.9: f"phys_{source}_{level}_grid_p90"})
            q[f"phys_{source}_{level}_grid_p90_minus_p10"] = (
                q[f"phys_{source}_{level}_grid_p90"] - q[f"phys_{source}_{level}_grid_p10"]
            )
            stats = stats.merge(q, on=TIME_KEY_COLS, how="left")
        frames.append(stats)

        near_cols = [speed_col, along_col, cross_col, sw_col]
        nearest = work.loc[work["grid_id"] == nearest_id, TIME_KEY_COLS + near_cols].copy()
        nearest = nearest.rename(
            columns={
                speed_col: f"phys_{source}_{level}_near_speed",
                along_col: f"phys_{source}_{level}_near_axis_along",
                cross_col: f"phys_{source}_{level}_near_axis_cross",
                sw_col: f"phys_{source}_{level}_near_southwest",
            }
        )
        frames.append(nearest)
        if include_advanced:
            near_delta = stats[[*TIME_KEY_COLS, f"phys_{source}_{level}_grid_mean"]].merge(
                nearest[[*TIME_KEY_COLS, f"phys_{source}_{level}_near_speed"]],
                on=TIME_KEY_COLS,
                how="left",
            )
            near_delta[f"phys_{source}_{level}_near_minus_grid_mean"] = (
                near_delta[f"phys_{source}_{level}_near_speed"] - near_delta[f"phys_{source}_{level}_grid_mean"]
            )
            frames.append(near_delta[[*TIME_KEY_COLS, f"phys_{source}_{level}_near_minus_grid_mean"]])

        idw = _weighted_time_mean(work, speed_col, "_idw")
        idw = idw.rename(columns={speed_col: f"phys_{source}_{level}_idw_speed"})
        frames.append(idw)
        if include_advanced:
            ew = _time_slope(work, speed_col, "_dx", f"phys_{source}_{level}_east_west_gradient")
            ns = _time_slope(work, speed_col, "_dy", f"phys_{source}_{level}_north_south_gradient")
            frames.extend([ew, ns])

            safe_speed = work[speed_col].replace(0, np.nan)
            ux = work[u_col] / safe_speed
            uy = work[v_col] / safe_speed
            along_upwind = -(work["_dx"] * ux + work["_dy"] * uy)
            cross_upwind = np.abs(-work["_dx"] * uy + work["_dy"] * ux)
            gate_upwind = _sigmoid(along_upwind / 2.0)
            gate_downwind = _sigmoid(-along_upwind / 2.0)
            cross_gate = np.exp(-((cross_upwind / 2.5) ** 2))
            work[f"_{level}_upwind_w"] = cross_gate * gate_upwind * work["_idw"]
            work[f"_{level}_downwind_w"] = cross_gate * gate_downwind * work["_idw"]

            upwind = _weighted_time_mean(work, speed_col, f"_{level}_upwind_w")
            downwind = _weighted_time_mean(work, speed_col, f"_{level}_downwind_w")
            upwind = upwind.rename(columns={speed_col: f"phys_{source}_{level}_upwind_speed"})
            downwind = downwind.rename(columns={speed_col: f"phys_{source}_{level}_downwind_speed"})
            updown = upwind.merge(downwind, on=TIME_KEY_COLS, how="left")
            updown[f"phys_{source}_{level}_upwind_minus_downwind"] = (
                updown[f"phys_{source}_{level}_upwind_speed"] - updown[f"phys_{source}_{level}_downwind_speed"]
            )
            frames.append(updown)

    if include_advanced:
        for high_idx, (high_u, high_v, high_level) in enumerate(levels):
            if high_u not in work.columns or high_v not in work.columns:
                continue
            for low_u, low_v, low_level in levels[:high_idx]:
                if low_u not in work.columns or low_v not in work.columns:
                    continue
                pair = work.loc[work["grid_id"] == nearest_id, TIME_KEY_COLS + [high_u, high_v, low_u, low_v]].copy()
                diff = _angular_diff(pair[high_u], pair[high_v], pair[low_u], pair[low_v])
                pair[f"phys_{source}_veer_{high_level}_{low_level}_sin"] = np.sin(diff)
                pair[f"phys_{source}_veer_{high_level}_{low_level}_cos"] = np.cos(diff)
                pair[f"phys_{source}_veer_{high_level}_{low_level}_abs"] = np.abs(diff)
                frames.append(
                    pair[
                        [
                            *TIME_KEY_COLS,
                            f"phys_{source}_veer_{high_level}_{low_level}_sin",
                            f"phys_{source}_veer_{high_level}_{low_level}_cos",
                            f"phys_{source}_veer_{high_level}_{low_level}_abs",
                        ]
                    ]
                )

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on="forecast_kst_dtm", how="outer", suffixes=("", "_dup"))
        out = out.drop(columns=[c for c in out.columns if c.endswith("_dup")])
    return out


def _add_regime_features(df, speed_cols, include_advanced=False):
    out = df.copy()
    for col in speed_cols:
        if col not in out.columns:
            continue
        ws = out[col].astype(float).clip(lower=0.0)
        out[f"{col}_sq"] = ws**2
        out[f"{col}_cube"] = ws**3
        out[f"{col}_regime_cutin"] = (ws < 3.0).astype(float)
        out[f"{col}_regime_ramp_low"] = ((ws >= 3.0) & (ws < 7.0)).astype(float)
        out[f"{col}_regime_ramp_high"] = ((ws >= 7.0) & (ws < 12.0)).astype(float)
        out[f"{col}_regime_rated"] = ((ws >= 12.0) & (ws < 25.0)).astype(float)
        out[f"{col}_regime_cutout"] = (ws >= 25.0).astype(float)
        if include_advanced:
            for center, scale in [(3.0, 0.7), (7.0, 1.0), (12.0, 1.2), (25.0, 2.0)]:
                out[f"{col}_sigmoid_gt_{center:g}"] = _sigmoid((ws - center) / scale)
    return out


def _add_cross_features(df, include_advanced=False):
    out = df.copy()
    eps = 0.1
    pairs = [
        ("gfs_ws100_speed", "gfs_ws10_speed", "phys_shear_gfs_100_10"),
        ("gfs_ws850_speed", "gfs_ws100_speed", "phys_shear_gfs_850_100"),
        ("gfs_ws850_speed", "gfs_ws10_speed", "phys_shear_gfs_850_10"),
        ("ldaps_ws50_max_speed", "ldaps_ws10_speed", "phys_shear_ldaps_50max_10"),
        ("ldaps_ws50_min_speed", "ldaps_ws10_speed", "phys_shear_ldaps_50min_10"),
    ]
    for high_col, low_col, out_col in pairs:
        if high_col in out.columns and low_col in out.columns:
            out[out_col] = np.log((out[high_col].clip(lower=0.0) + eps) / (out[low_col].clip(lower=0.0) + eps))

    if "phys_gfs_gust" in out.columns and "gfs_ws10_speed" in out.columns:
        out["phys_gfs_gust_factor"] = out["phys_gfs_gust"] / (out["gfs_ws10_speed"].clip(lower=0.0) + eps)
        out["phys_gfs_gust_minus_ws10"] = out["phys_gfs_gust"] - out["gfs_ws10_speed"]

    for rho_col, speed_col in [
        ("phys_gfs_air_density", "gfs_ws100_speed"),
        ("phys_gfs_air_density", "gfs_ws850_speed"),
        ("phys_ldaps_air_density", "ldaps_ws50_max_speed"),
        ("phys_ldaps_air_density", "ldaps_ws10_speed"),
    ]:
        if rho_col in out.columns and speed_col in out.columns:
            out[f"{rho_col}_x_{speed_col}_cube"] = out[rho_col] * out[speed_col].clip(lower=0.0) ** 3

    trend_cols = ["gfs_ws100_speed", "gfs_ws850_speed", "ldaps_ws10_speed", "ldaps_ws50_max_speed"]
    if include_advanced:
        trend_cols += [
            "phys_gfs_gust",
            "phys_gfs_surface_pressure",
            "phys_ldaps_surface_pressure",
            "phys_gfs_pbl_vrate",
            "phys_gfs_cloud_sum",
            "phys_ldaps_cloud_sum",
        ]
    out = out.sort_values("forecast_kst_dtm").reset_index(drop=True)
    for col in trend_cols:
        if col in out.columns:
            out[f"phys_{col}_diff1"] = out[col].diff(1)
            out[f"phys_{col}_diff3"] = out[col].diff(3)
    return out


def build_compact_physics_features(ldaps, gfs, group=None, include_advanced=False):
    target_lat, target_lon, axis_x, axis_y = _target_for_group(group)
    ldaps_spatial = _source_spatial_stats(ldaps, "ldaps", LDAPS_LEVELS, target_lat, target_lon, axis_x, axis_y, include_advanced)
    gfs_spatial = _source_spatial_stats(gfs, "gfs", GFS_LEVELS, target_lat, target_lon, axis_x, axis_y, include_advanced)
    thermo = _aggregate_thermo(ldaps, gfs, target_lat, target_lon)

    out = ldaps_spatial.merge(gfs_spatial, on="forecast_kst_dtm", how="inner", suffixes=("", "_dup"))
    out = out.drop(columns=[c for c in out.columns if c.endswith("_dup")])
    out = out.merge(thermo.drop(columns=["data_available_kst_dtm"], errors="ignore"), on="forecast_kst_dtm", how="left")
    out = out.drop(columns=["data_available_kst_dtm"], errors="ignore")
    out = out.sort_values("forecast_kst_dtm").ffill().fillna(0).reset_index(drop=True)
    return out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)


def add_compact_physics_features(weather, ldaps, gfs, group=None, include_advanced=False):
    out = weather.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    compact = build_compact_physics_features(ldaps, gfs, group=group, include_advanced=include_advanced)
    out = out.merge(compact, on="forecast_kst_dtm", how="left").sort_values("forecast_kst_dtm")
    speed_cols = [
        "gfs_ws10_speed",
        "gfs_ws100_speed",
        "gfs_ws850_speed",
        "ldaps_ws10_speed",
        "ldaps_ws50_max_speed",
        "phys_gfs_ws850_near_speed",
        "phys_gfs_ws100_near_speed",
        "phys_ldaps_ws10_grid_mean",
    ]
    out = _add_regime_features(out, speed_cols, include_advanced=include_advanced)
    out = _add_cross_features(out, include_advanced=include_advanced)
    return out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0).reset_index(drop=True)

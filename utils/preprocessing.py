import numpy as np
import pandas as pd


def add_wind_speed(df, u_col, v_col, prefix):
    # direction was dropped: permutation importance on a 2024 time holdout showed every
    # *_direction feature near zero or negative across both sources -- this site's
    # generation is driven by wind speed, not wind direction
    df[f"{prefix}_speed"] = np.sqrt(df[u_col] ** 2 + df[v_col] ** 2)
    return df


# (u_col, v_col, feature_prefix) per height/level available in each forecast source
LDAPS_WIND_LEVELS = [
    ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "ws10"),
    ("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax", "ws50_max"),
    ("heightAboveGround_50_50MUmin", "heightAboveGround_50_50MVmin", "ws50_min"),
    ("heightAboveGround_5_XBLWS", "heightAboveGround_5_YBLWS", "ws5_bl"),
]

GFS_WIND_LEVELS = [
    ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "ws10"),
    ("heightAboveGround_80_u", "heightAboveGround_80_v", "ws80"),
    ("heightAboveGround_100_100u", "heightAboveGround_100_100v", "ws100"),
    ("planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v", "ws_pbl"),
    ("isobaricInhPa_850_u", "isobaricInhPa_850_v", "ws850"),
    ("isobaricInhPa_700_u", "isobaricInhPa_700_v", "ws700"),
    ("isobaricInhPa_500_u", "isobaricInhPa_500_v", "ws500"),
]

# raw wind columns to keep -- everything else (temperature, humidity, pressure,
# radiation, cloud, precipitation, ...) is dropped to cut feature count/overfitting risk
LDAPS_WIND_RAW_COLS = [c for u, v, _ in LDAPS_WIND_LEVELS for c in (u, v)]
GFS_WIND_RAW_COLS = [c for u, v, _ in GFS_WIND_LEVELS for c in (u, v)] + [
    "planetaryBoundaryLayer_0_VRATE",  # PBL vertical wind speed
    "surface_0_gust",  # surface wind gust
]


def add_ldaps_derived_features(df):
    df = df.copy()
    for u_col, v_col, prefix in LDAPS_WIND_LEVELS:
        df = add_wind_speed(df, u_col, v_col, prefix)
    return df


def add_gfs_derived_features(df):
    df = df.copy()
    for u_col, v_col, prefix in GFS_WIND_LEVELS:
        df = add_wind_speed(df, u_col, v_col, prefix)
    return df


# Farm centroid derived from info.xlsx turbine coordinates. The 3 KPX groups sit
# within ~2km of each other, far smaller than the GFS grid spacing (~28km), so all
# groups resolve to the same nearest GFS grid -- one shared centroid is enough.
FARM_CENTROID = (37.2819, 128.96237)

TIME_KEY_COLS = ["forecast_kst_dtm", "data_available_kst_dtm"]


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlmb = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _ffill_by_time(df):
    """Some forecast hours have a variable missing across all grids at once (rare source gaps).
    Forward-fill only (never backfill) so later timestamps never leak into earlier ones."""
    return df.sort_values("forecast_kst_dtm").ffill().reset_index(drop=True)


def aggregate_ldaps_grids(df):
    """LDAPS: all 16 grids sit within ~1km of the farm, so average them into one row per forecast time."""
    agg = df.groupby(TIME_KEY_COLS, as_index=False)[LDAPS_WIND_RAW_COLS].mean()
    return _ffill_by_time(agg)


def nearest_gfs_grid_id(df, target_lat, target_lon):
    grids = df[["grid_id", "latitude", "longitude"]].drop_duplicates()
    dist = haversine_km(target_lat, target_lon, grids["latitude"], grids["longitude"])
    return grids.loc[dist.idxmin(), "grid_id"]


def aggregate_gfs_grids(df, target_lat, target_lon):
    """GFS grids are ~28km apart; the farm sits ~4-6km from one grid and 18km+ from the rest,
    so pick the single nearest grid instead of averaging (averaging would dilute the local signal)."""
    grid_id = nearest_gfs_grid_id(df, target_lat, target_lon)
    selected = df.loc[df["grid_id"] == grid_id, TIME_KEY_COLS + GFS_WIND_RAW_COLS].reset_index(drop=True)
    return _ffill_by_time(selected)


# turbine hub height is 117m; GFS 100m wind is the closest available level and is
# used as the hub-height proxy for the SCADA power-curve feature (utils/power_curve.py)
HUB_HEIGHT_PROXY_COL = "gfs_ws100_speed"


def build_weather_features(ldaps_df, gfs_df, target_lat=FARM_CENTROID[0], target_lon=FARM_CENTROID[1]):
    """Aggregate LDAPS + GFS grids to a single per-forecast-hour feature table, shared across all KPX groups."""
    ldaps_agg = aggregate_ldaps_grids(ldaps_df)
    ldaps_agg = add_ldaps_derived_features(ldaps_agg)
    ldaps_agg = ldaps_agg.rename(columns={c: f"ldaps_{c}" for c in ldaps_agg.columns if c not in TIME_KEY_COLS})

    gfs_agg = aggregate_gfs_grids(gfs_df, target_lat, target_lon)
    gfs_agg = add_gfs_derived_features(gfs_agg)
    gfs_agg = gfs_agg.rename(columns={c: f"gfs_{c}" for c in gfs_agg.columns if c not in TIME_KEY_COLS})

    merged = ldaps_agg.merge(
        gfs_agg, on="forecast_kst_dtm", how="inner", suffixes=("", "_gfs_dup")
    )
    merged = merged.drop(columns=[c for c in merged.columns if c.endswith("_gfs_dup")])
    merged = add_cyclical_time_features(merged, "forecast_kst_dtm")
    return merged


def add_cyclical_time_features(df, dt_col):
    # day-of-week/month-of-year sin-cos pairs and hdw_cos were dropped: permutation
    # importance on a 2024 time holdout showed them near zero or negative (noise),
    # while day-of-year and hour-of-day (and their sin-based combination) held up
    df = df.copy()
    dt = pd.to_datetime(df[dt_col])

    doy = dt.dt.dayofyear
    dow = dt.dt.dayofweek
    hod = dt.dt.hour

    df["sin_doy"] = np.sin(2 * np.pi * doy / 365)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365)
    df["sin_hod"] = np.sin(2 * np.pi * hod / 24)
    df["cos_hod"] = np.cos(2 * np.pi * hod / 24)

    sin_dow = np.sin(2 * np.pi * dow / 7)

    # day-of-year + hour-of-day / day-of-week + hour-of-day combined signals
    df["hour_day_year_sin"] = df["sin_doy"] + df["sin_hod"]
    df["hour_day_year_cos"] = df["cos_doy"] + df["cos_hod"]
    df["hdw_sin"] = sin_dow + df["sin_hod"]

    return df


def build_group_dataset(weather_df, labels_df, group):
    """Merge the shared weather feature table with one KPX group's labels,
    keeping only rows where that group's label is present (handles kpx_group_3's 2022 gap)."""
    merged = weather_df.merge(
        labels_df[["kst_dtm", group]], left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner"
    )
    merged = merged.dropna(subset=[group])
    feature_cols = [c for c in weather_df.columns if c not in TIME_KEY_COLS]
    X = merged[feature_cols].reset_index(drop=True)
    y = merged[group].reset_index(drop=True)
    return X, y

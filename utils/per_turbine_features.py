from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.preprocessing import TIME_KEY_COLS, haversine_km
from utils.site_metadata import _latlon_to_xy_km, load_turbine_metadata
from utils.tree_feature_profiles import (
    FEATURE_PROFILE_GROUP_FAMILY_QUOTA65_V1,
    GROUP_FAMILY_QUOTA65_V1_FEATURES,
    build_tree_features,
)


FEATURE_CACHE_VERSION = "per_turbine_features_v2_issue_safe"

LDAPS_LEVELS = {
    "ws10": ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
    "ws50max": ("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax"),
    "ws50min": ("heightAboveGround_50_50MUmin", "heightAboveGround_50_50MVmin"),
    "ws5bl": ("heightAboveGround_5_XBLWS", "heightAboveGround_5_YBLWS"),
}
GFS_LEVELS = {
    "ws10": ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
    "ws80": ("heightAboveGround_80_u", "heightAboveGround_80_v"),
    "ws100": ("heightAboveGround_100_100u", "heightAboveGround_100_100v"),
    "ws850": ("isobaricInhPa_850_u", "isobaricInhPa_850_v"),
    "ws700": ("isobaricInhPa_700_u", "isobaricInhPa_700_v"),
    "ws500": ("isobaricInhPa_500_u", "isobaricInhPa_500_v"),
}

LOCAL_TREE_FEATURES = [
    "local_ldaps_ws10",
    "local_ldaps_ws50max",
    "local_ldaps_ws50min",
    "local_ldaps_ws5bl",
    "local_gfs_ws10",
    "local_gfs_ws80",
    "local_gfs_ws100",
    "local_gfs_ws850",
    "local_gfs_ws700",
    "local_gfs_ws500",
    "local_ldaps_50max_u",
    "local_ldaps_50max_v",
    "local_gfs_100_u",
    "local_gfs_100_v",
    "local_shear_ldaps_50max_10",
    "local_shear_gfs_100_10",
    "wake_exposure",
    "wake_upstream_count",
]

GROUP_WAKE_FEATURES = [
    "wake_exposure_mean",
    "wake_exposure_max",
    "wake_exposure_std",
    "wake_upstream_count_mean",
    "wake_upstream_count_max",
    "wake_upstream_count_std",
]


def _metadata_for_group(group: str) -> pd.DataFrame:
    meta = load_turbine_metadata()
    meta = meta.loc[meta["group"] == group, ["turbine_id", "latitude", "longitude"]].copy()
    expected = GROUP_TURBINE_PREFIXES[group]
    meta = meta.set_index("turbine_id").reindex(expected).reset_index()
    if meta[["latitude", "longitude"]].isna().any().any():
        raise ValueError(f"Missing turbine coordinates for {group}")
    ref_lat = float(meta["latitude"].mean())
    ref_lon = float(meta["longitude"].mean())
    meta["x_km"], meta["y_km"] = _latlon_to_xy_km(
        meta["latitude"], meta["longitude"], ref_lat, ref_lon
    )
    meta["turbine_index"] = np.arange(len(meta), dtype=int)
    return meta


def _prepare_time_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TIME_KEY_COLS:
        out[col] = pd.to_datetime(out[col])
    return out


def _idw_aggregate(
    raw: pd.DataFrame,
    latitude: float,
    longitude: float,
    columns: list[str],
) -> pd.DataFrame:
    grids = raw[["grid_id", "latitude", "longitude"]].drop_duplicates("grid_id").copy()
    grids["weight"] = 1.0 / np.maximum(
        haversine_km(latitude, longitude, grids["latitude"], grids["longitude"]) ** 2,
        0.01,
    )
    weight_map = grids.set_index("grid_id")["weight"]
    work = raw[TIME_KEY_COLS + ["grid_id", *columns]].copy()
    work["_weight"] = work["grid_id"].map(weight_map).astype(float)
    work[columns] = work[columns].multiply(work["_weight"], axis=0)
    grouped = work.groupby(TIME_KEY_COLS, as_index=False)[["_weight", *columns]].sum()
    grouped[columns] = grouped[columns].div(grouped["_weight"], axis=0)
    return grouped.drop(columns="_weight")


def _nearest_aggregate(
    raw: pd.DataFrame,
    latitude: float,
    longitude: float,
    columns: list[str],
) -> pd.DataFrame:
    grids = raw[["grid_id", "latitude", "longitude"]].drop_duplicates("grid_id").copy()
    distances = haversine_km(latitude, longitude, grids["latitude"], grids["longitude"])
    grid_id = grids.iloc[int(np.asarray(distances).argmin())]["grid_id"]
    return raw.loc[raw["grid_id"] == grid_id, TIME_KEY_COLS + columns].copy()


def _add_speed_features(
    table: pd.DataFrame,
    levels: dict[str, tuple[str, str]],
    source: str,
) -> pd.DataFrame:
    out = table.copy()
    for name, (u_col, v_col) in levels.items():
        out[f"local_{source}_{name}"] = np.hypot(out[u_col], out[v_col])
    return out


def _local_weather(
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
    latitude: float,
    longitude: float,
) -> pd.DataFrame:
    ldaps_cols = sorted({col for pair in LDAPS_LEVELS.values() for col in pair})
    gfs_cols = sorted({col for pair in GFS_LEVELS.values() for col in pair})
    local_ldaps = _idw_aggregate(ldaps, latitude, longitude, ldaps_cols)
    local_gfs = _nearest_aggregate(gfs, latitude, longitude, gfs_cols)
    local_ldaps = _add_speed_features(local_ldaps, LDAPS_LEVELS, "ldaps")
    local_gfs = _add_speed_features(local_gfs, GFS_LEVELS, "gfs")

    local_ldaps = local_ldaps.rename(
        columns={
            "heightAboveGround_50_50MUmax": "local_ldaps_50max_u",
            "heightAboveGround_50_50MVmax": "local_ldaps_50max_v",
        }
    )
    local_gfs = local_gfs.rename(
        columns={
            "heightAboveGround_100_100u": "local_gfs_100_u",
            "heightAboveGround_100_100v": "local_gfs_100_v",
        }
    )
    out = _prepare_time_keys(local_ldaps).merge(
        _prepare_time_keys(local_gfs), on=TIME_KEY_COLS, how="inner", suffixes=("_ldaps_raw", "_gfs_raw")
    )
    out["local_shear_ldaps_50max_10"] = out["local_ldaps_ws50max"] - out["local_ldaps_ws10"]
    out["local_shear_gfs_100_10"] = out["local_gfs_ws100"] - out["local_gfs_ws10"]
    return out[TIME_KEY_COLS + [c for c in LOCAL_TREE_FEATURES if not c.startswith("wake_")]]


def _add_wake_features(table: pd.DataFrame, turbine: pd.Series, meta: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    u = out["local_gfs_100_u"].to_numpy(float)
    v = out["local_gfs_100_v"].to_numpy(float)
    speed = np.maximum(np.hypot(u, v), 1e-6)
    unit_x = u / speed
    unit_y = v / speed
    exposure = np.zeros(len(out), dtype=float)
    count = np.zeros(len(out), dtype=float)

    for other in meta.itertuples(index=False):
        if other.turbine_id == turbine["turbine_id"]:
            continue
        dx = float(other.x_km - turbine["x_km"])
        dy = float(other.y_km - turbine["y_km"])
        along = -(dx * unit_x + dy * unit_y)
        cross = np.abs(-dx * unit_y + dy * unit_x)
        upstream = along > 0
        wake_width = 0.15 + 0.10 * np.maximum(along, 0)
        exposure += np.where(
            upstream,
            np.exp(-np.maximum(along, 0) / 2.0) * np.exp(-np.square(cross / wake_width)),
            0.0,
        )
        count += upstream & (cross <= 0.20 + 0.15 * np.maximum(along, 0))

    out["wake_exposure"] = exposure
    out["wake_upstream_count"] = count
    return out


def build_group_per_turbine_features(
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
    group: str,
) -> pd.DataFrame:
    base = build_tree_features(
        ldaps,
        gfs,
        group,
        feature_profile=FEATURE_PROFILE_GROUP_FAMILY_QUOTA65_V1,
    )
    base = _prepare_time_keys(base)
    quota_features = GROUP_FAMILY_QUOTA65_V1_FEATURES[group]
    missing = [col for col in quota_features if col not in base.columns]
    if missing:
        raise ValueError(f"Missing quota65 features for {group}: {missing}")
    base = base[TIME_KEY_COLS + quota_features]

    meta = _metadata_for_group(group)
    parts = []
    for _, turbine in meta.iterrows():
        local = _local_weather(
            ldaps,
            gfs,
            float(turbine["latitude"]),
            float(turbine["longitude"]),
        )
        local = _add_wake_features(local, turbine, meta)
        table = base.merge(local[TIME_KEY_COLS + LOCAL_TREE_FEATURES], on=TIME_KEY_COLS, how="inner")
        table["group"] = group
        table["turbine_id"] = turbine["turbine_id"]
        table["turbine_index"] = int(turbine["turbine_index"])
        table["turbine_x_km"] = float(turbine["x_km"])
        table["turbine_y_km"] = float(turbine["y_km"])
        parts.append(table)
    return pd.concat(parts, ignore_index=True)


def build_group_wake_features(
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
    group: str,
) -> pd.DataFrame:
    """Summarize turbine-level wake exposure into one row per group forecast."""
    meta = _metadata_for_group(group)
    parts = []
    for _, turbine in meta.iterrows():
        local = _local_weather(
            ldaps,
            gfs,
            float(turbine["latitude"]),
            float(turbine["longitude"]),
        )
        local = _add_wake_features(local, turbine, meta)
        parts.append(
            local[TIME_KEY_COLS + ["wake_exposure", "wake_upstream_count"]]
        )

    turbine_wake = pd.concat(parts, ignore_index=True)
    grouped = turbine_wake.groupby(TIME_KEY_COLS, sort=True)
    output = grouped[["wake_exposure", "wake_upstream_count"]].agg(
        ["mean", "max", "std"]
    )
    output.columns = [f"{name}_{stat}" for name, stat in output.columns]
    return output.reset_index().fillna(0.0)


def get_or_build_group_feature_cache(
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
    group: str,
    cache_root: Path = Path("cache"),
    rebuild: bool = False,
    cache_tag: str | None = None,
) -> pd.DataFrame:
    cache_dir = cache_root / FEATURE_CACHE_VERSION
    suffix = f"_{cache_tag}" if cache_tag else ""
    cache_path = cache_dir / f"{group}{suffix}.pkl"
    if cache_path.exists() and not rebuild:
        return pd.read_pickle(cache_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    features = build_group_per_turbine_features(ldaps, gfs, group)
    features.to_pickle(cache_path)
    return features


def tree_feature_columns(group: str) -> list[str]:
    return [*GROUP_FAMILY_QUOTA65_V1_FEATURES[group], *LOCAL_TREE_FEATURES]

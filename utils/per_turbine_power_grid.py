from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.preprocessing import TIME_KEY_COLS
from utils.tree_feature_profiles import GROUP_FAMILY_QUOTA65_V1_FEATURES


POWER_GRID_CACHE_VERSION = "per_turbine_power_grid_pair_v1"
POWER_GRID_TEACHER_TAG = "power_grid_pair12_v1"
POWER_GRID_FEATURES = [
    "pgrid_local_ws",
    "pgrid_local_wd_sin",
    "pgrid_local_wd_cos",
    "pgrid_local_power",
    "pgrid_synoptic_ws",
    "pgrid_synoptic_wd_sin",
    "pgrid_synoptic_wd_cos",
    "pgrid_synoptic_power",
    "pgrid_power_mean",
    "pgrid_power_max",
    "pgrid_power_q65",
    "pgrid_power_spread",
]
WAKE_FEATURES = ["wake_exposure", "wake_upstream_count"]


def power_grid_input_columns(group: str) -> list[str]:
    return [*GROUP_FAMILY_QUOTA65_V1_FEATURES[group], *WAKE_FEATURES, *POWER_GRID_FEATURES]


def load_power_grid_fold_features(
    base_features: pd.DataFrame,
    cache_root: Path,
    group: str,
    pred_year: int,
) -> pd.DataFrame:
    path = cache_root / POWER_GRID_CACHE_VERSION / f"{group}_pred{pred_year}_features.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Power-grid pair cache is missing: {path}")
    pair = pd.read_pickle(path)
    keys = [*TIME_KEY_COLS, "turbine_id"]
    before = len(base_features)
    out = base_features.merge(pair[keys + POWER_GRID_FEATURES], on=keys, how="left")
    if len(out) != before:
        raise ValueError(f"Power-grid merge changed row count: {before} -> {len(out)}")
    coverage = float(out["pgrid_local_power"].notna().mean())
    if coverage < 0.95:
        raise ValueError(f"Power-grid feature coverage too low: {group} {pred_year} {coverage}")
    return out

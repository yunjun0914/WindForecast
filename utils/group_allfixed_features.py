from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.group_local_panel import GroupLocalPanel, build_group_local_panel
from utils.group_quota_v2 import (
    GROUP_FAMILY_QUOTA64_V2_FEATURES,
    get_or_build_group_quota_v2,
)
from utils.per_turbine_features import get_or_build_group_feature_cache
from utils.per_turbine_fixed_grid import (
    FIXED_TURBINE_GRID_FEATURES,
    get_or_build_fixed_turbine_grid_features,
)
from utils.per_turbine_optimal_grid import WAKE_FEATURES
from utils.preprocessing import TIME_KEY_COLS


QUOTA_V1_FIXED_AUX_CONTROL = "quota_v1_fixed_aux_control"
QUOTA_V2_ALL_FIXED = "quota_v2_all_fixed"
ALLFIXED_VARIANTS = (QUOTA_V1_FIXED_AUX_CONTROL, QUOTA_V2_ALL_FIXED)
ALLFIXED_LOCAL_FEATURES = [*WAKE_FEATURES, *FIXED_TURBINE_GRID_FEATURES]


def get_or_build_group_allfixed_panels(
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
    group: str,
    *,
    cache_root: Path = Path("cache"),
    rebuild_base: bool = False,
    rebuild_fixed_grid: bool = False,
    rebuild_quota_v2: bool = False,
) -> tuple[dict[str, GroupLocalPanel], pd.DataFrame, pd.DataFrame]:
    """Build the identical v1/v2 group contracts used by TCN, TREE and PINN."""
    base = get_or_build_group_feature_cache(
        ldaps,
        gfs,
        group,
        cache_root=cache_root,
        rebuild=rebuild_base,
    )
    fixed_aux, aux_contract = get_or_build_fixed_turbine_grid_features(
        ldaps,
        group,
        cache_root=cache_root,
        rebuild=rebuild_fixed_grid,
    )
    keys = [*TIME_KEY_COLS, "turbine_id"]
    before = len(base)
    features = base.merge(
        fixed_aux[keys + FIXED_TURBINE_GRID_FEATURES],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    if len(features) != before:
        raise ValueError(
            f"All-fixed auxiliary merge changed rows: {before} -> {len(features)}"
        )
    coverage = float(features[FIXED_TURBINE_GRID_FEATURES].notna().mean().min())
    if coverage < 0.95:
        raise ValueError(f"Low all-fixed auxiliary coverage for {group}: {coverage}")

    v1 = build_group_local_panel(
        features,
        group,
        local_feature_cols=ALLFIXED_LOCAL_FEATURES,
    )
    group_quota, quota_contract = get_or_build_group_quota_v2(
        ldaps,
        gfs,
        group,
        cache_root=cache_root,
        rebuild=rebuild_quota_v2,
    )
    before = len(features)
    v2_features = features.merge(
        group_quota,
        on=TIME_KEY_COLS,
        how="left",
        validate="many_to_one",
    )
    if len(v2_features) != before:
        raise ValueError(
            f"All-fixed Quota v2 merge changed rows: {before} -> {len(v2_features)}"
        )
    v2_cols = GROUP_FAMILY_QUOTA64_V2_FEATURES[group]
    missing = [column for column in v2_cols if column not in v2_features.columns]
    if missing:
        raise ValueError(f"All-fixed Quota v2 missing for {group}: {missing}")
    v2 = build_group_local_panel(
        v2_features,
        group,
        common_feature_cols=v2_cols,
        local_feature_cols=ALLFIXED_LOCAL_FEATURES,
    )
    if len(v1.full_feature_cols) != len(v2.full_feature_cols):
        raise ValueError(f"All-fixed v1/v2 feature counts differ for {group}")
    return (
        {QUOTA_V1_FIXED_AUX_CONTROL: v1, QUOTA_V2_ALL_FIXED: v2},
        aux_contract,
        quota_contract,
    )

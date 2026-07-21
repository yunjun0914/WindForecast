from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.preprocessing import TIME_KEY_COLS
from utils.site_metadata import group_site_summary
from utils.tree_feature_profiles import (
    FEATURE_PROFILE_GROUP_FAMILY_QUOTA65_V1,
    GROUP_FAMILY_QUOTA65_V1_FEATURES,
    build_tree_features,
)


GROUP_QUOTA_V2_CACHE_VERSION = "group_quota64_v2_fixed_group_grids_v1"
GROUP_QUOTA_V2_PREFIX = "gqv2__"
GROUP_QUOTA_V2_CONTRACT_NAME = "fixed_eda_group_mapping_v1"
GROUP_FAMILY_QUOTA64_V2_FEATURES = {
    group: [f"{GROUP_QUOTA_V2_PREFIX}{name}" for name in names]
    for group, names in GROUP_FAMILY_QUOTA65_V1_FEATURES.items()
}

# Fixed group contracts reconstructed from wind_eda_dashboard.html. Counts are
# fixed spatial weights, not fold-fitted selections: one count represents one
# member turbine assigned to the grid in the dashboard mapping.
GROUP_QUOTA_V2_FIXED_GRID_COUNTS = {
    "ldaps": {
        "kpx_group_1": {2: 3, 12: 3},
        "kpx_group_2": {2: 2, 3: 1, 7: 2, 12: 1},
        "kpx_group_3": {3: 1, 12: 1, 13: 3},
    },
    "gfs": {
        "kpx_group_1": {2: 2, 4: 3, 7: 1},
        "kpx_group_2": {2: 1, 4: 5},
        "kpx_group_3": {2: 4, 4: 1},
    },
}


def fixed_group_grid_contract(group: str, source: str) -> pd.DataFrame:
    """Return the immutable grid/weight definition for one group and source."""
    if source not in GROUP_QUOTA_V2_FIXED_GRID_COUNTS:
        raise ValueError(f"Unsupported NWP source: {source}")
    if group not in GROUP_TURBINE_PREFIXES:
        raise ValueError(f"Unknown group: {group}")
    counts = GROUP_QUOTA_V2_FIXED_GRID_COUNTS[source][group]
    expected_count = len(GROUP_TURBINE_PREFIXES[group])
    if sum(counts.values()) != expected_count:
        raise ValueError(
            f"Fixed grid counts for {group}/{source} sum to {sum(counts.values())}, "
            f"expected {expected_count}"
        )
    return pd.DataFrame(
        [
            {
                "group": group,
                "source": source,
                "grid_id": int(grid_id),
                "group_weight_count": int(count),
                "group_weight": float(count / expected_count),
                "contract": GROUP_QUOTA_V2_CONTRACT_NAME,
            }
            for grid_id, count in sorted(counts.items())
        ]
    )


def build_fixed_source_panel(
    raw: pd.DataFrame,
    contract: pd.DataFrame,
    group: str,
    source: str,
) -> pd.DataFrame:
    """Build the same fixed weighted grid panel for every year and outer fold."""
    required = {
        "group",
        "source",
        "grid_id",
        "group_weight_count",
        "group_weight",
        "contract",
    }
    missing = sorted(required.difference(contract.columns))
    if missing:
        raise ValueError(f"Fixed grid contract missing columns: {missing}")
    rows = contract.loc[
        contract["group"].eq(group) & contract["source"].eq(source)
    ].sort_values("grid_id")
    if rows.empty:
        raise ValueError(f"Empty fixed grid contract for {group}/{source}")

    expected = fixed_group_grid_contract(group, source).reset_index(drop=True)
    actual = rows[list(expected.columns)].reset_index(drop=True)
    if not actual.equals(expected):
        raise ValueError(f"Modified fixed grid contract for {group}/{source}")

    parts = []
    raw_grid = pd.to_numeric(raw["grid_id"], errors="coerce")
    for row in rows.itertuples(index=False):
        grid_id = int(row.grid_id)
        source_grid = raw.loc[raw_grid.eq(grid_id)].copy()
        if source_grid.empty:
            raise ValueError(f"Missing {source} grid {grid_id} for {group}")
        for replica in range(int(row.group_weight_count)):
            part = source_grid.copy()
            part["grid_id"] = f"fixed:{source}:{grid_id}:w{replica + 1}"
            parts.append(part)
    selected = pd.concat(parts, ignore_index=True)

    excluded = {*TIME_KEY_COLS, "grid_id", "latitude", "longitude"}
    value_cols = [column for column in raw.columns if column not in excluded]
    anchor = selected.groupby(TIME_KEY_COLS, as_index=False)[value_cols].mean()
    site = group_site_summary()[group]
    anchor["grid_id"] = "group_anchor"
    anchor["latitude"] = float(site["latitude"])
    anchor["longitude"] = float(site["longitude"])
    anchor = anchor[
        [
            *TIME_KEY_COLS,
            "grid_id",
            "latitude",
            "longitude",
            *value_cols,
        ]
    ]
    output = pd.concat([selected, anchor], ignore_index=True)
    for column in TIME_KEY_COLS:
        output[column] = pd.to_datetime(output[column])
    return output.sort_values(["forecast_kst_dtm", "grid_id"]).reset_index(
        drop=True
    )


def build_group_quota_v2(
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
    group: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute Quota v1 formulas on one immutable group-level grid contract."""
    contracts = pd.concat(
        [fixed_group_grid_contract(group, source) for source in ("ldaps", "gfs")],
        ignore_index=True,
    )
    ldaps_panel = build_fixed_source_panel(ldaps, contracts, group, "ldaps")
    gfs_panel = build_fixed_source_panel(gfs, contracts, group, "gfs")
    quota = build_tree_features(
        ldaps_panel,
        gfs_panel,
        group,
        feature_profile=FEATURE_PROFILE_GROUP_FAMILY_QUOTA65_V1,
    )
    v1_names = GROUP_FAMILY_QUOTA65_V1_FEATURES[group]
    missing = [name for name in v1_names if name not in quota.columns]
    if missing:
        raise ValueError(f"Group quota v2 missing features for {group}: {missing}")
    rename = {name: f"{GROUP_QUOTA_V2_PREFIX}{name}" for name in v1_names}
    output = quota[[*TIME_KEY_COLS, *v1_names]].rename(columns=rename)
    return output, contracts


def get_or_build_group_quota_v2(
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
    group: str,
    *,
    cache_root: Path = Path("cache"),
    rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = Path(cache_root) / GROUP_QUOTA_V2_CACHE_VERSION
    feature_path = cache_dir / f"{group}_features.pkl"
    contract_path = cache_dir / f"{group}_grid_contract.csv"
    if feature_path.exists() and contract_path.exists() and not rebuild:
        cached_contract = pd.read_csv(contract_path)
        expected_contract = pd.concat(
            [
                fixed_group_grid_contract(group, source)
                for source in ("ldaps", "gfs")
            ],
            ignore_index=True,
        )
        if cached_contract[list(expected_contract.columns)].equals(expected_contract):
            return pd.read_pickle(feature_path), cached_contract

    features, contract = build_group_quota_v2(ldaps, gfs, group)
    cache_dir.mkdir(parents=True, exist_ok=True)
    features.to_pickle(feature_path)
    contract.to_csv(contract_path, index=False, encoding="utf-8-sig")
    return features, contract

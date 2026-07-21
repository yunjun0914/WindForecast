from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.preprocessing import TIME_KEY_COLS


FIXED_TURBINE_GRID_CACHE_VERSION = "per_turbine_fixed_ldaps_grid_v1"
FIXED_TURBINE_GRID_CONTRACT_NAME = "wind_eda_fixed_ldaps_v1"
FIXED_TURBINE_GRID_FEATURES = [
    "fixedgrid_ws_raw",
    "fixedgrid_ws_cube",
    "fixedgrid_wd_sin",
    "fixedgrid_wd_cos",
]

# Immutable turbine-to-grid mapping from wind_eda_dashboard.html. This table is
# applied unchanged to every train/validation year and to final inference.
FIXED_TURBINE_LDAPS_CONTRACT = {
    "kpx_group_1": {
        "vestas_wtg01": (2, "ws50_midpoint"),
        "vestas_wtg02": (2, "ws50_midpoint"),
        "vestas_wtg03": (2, "ws50_midpoint"),
        "vestas_wtg04": (12, "ws10"),
        "vestas_wtg05": (12, "ws10"),
        "vestas_wtg06": (12, "ws10"),
    },
    "kpx_group_2": {
        "vestas_wtg07": (12, "ws10"),
        "vestas_wtg08": (2, "ws50_midpoint"),
        "vestas_wtg09": (2, "ws50_midpoint"),
        "vestas_wtg10": (7, "ws50_midpoint"),
        "vestas_wtg11": (7, "ws50_midpoint"),
        "vestas_wtg12": (3, "ws50_midpoint"),
    },
    "kpx_group_3": {
        "unison_wtg01": (12, "ws10"),
        "unison_wtg02": (13, "ws10"),
        "unison_wtg03": (13, "ws10"),
        "unison_wtg04": (13, "ws10"),
        "unison_wtg05": (3, "ws50_midpoint"),
    },
}


def fixed_turbine_grid_contract(group: str) -> pd.DataFrame:
    if group not in GROUP_TURBINE_PREFIXES:
        raise ValueError(f"Unknown group: {group}")
    mapping = FIXED_TURBINE_LDAPS_CONTRACT[group]
    expected = list(GROUP_TURBINE_PREFIXES[group])
    if list(mapping) != expected:
        raise ValueError(f"Fixed turbine-grid order differs for {group}")
    return pd.DataFrame(
        [
            {
                "group": group,
                "turbine_id": turbine,
                "source": "ldaps",
                "grid_id": int(mapping[turbine][0]),
                "level": mapping[turbine][1],
                "contract": FIXED_TURBINE_GRID_CONTRACT_NAME,
            }
            for turbine in expected
        ]
    )


def _fixed_uv(grid: pd.DataFrame, level: str) -> tuple[pd.Series, pd.Series]:
    if level == "ws10":
        u = pd.to_numeric(grid["heightAboveGround_10_10u"], errors="coerce")
        v = pd.to_numeric(grid["heightAboveGround_10_10v"], errors="coerce")
        return u, v
    if level == "ws50_midpoint":
        # Midpoint is the vector halfway between the LDAPS 50 m max/min wind
        # components, followed by vector magnitude and direction calculation.
        u_max = pd.to_numeric(
            grid["heightAboveGround_50_50MUmax"], errors="coerce"
        )
        u_min = pd.to_numeric(
            grid["heightAboveGround_50_50MUmin"], errors="coerce"
        )
        v_max = pd.to_numeric(
            grid["heightAboveGround_50_50MVmax"], errors="coerce"
        )
        v_min = pd.to_numeric(
            grid["heightAboveGround_50_50MVmin"], errors="coerce"
        )
        return 0.5 * (u_max + u_min), 0.5 * (v_max + v_min)
    raise ValueError(f"Unsupported fixed LDAPS level: {level}")


def build_fixed_turbine_grid_features(
    ldaps: pd.DataFrame,
    group: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build target-free turbine-local wind features from one fixed contract."""
    contract = fixed_turbine_grid_contract(group)
    raw = ldaps.copy()
    for column in TIME_KEY_COLS:
        raw[column] = pd.to_datetime(raw[column])
    raw_grid = pd.to_numeric(raw["grid_id"], errors="coerce")

    parts = []
    for row in contract.itertuples(index=False):
        grid = raw.loc[raw_grid.eq(int(row.grid_id))].copy()
        if grid.empty:
            raise ValueError(
                f"Missing fixed LDAPS grid {row.grid_id} for {row.turbine_id}"
            )
        if grid.duplicated(TIME_KEY_COLS).any():
            raise ValueError(
                f"Duplicate fixed LDAPS rows for {row.turbine_id}/grid{row.grid_id}"
            )
        u, v = _fixed_uv(grid, row.level)
        speed = np.hypot(u.to_numpy(float), v.to_numpy(float))
        safe_speed = np.maximum(speed, 1e-6)
        part = grid[TIME_KEY_COLS].copy()
        part["turbine_id"] = row.turbine_id
        part["fixedgrid_ws_raw"] = speed
        part["fixedgrid_ws_cube"] = speed**3
        part["fixedgrid_wd_sin"] = v.to_numpy(float) / safe_speed
        part["fixedgrid_wd_cos"] = u.to_numpy(float) / safe_speed
        parts.append(part)

    output = pd.concat(parts, ignore_index=True)
    expected_rows = len(contract) * raw["forecast_kst_dtm"].nunique()
    if len(output) != expected_rows:
        raise ValueError(
            f"Fixed turbine-grid row count mismatch for {group}: "
            f"{len(output)} != {expected_rows}"
        )
    return output, contract


def get_or_build_fixed_turbine_grid_features(
    ldaps: pd.DataFrame,
    group: str,
    *,
    cache_root: Path = Path("cache"),
    rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = Path(cache_root) / FIXED_TURBINE_GRID_CACHE_VERSION
    feature_path = cache_dir / f"{group}_features.pkl"
    contract_path = cache_dir / f"{group}_contract.csv"
    expected_contract = fixed_turbine_grid_contract(group)
    if feature_path.exists() and contract_path.exists() and not rebuild:
        cached_contract = pd.read_csv(contract_path)
        if cached_contract[list(expected_contract.columns)].equals(expected_contract):
            return pd.read_pickle(feature_path), cached_contract

    features, contract = build_fixed_turbine_grid_features(ldaps, group)
    cache_dir.mkdir(parents=True, exist_ok=True)
    features.to_pickle(feature_path)
    contract.to_csv(contract_path, index=False, encoding="utf-8-sig")
    return features, contract

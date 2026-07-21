from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.per_turbine_optimal_grid_builder import WindCandidateMatrix
from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.preprocessing import TIME_KEY_COLS
from utils.site_metadata import group_site_summary
from utils.tree_feature_profiles import (
    FEATURE_PROFILE_GROUP_FAMILY_QUOTA65_V1,
    GROUP_FAMILY_QUOTA65_V1_FEATURES,
    build_tree_features,
)


GROUP_QUOTA_V2_CACHE_VERSION = "group_quota64_v2_source_raw_l3_v1"
GROUP_QUOTA_V2_PREFIX = "gqv2__"
GROUP_FAMILY_QUOTA64_V2_FEATURES = {
    group: [f"{GROUP_QUOTA_V2_PREFIX}{name}" for name in names]
    for group, names in GROUP_FAMILY_QUOTA65_V1_FEATURES.items()
}


def _parse_candidate(candidate: str) -> tuple[str, int, str]:
    parts = str(candidate).split("|", maxsplit=2)
    if len(parts) != 3:
        raise ValueError(f"Invalid wind candidate: {candidate}")
    source, grid_id, level = parts
    return source, int(float(grid_id)), level


def select_group_source_grids(
    candidates: WindCandidateMatrix,
    targets: pd.DataFrame,
    group: str,
    train_years: list[int] | tuple[int, ...],
    source: str,
    *,
    min_observations: int = 100,
) -> pd.DataFrame:
    """Select one raw-L3 grid/level per turbine using outer-train years only."""
    if source not in {"ldaps", "gfs"}:
        raise ValueError(f"Unsupported NWP source: {source}")
    if group not in GROUP_TURBINE_PREFIXES:
        raise ValueError(f"Unknown group: {group}")
    required = {"forecast_kst_dtm", "turbine_id", "scada_ws_mean"}
    missing = sorted(required.difference(targets.columns))
    if missing:
        raise ValueError(f"Group quota v2 targets missing columns: {missing}")

    source_indices = np.asarray(
        [
            index
            for index, name in enumerate(candidates.names)
            if str(name).startswith(f"{source}|")
        ],
        dtype=int,
    )
    if len(source_indices) == 0:
        raise ValueError(f"No {source} candidates found")

    forecast_times = pd.to_datetime(candidates.keys["forecast_kst_dtm"])
    fit_year = forecast_times.dt.year.isin(list(train_years)).to_numpy()
    candidate_values = candidates.ws[:, source_indices].astype(float)
    rows = []
    for turbine in GROUP_TURBINE_PREFIXES[group]:
        turbine_targets = targets.loc[
            targets["turbine_id"].eq(turbine),
            ["forecast_kst_dtm", "scada_ws_mean"],
        ].copy()
        turbine_targets["forecast_kst_dtm"] = pd.to_datetime(
            turbine_targets["forecast_kst_dtm"]
        )
        if turbine_targets["forecast_kst_dtm"].duplicated().any():
            raise ValueError(f"Duplicate wind targets for {turbine}")
        target_map = turbine_targets.set_index("forecast_kst_dtm")["scada_ws_mean"]
        target = pd.to_numeric(forecast_times.map(target_map), errors="coerce").to_numpy(
            float
        )
        observed = (
            fit_year[:, None]
            & np.isfinite(target)[:, None]
            & np.isfinite(candidate_values)
        )
        count = observed.sum(axis=0)
        absolute_cube_error = np.where(
            observed,
            np.abs(candidate_values - target[:, None]) ** 3,
            0.0,
        )
        mean_cube_error = np.divide(
            absolute_cube_error.sum(axis=0),
            np.maximum(count, 1),
        )
        l3_error = np.cbrt(mean_cube_error)
        l3_error[count < int(min_observations)] = np.inf
        selected_local = int(np.argmin(l3_error))
        if not np.isfinite(l3_error[selected_local]):
            raise ValueError(
                f"Too few {source} wind observations for {group}/{turbine}"
            )
        selected_global = int(source_indices[selected_local])
        candidate = candidates.names[selected_global]
        parsed_source, grid_id, level = _parse_candidate(candidate)
        rows.append(
            {
                "group": group,
                "turbine_id": turbine,
                "source": parsed_source,
                "grid_id": grid_id,
                "level": level,
                "candidate": candidate,
                "train_years": ",".join(map(str, train_years)),
                "selection_metric": "raw_l3",
                "train_selection_l3": float(l3_error[selected_local]),
                "n_train_wind": int(count[selected_local]),
            }
        )
    return pd.DataFrame(rows)


def build_selected_source_panel(
    raw: pd.DataFrame,
    selections: pd.DataFrame,
    group: str,
    source: str,
) -> pd.DataFrame:
    """Build a turbine-weighted selected-grid panel plus a group-centre anchor."""
    required = {"group", "turbine_id", "source", "grid_id"}
    missing = sorted(required.difference(selections.columns))
    if missing:
        raise ValueError(f"Grid selections missing columns: {missing}")
    selected_rows = selections.loc[
        selections["group"].eq(group) & selections["source"].eq(source)
    ].copy()
    expected_turbines = list(GROUP_TURBINE_PREFIXES[group])
    if selected_rows["turbine_id"].tolist() != expected_turbines:
        selected_rows = selected_rows.set_index("turbine_id").reindex(
            expected_turbines
        ).reset_index()
    if selected_rows[["source", "grid_id"]].isna().any().any():
        raise ValueError(f"Incomplete {source} selections for {group}")

    parts = []
    raw_grid = pd.to_numeric(raw["grid_id"], errors="coerce")
    for row in selected_rows.itertuples(index=False):
        grid_id = int(row.grid_id)
        part = raw.loc[raw_grid.eq(grid_id)].copy()
        if part.empty:
            raise ValueError(f"Missing {source} grid {grid_id} for {row.turbine_id}")
        part["grid_id"] = f"selected:{row.turbine_id}:{grid_id}"
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
    candidates: WindCandidateMatrix,
    targets: pd.DataFrame,
    group: str,
    train_years: list[int] | tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selection_parts = [
        select_group_source_grids(
            candidates, targets, group, train_years, source
        )
        for source in ("ldaps", "gfs")
    ]
    selections = pd.concat(selection_parts, ignore_index=True)
    ldaps_panel = build_selected_source_panel(
        ldaps, selections, group, "ldaps"
    )
    gfs_panel = build_selected_source_panel(gfs, selections, group, "gfs")
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
    return output, selections


def get_or_build_group_quota_v2(
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
    candidates: WindCandidateMatrix,
    targets: pd.DataFrame,
    group: str,
    pred_year: int,
    train_years: list[int] | tuple[int, ...],
    *,
    cache_root: Path = Path("cache"),
    rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = Path(cache_root) / GROUP_QUOTA_V2_CACHE_VERSION
    feature_path = cache_dir / f"{group}_pred{pred_year}_features.pkl"
    selection_path = cache_dir / f"{group}_pred{pred_year}_selections.csv"
    if feature_path.exists() and selection_path.exists() and not rebuild:
        return pd.read_pickle(feature_path), pd.read_csv(selection_path)

    features, selections = build_group_quota_v2(
        ldaps,
        gfs,
        candidates,
        targets,
        group,
        train_years,
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    features.to_pickle(feature_path)
    selections.to_csv(selection_path, index=False, encoding="utf-8-sig")
    return features, selections

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils.per_turbine_optimal_grid import OPTIMAL_GRID_FEATURES, WAKE_FEATURES
from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.preprocessing import TIME_KEY_COLS
from utils.tree_feature_profiles import GROUP_FAMILY_QUOTA65_V1_FEATURES


LOCAL_PANEL_FEATURES = [*WAKE_FEATURES, *OPTIMAL_GRID_FEATURES]


@dataclass(frozen=True)
class GroupLocalPanel:
    table: pd.DataFrame
    common_feature_cols: tuple[str, ...]
    mean_feature_cols: tuple[str, ...]
    full_feature_cols: tuple[str, ...]


def build_group_local_panel(
    features: pd.DataFrame,
    group: str,
    turbines: list[str] | tuple[str, ...] | None = None,
    common_feature_cols: list[str] | tuple[str, ...] | None = None,
    local_feature_cols: list[str] | tuple[str, ...] = LOCAL_PANEL_FEATURES,
) -> GroupLocalPanel:
    turbine_order = tuple(turbines or GROUP_TURBINE_PREFIXES[group])
    common_cols = tuple(
        common_feature_cols or GROUP_FAMILY_QUOTA65_V1_FEATURES[group]
    )
    local_cols = tuple(local_feature_cols)
    required = [*TIME_KEY_COLS, "turbine_id", *common_cols, *local_cols]
    missing = [column for column in required if column not in features.columns]
    if missing:
        raise ValueError(f"Group local panel missing columns: {missing}")

    work = features[required].copy()
    for column in TIME_KEY_COLS:
        work[column] = pd.to_datetime(work[column])
    work = work.loc[work["turbine_id"].isin(turbine_order)]
    if work.duplicated([*TIME_KEY_COLS, "turbine_id"]).any():
        raise ValueError("Group local panel has duplicate turbine/time rows")

    present = set(work["turbine_id"].unique())
    missing_turbines = [turbine for turbine in turbine_order if turbine not in present]
    if missing_turbines:
        raise ValueError(f"Group local panel missing turbines: {missing_turbines}")

    reference = (
        work.loc[work["turbine_id"].eq(turbine_order[0]), [*TIME_KEY_COLS, *common_cols]]
        .sort_values(TIME_KEY_COLS)
        .reset_index(drop=True)
    )
    reference_index = pd.MultiIndex.from_frame(reference[TIME_KEY_COLS])
    reference_common = reference[list(common_cols)].to_numpy(dtype=float)
    for turbine in turbine_order[1:]:
        current = (
            work.loc[work["turbine_id"].eq(turbine), [*TIME_KEY_COLS, *common_cols]]
            .set_index(TIME_KEY_COLS)
            .reindex(reference_index)
        )
        if current[list(common_cols)].isna().all(axis=1).any():
            raise ValueError(f"Group local panel has incomplete rows for {turbine}")
        if not np.allclose(
            current[list(common_cols)].to_numpy(dtype=float),
            reference_common,
            equal_nan=True,
        ):
            raise ValueError(f"Common weather differs across turbines for {turbine}")

    panel = reference.copy()
    mean_names = []
    local_mean = (
        work.groupby(TIME_KEY_COLS, sort=True)[list(local_cols)]
        .mean()
        .reindex(reference_index)
    )
    for column in local_cols:
        output = f"mean__{column}"
        panel[output] = local_mean[column].to_numpy(dtype=float)
        mean_names.append(output)

    full_names = []
    for turbine in turbine_order:
        local = (
            work.loc[work["turbine_id"].eq(turbine), [*TIME_KEY_COLS, *local_cols]]
            .set_index(TIME_KEY_COLS)
            .reindex(reference_index)
        )
        if local[list(local_cols)].isna().all(axis=1).any():
            raise ValueError(f"Group local panel has incomplete local rows for {turbine}")
        for column in local_cols:
            output = f"{turbine}__{column}"
            panel[output] = local[column].to_numpy(dtype=float)
            full_names.append(output)

    return GroupLocalPanel(
        table=panel,
        common_feature_cols=common_cols,
        mean_feature_cols=tuple([*common_cols, *mean_names]),
        full_feature_cols=tuple([*common_cols, *full_names]),
    )

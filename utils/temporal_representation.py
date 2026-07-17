from __future__ import annotations

import pandas as pd


REPRESENTATIONS = ("original", "min", "median", "max")
ISSUE_KEYS = ["turbine_id", "data_available_kst_dtm"]


def is_temporally_aggregatable(column: str) -> bool:
    """Return whether a scalar feature can be pooled over forecast horizons."""
    name = column.lower()
    if (
        name.startswith(("sin_", "cos_"))
        or name.endswith(("_sin", "_cos"))
        or "optgrid_wd_" in name
    ):
        return False
    vector_markers = (
        "_10u",
        "_10v",
        "_100u",
        "_100v",
        "_50mu",
        "_50mv",
        "_xblws",
        "_yblws",
    )
    if any(marker in name for marker in vector_markers):
        return False
    if name.endswith(("_u", "_v")):
        return False
    return True


def temporal_aggregation_columns(feature_cols: list[str]) -> list[str]:
    return [column for column in feature_cols if is_temporally_aggregatable(column)]


def apply_issue_temporal_representation(
    table: pd.DataFrame,
    feature_cols: list[str],
    representation: str,
    window: int = 7,
) -> tuple[pd.DataFrame, list[str]]:
    """Replace scalar inputs with centered same-issue horizon summaries."""
    if representation not in REPRESENTATIONS:
        raise ValueError(
            f"Unknown temporal representation {representation!r}; "
            f"expected one of {REPRESENTATIONS}"
        )
    if window < 1 or window % 2 == 0:
        raise ValueError("Temporal representation window must be a positive odd number")

    missing = [
        column
        for column in [*ISSUE_KEYS, "forecast_kst_dtm", *feature_cols]
        if column not in table.columns
    ]
    if missing:
        raise ValueError(f"Temporal representation missing columns: {missing}")

    aggregate_cols = temporal_aggregation_columns(feature_cols)
    if representation == "original":
        return table.copy(), aggregate_cols
    if not aggregate_cols:
        raise ValueError("No scalar features available for temporal aggregation")

    work = table.copy()
    work["forecast_kst_dtm"] = pd.to_datetime(work["forecast_kst_dtm"])
    work["data_available_kst_dtm"] = pd.to_datetime(
        work["data_available_kst_dtm"]
    )
    work["__temporal_original_order"] = range(len(work))
    work = work.sort_values(
        [*ISSUE_KEYS, "forecast_kst_dtm", "__temporal_original_order"]
    )

    grouped_rolling = work.groupby(ISSUE_KEYS, sort=False)[aggregate_cols].rolling(
        window=window,
        center=True,
        min_periods=1,
    )
    rolled = getattr(grouped_rolling, representation)().reset_index(
        level=ISSUE_KEYS,
        drop=True,
    )
    rolled = rolled.reindex(work.index)
    if rolled.isna().all(axis=1).any():
        raise ValueError("Temporal representation produced an all-missing feature row")
    work.loc[:, aggregate_cols] = rolled.to_numpy()

    work = work.sort_values("__temporal_original_order").drop(
        columns="__temporal_original_order"
    )
    return work, aggregate_cols

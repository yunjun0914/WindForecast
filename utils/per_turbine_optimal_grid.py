from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.preprocessing import TIME_KEY_COLS
from utils.tree_feature_profiles import GROUP_FAMILY_QUOTA65_V1_FEATURES


OPTIMAL_GRID_CACHE_VERSION = "per_turbine_optimal_grid_v1"
OPTIMAL_GRID_CUBIC_CACHE_VERSION = "per_turbine_cubic_grid_v1"
OPTIMAL_GRID_REPLACE_TAG = "optimal_grid_replace_local16_v1"
OPTIMAL_GRID_ISSUE_CONTEXT_TAG = "optimal_grid_issue_context_v1"
OPTIMAL_GRID_ISSUE_RELATIVE_TAG = "optimal_grid_issue_relative_v1"
OPTIMAL_GRID_CUBIC_ISSUE_CONTEXT_TAG = "cubic_grid_issue_context_v1"
OPTIMAL_GRID_FEATURES = [
    "optgrid_ws_raw",
    "optgrid_ws_calibrated",
    "optgrid_ws_cube",
    "optgrid_wd_sin",
    "optgrid_wd_cos",
]
OPTIMAL_GRID_ISSUE_CONTEXT_FEATURES = [
    "optgrid_ws_calibrated_lead1",
    "optgrid_ws_calibrated_lead3",
    "optgrid_ws_calibrated_center7_mean",
    "optgrid_ws_calibrated_center7_max",
]
OPTIMAL_GRID_ISSUE_RELATIVE_FEATURES = [
    "optgrid_ws_calibrated_issue24_mean",
    "optgrid_ws_calibrated_issue24_max",
    "optgrid_ws_calibrated_issue24_rank_pct",
]
WAKE_FEATURES = ["wake_exposure", "wake_upstream_count"]


def optimal_grid_cache_version(wind_target: str) -> str:
    if wind_target == "mean":
        return OPTIMAL_GRID_CACHE_VERSION
    if wind_target == "cubic":
        return OPTIMAL_GRID_CUBIC_CACHE_VERSION
    raise ValueError(f"Unknown optimal-grid wind target: {wind_target}")


def optimal_grid_input_columns(
    group: str,
    include_issue_context: bool = False,
    include_issue_relative: bool = False,
) -> list[str]:
    context = (
        OPTIMAL_GRID_ISSUE_CONTEXT_FEATURES
        if include_issue_context or include_issue_relative
        else []
    )
    relative = OPTIMAL_GRID_ISSUE_RELATIVE_FEATURES if include_issue_relative else []
    return [
        *GROUP_FAMILY_QUOTA65_V1_FEATURES[group],
        *WAKE_FEATURES,
        *OPTIMAL_GRID_FEATURES,
        *context,
        *relative,
    ]


def add_optimal_grid_issue_context(optimal: pd.DataFrame) -> pd.DataFrame:
    required = [
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "turbine_id",
        "optgrid_ws_calibrated",
    ]
    missing = [col for col in required if col not in optimal.columns]
    if missing:
        raise ValueError(f"Optimal-grid issue context missing columns: {missing}")

    out = optimal.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out["data_available_kst_dtm"] = pd.to_datetime(out["data_available_kst_dtm"])
    out = out.sort_values(
        ["turbine_id", "data_available_kst_dtm", "forecast_kst_dtm"]
    ).copy()
    current = pd.to_numeric(out["optgrid_ws_calibrated"], errors="coerce")
    grouped = out.groupby(
        ["turbine_id", "data_available_kst_dtm"], sort=False
    )["optgrid_ws_calibrated"]
    out["optgrid_ws_calibrated_lead1"] = grouped.shift(-1).fillna(current)
    out["optgrid_ws_calibrated_lead3"] = grouped.shift(-3).fillna(current)
    out["optgrid_ws_calibrated_center7_mean"] = grouped.transform(
        lambda values: values.rolling(window=7, center=True, min_periods=1).mean()
    )
    out["optgrid_ws_calibrated_center7_max"] = grouped.transform(
        lambda values: values.rolling(window=7, center=True, min_periods=1).max()
    )
    return out.reset_index(drop=True)


def add_optimal_grid_issue_relative_features(optimal: pd.DataFrame) -> pd.DataFrame:
    out = add_optimal_grid_issue_context(optimal)
    grouped = out.groupby(
        ["turbine_id", "data_available_kst_dtm"], sort=False
    )["optgrid_ws_calibrated"]
    out["optgrid_ws_calibrated_issue24_mean"] = grouped.transform("mean")
    out["optgrid_ws_calibrated_issue24_max"] = grouped.transform("max")
    out["optgrid_ws_calibrated_issue24_rank_pct"] = grouped.rank(
        method="average", pct=True
    )
    return out


def load_optimal_grid_fold_features(
    base_features: pd.DataFrame,
    cache_root: Path,
    group: str,
    pred_year: int,
    wind_target: str = "mean",
    include_issue_context: bool = False,
    include_issue_relative: bool = False,
) -> pd.DataFrame:
    cache_path = (
        cache_root
        / optimal_grid_cache_version(wind_target)
        / f"{group}_pred{pred_year}_features.pkl"
    )
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Optimal-grid fold cache is missing; run the selection OOF first: {cache_path}"
        )
    optimal = pd.read_pickle(cache_path)
    keys = [*TIME_KEY_COLS, "turbine_id"]
    missing = [col for col in [*keys, *OPTIMAL_GRID_FEATURES] if col not in optimal.columns]
    if missing:
        raise ValueError(f"Optimal-grid cache missing columns: {missing}")
    if include_issue_relative:
        optimal = add_optimal_grid_issue_relative_features(optimal)
    elif include_issue_context:
        optimal = add_optimal_grid_issue_context(optimal)
    feature_cols = [*OPTIMAL_GRID_FEATURES]
    if include_issue_context or include_issue_relative:
        feature_cols.extend(OPTIMAL_GRID_ISSUE_CONTEXT_FEATURES)
    if include_issue_relative:
        feature_cols.extend(OPTIMAL_GRID_ISSUE_RELATIVE_FEATURES)
    before = len(base_features)
    out = base_features.merge(optimal[keys + feature_cols], on=keys, how="left")
    if len(out) != before:
        raise ValueError(f"Optimal-grid merge changed row count: {before} -> {len(out)}")
    coverage = float(out["optgrid_ws_raw"].notna().mean())
    if coverage < 0.95:
        raise ValueError(f"Optimal-grid feature coverage too low: {group} {pred_year} {coverage}")
    return out


def _merge_optimal_grid_features(
    base_features: pd.DataFrame,
    optimal: pd.DataFrame,
    label: str,
    include_issue_context: bool = False,
    include_issue_relative: bool = False,
) -> pd.DataFrame:
    keys = [*TIME_KEY_COLS, "turbine_id"]
    if include_issue_relative:
        optimal = add_optimal_grid_issue_relative_features(optimal)
    elif include_issue_context:
        optimal = add_optimal_grid_issue_context(optimal)
    feature_cols = [*OPTIMAL_GRID_FEATURES]
    if include_issue_context or include_issue_relative:
        feature_cols.extend(OPTIMAL_GRID_ISSUE_CONTEXT_FEATURES)
    if include_issue_relative:
        feature_cols.extend(OPTIMAL_GRID_ISSUE_RELATIVE_FEATURES)
    before = len(base_features)
    out = base_features.merge(optimal[keys + feature_cols], on=keys, how="left")
    if len(out) != before:
        raise ValueError(f"{label} optimal-grid merge changed row count: {before} -> {len(out)}")
    coverage = float(out["optgrid_ws_raw"].notna().mean())
    if coverage < 0.95:
        raise ValueError(f"{label} optimal-grid feature coverage too low: {coverage}")
    return out


def load_optimal_grid_full_features(
    train_features: pd.DataFrame,
    test_features: pd.DataFrame,
    cache_root: Path,
    group: str,
    wind_target: str = "mean",
    include_issue_context: bool = False,
    include_issue_relative: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = cache_root / optimal_grid_cache_version(wind_target)
    train_path = cache_dir / f"{group}_full_train_features.pkl"
    test_path = cache_dir / f"{group}_full_test_features.pkl"
    for path in [train_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Full optimal-grid cache is missing; build it before prediction: {path}"
            )
    train_optimal = pd.read_pickle(train_path)
    test_optimal = pd.read_pickle(test_path)
    return (
        _merge_optimal_grid_features(
            train_features,
            train_optimal,
            f"{group} train",
            include_issue_context=include_issue_context,
            include_issue_relative=include_issue_relative,
        ),
        _merge_optimal_grid_features(
            test_features,
            test_optimal,
            f"{group} test",
            include_issue_context=include_issue_context,
            include_issue_relative=include_issue_relative,
        ),
    )

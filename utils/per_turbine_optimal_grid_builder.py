from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from utils.per_turbine_features import GFS_LEVELS, LDAPS_LEVELS
from utils.per_turbine_optimal_grid import OPTIMAL_GRID_FEATURES
from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.preprocessing import TIME_KEY_COLS


REBUILT_OPTIMAL_GRID_CACHE_VERSION = "per_turbine_optimal_grid_rebuilt_v3_cubic"


@dataclass(frozen=True)
class WindCandidateMatrix:
    keys: pd.DataFrame
    names: tuple[str, ...]
    u: np.ndarray
    v: np.ndarray
    ws: np.ndarray


def _candidate_name(source: str, grid_id: object, level: str) -> str:
    return f"{source}|{grid_id}|{level}"


def build_wind_candidate_matrix(
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
) -> WindCandidateMatrix:
    """Build the 16x4 LDAPS + 9x6 GFS wind candidates on common issue rows."""
    prepared = []
    for source, raw, levels in [
        ("ldaps", ldaps, LDAPS_LEVELS),
        ("gfs", gfs, GFS_LEVELS),
    ]:
        table = raw.copy()
        for column in TIME_KEY_COLS:
            table[column] = pd.to_datetime(table[column])
        prepared.append((source, table, levels))

    keys = (
        pd.concat([table[TIME_KEY_COLS] for _, table, _ in prepared], ignore_index=True)
        .drop_duplicates()
        .sort_values(TIME_KEY_COLS)
        .reset_index(drop=True)
    )
    key_index = pd.MultiIndex.from_frame(keys)
    names: list[str] = []
    u_parts: list[np.ndarray] = []
    v_parts: list[np.ndarray] = []

    for source, table, levels in prepared:
        for grid_id, grid in table.groupby("grid_id", sort=True):
            grid = grid.set_index(TIME_KEY_COLS)
            if not grid.index.is_unique:
                raise ValueError(f"Duplicate {source} issue rows for grid {grid_id}")
            grid = grid.reindex(key_index)
            for level, (u_column, v_column) in levels.items():
                names.append(_candidate_name(source, grid_id, level))
                u_parts.append(pd.to_numeric(grid[u_column], errors="coerce").to_numpy(float))
                v_parts.append(pd.to_numeric(grid[v_column], errors="coerce").to_numpy(float))

    u = np.column_stack(u_parts).astype(np.float32)
    v = np.column_stack(v_parts).astype(np.float32)
    ws = np.hypot(u, v).astype(np.float32)
    if len(names) != 118:
        raise ValueError(f"Expected 118 wind candidates, found {len(names)}")
    return WindCandidateMatrix(keys=keys, names=tuple(names), u=u, v=v, ws=ws)


def _fit_affine_candidates(
    candidates: np.ndarray,
    target: np.ndarray,
    fit_mask: np.ndarray,
    selection_metric: str = "mae",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    observed = (
        fit_mask[:, None]
        & np.isfinite(candidates)
        & np.isfinite(target)[:, None]
    )
    count = observed.sum(axis=0)
    if int(count.max(initial=0)) < 100:
        raise ValueError("Too few wind observations to select an optimal grid")

    safe_count = np.maximum(count, 1)
    x = np.where(observed, candidates, 0.0)
    y = np.where(observed, target[:, None], 0.0)
    mean_x = x.sum(axis=0) / safe_count
    mean_y = y.sum(axis=0) / safe_count
    centered_x = np.where(observed, candidates - mean_x, 0.0)
    centered_y = np.where(observed, target[:, None] - mean_y, 0.0)
    variance = np.square(centered_x).sum(axis=0)
    covariance = (centered_x * centered_y).sum(axis=0)
    slope = np.divide(
        covariance,
        variance,
        out=np.ones_like(covariance, dtype=float),
        where=variance > 1e-8,
    )
    slope = np.clip(slope, 0.25, 4.0)
    intercept = np.clip(mean_y - slope * mean_x, -10.0, 10.0)
    calibrated = np.clip(candidates * slope + intercept, 0.0, 40.0)
    if selection_metric == "mae":
        absolute_error = np.abs(calibrated - target[:, None])
    elif selection_metric == "cubic_mae":
        absolute_error = np.abs(calibrated**3 - target[:, None] ** 3)
    else:
        raise ValueError(f"Unknown optimal-grid selection metric: {selection_metric}")
    absolute_error = np.where(observed, absolute_error, 0.0)
    selection_error = absolute_error.sum(axis=0) / safe_count
    selection_error[count < 100] = np.inf
    return slope, intercept, selection_error, count


def select_optimal_grid_features(
    candidates: WindCandidateMatrix,
    targets: pd.DataFrame,
    group: str,
    train_years: list[int],
    wind_target: str = "scada_ws_mean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = ["forecast_kst_dtm", "turbine_id", wind_target]
    missing = [column for column in required if column not in targets.columns]
    if missing:
        raise ValueError(f"Optimal-grid targets missing columns: {missing}")

    forecast_times = pd.to_datetime(candidates.keys["forecast_kst_dtm"])
    fit_year = forecast_times.dt.year.isin(train_years).to_numpy()
    feature_parts = []
    selection_rows = []
    for turbine in GROUP_TURBINE_PREFIXES[group]:
        turbine_target = targets.loc[
            targets["turbine_id"].eq(turbine),
            ["forecast_kst_dtm", wind_target],
        ].copy()
        turbine_target["forecast_kst_dtm"] = pd.to_datetime(
            turbine_target["forecast_kst_dtm"]
        )
        if turbine_target["forecast_kst_dtm"].duplicated().any():
            raise ValueError(f"Duplicate SCADA wind targets for {turbine}")
        target_map = turbine_target.set_index("forecast_kst_dtm")[wind_target]
        target = pd.to_numeric(
            forecast_times.map(target_map), errors="coerce"
        ).to_numpy(float)
        selection_metric = (
            "cubic_mae" if wind_target == "scada_ws_cubic" else "mae"
        )
        slope, intercept, selection_error, count = _fit_affine_candidates(
            candidates.ws,
            target,
            fit_year,
            selection_metric=selection_metric,
        )
        selected = int(np.argmin(selection_error))
        raw_ws = candidates.ws[:, selected].astype(float)
        calibrated = np.clip(
            slope[selected] * raw_ws + intercept[selected], 0.0, 40.0
        )
        selected_u = candidates.u[:, selected].astype(float)
        selected_v = candidates.v[:, selected].astype(float)
        speed = np.maximum(raw_ws, 1e-6)

        features = candidates.keys.copy()
        features["turbine_id"] = turbine
        features["optgrid_ws_raw"] = raw_ws
        features["optgrid_ws_calibrated"] = calibrated
        features["optgrid_ws_cube"] = calibrated**3
        features["optgrid_wd_sin"] = selected_v / speed
        features["optgrid_wd_cos"] = selected_u / speed
        feature_parts.append(features)
        selection_rows.append(
            {
                "group": group,
                "turbine_id": turbine,
                "train_years": ",".join(map(str, train_years)),
                "candidate": candidates.names[selected],
                "slope": float(slope[selected]),
                "intercept": float(intercept[selected]),
                "selection_metric": selection_metric,
                "train_selection_error": float(selection_error[selected]),
                "n_train_wind": int(count[selected]),
            }
        )

    return pd.concat(feature_parts, ignore_index=True), pd.DataFrame(selection_rows)


def get_or_build_optimal_grid_fold(
    candidates: WindCandidateMatrix,
    targets: pd.DataFrame,
    group: str,
    pred_year: int,
    train_years: list[int],
    cache_root: Path = Path("cache"),
    rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache_dir = cache_root / REBUILT_OPTIMAL_GRID_CACHE_VERSION
    feature_path = cache_dir / f"{group}_pred{pred_year}_features.pkl"
    selection_path = cache_dir / f"{group}_pred{pred_year}_selection.csv"
    if feature_path.exists() and selection_path.exists() and not rebuild:
        return pd.read_pickle(feature_path), pd.read_csv(selection_path)

    features, selections = select_optimal_grid_features(
        candidates,
        targets,
        group,
        train_years,
    )
    missing = [column for column in OPTIMAL_GRID_FEATURES if column not in features]
    if missing:
        raise ValueError(f"Built optimal-grid features missing columns: {missing}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    features.to_pickle(feature_path)
    selections.to_csv(selection_path, index=False, encoding="utf-8-sig")
    return features, selections

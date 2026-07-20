from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # type: ignore[no-redef]  # noqa: F401

    sys.modules.setdefault("_bootstrap", _bootstrap)

from utils.metrics import TARGET_COLS
from utils.per_turbine_features import GFS_LEVELS, LDAPS_LEVELS
from utils.per_turbine_optimal_grid import OPTIMAL_GRID_CACHE_VERSION
from utils.per_turbine_scada import build_official_aligned_turbine_targets
from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.preprocessing import TIME_KEY_COLS, haversine_km
from utils.site_metadata import load_turbine_metadata


YEARS = [2022, 2023, 2024]
WIND_LEVELS = {"ldaps": LDAPS_LEVELS, "gfs": GFS_LEVELS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def build_candidate_catalog(
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
) -> dict[str, dict[str, object]]:
    catalog: dict[str, dict[str, object]] = {}
    for source, raw in (("ldaps", ldaps), ("gfs", gfs)):
        work = raw.copy()
        for column in TIME_KEY_COLS:
            work[column] = pd.to_datetime(work[column])
        grid_ids = sorted(int(value) for value in work["grid_id"].dropna().unique())
        for grid_id in grid_ids:
            grid = work.loc[work["grid_id"].eq(grid_id)].copy()
            latitude = float(grid["latitude"].iloc[0])
            longitude = float(grid["longitude"].iloc[0])
            for level, (u_col, v_col) in WIND_LEVELS[source].items():
                name = f"{source}__grid{grid_id}__{level}"
                frame = grid[[*TIME_KEY_COLS, u_col, v_col]].rename(
                    columns={u_col: "u", v_col: "v"}
                )
                frame["ws"] = np.hypot(frame["u"], frame["v"])
                frame = (
                    frame.drop_duplicates(TIME_KEY_COLS)
                    .sort_values(TIME_KEY_COLS)
                    .reset_index(drop=True)
                )
                catalog[name] = {
                    "source": source,
                    "grid_id": grid_id,
                    "level": level,
                    "latitude": latitude,
                    "longitude": longitude,
                    "frame": frame,
                }
    return catalog


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 200 or float(np.var(x)) < 1e-8:
        return 1.0, 0.0
    slope = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x))
    slope = float(np.clip(slope, 0.0, 3.0))
    intercept = float(np.mean(y) - slope * np.mean(x))
    return slope, intercept


def calibration_splits(table: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    years = sorted(table["year"].unique().tolist())
    if len(years) >= 2:
        return [
            (
                table["year"].ne(held_year).to_numpy(),
                table["year"].eq(held_year).to_numpy(),
            )
            for held_year in years
        ]
    month = table["forecast_kst_dtm"].dt.month
    parity = month.mod(2)
    return [
        (parity.ne(value).to_numpy(), parity.eq(value).to_numpy())
        for value in sorted(parity.unique().tolist())
    ]


def calibrated_cv_mae(
    candidate: pd.DataFrame,
    target: pd.DataFrame,
    train_years: list[int],
) -> tuple[float, int]:
    one = candidate[["forecast_kst_dtm", "ws"]].drop_duplicates(
        "forecast_kst_dtm"
    )
    table = target.merge(one, on="forecast_kst_dtm", how="inner")
    table["year"] = table["forecast_kst_dtm"].dt.year
    table = table.loc[table["year"].isin(train_years)].dropna(
        subset=["scada_ws_mean", "ws"]
    )
    errors = []
    for fit_mask, validation_mask in calibration_splits(table):
        if int(fit_mask.sum()) < 200 or int(validation_mask.sum()) < 100:
            continue
        slope, intercept = fit_linear(
            table.loc[fit_mask, "ws"].to_numpy(float),
            table.loc[fit_mask, "scada_ws_mean"].to_numpy(float),
        )
        prediction = np.clip(
            intercept
            + slope * table.loc[validation_mask, "ws"].to_numpy(float),
            0.0,
            40.0,
        )
        errors.append(
            np.abs(
                prediction
                - table.loc[validation_mask, "scada_ws_mean"].to_numpy(float)
            )
        )
    if not errors:
        return float("inf"), len(table)
    return float(np.mean(np.concatenate(errors))), len(table)


def _build_selected_features(
    candidate: pd.DataFrame,
    turbine_id: str,
    slope: float,
    intercept: float,
) -> pd.DataFrame:
    raw_ws = candidate["ws"].to_numpy(float)
    speed = np.maximum(raw_ws, 1e-6)
    selected = candidate[[*TIME_KEY_COLS]].copy()
    selected["turbine_id"] = turbine_id
    selected["optgrid_ws_raw"] = raw_ws
    selected["optgrid_ws_calibrated"] = np.clip(
        intercept + slope * raw_ws, 0.0, 40.0
    )
    selected["optgrid_ws_cube"] = selected["optgrid_ws_calibrated"] ** 3
    selected["optgrid_wd_sin"] = candidate["v"].to_numpy(float) / speed
    selected["optgrid_wd_cos"] = candidate["u"].to_numpy(float) / speed
    return selected


def select_and_build_optimal_grid(
    catalog: dict[str, dict[str, object]],
    targets: pd.DataFrame,
    group: str,
    train_years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = load_turbine_metadata().set_index("turbine_id")
    feature_parts = []
    selection_rows = []
    for turbine_id in GROUP_TURBINE_PREFIXES[group]:
        target = targets.loc[
            targets["turbine_id"].eq(turbine_id),
            ["forecast_kst_dtm", "scada_ws_mean"],
        ].copy()
        target["forecast_kst_dtm"] = pd.to_datetime(target["forecast_kst_dtm"])

        ranked = []
        for name, candidate in catalog.items():
            cv_mae, n_rows = calibrated_cv_mae(
                candidate["frame"], target, train_years
            )
            ranked.append((cv_mae, name, n_rows))
        cv_mae, name, n_rows = min(ranked)
        chosen = catalog[name]
        frame = chosen["frame"]

        calibration = target.merge(
            frame[["forecast_kst_dtm", "ws"]].drop_duplicates(
                "forecast_kst_dtm"
            ),
            on="forecast_kst_dtm",
            how="inner",
        )
        calibration["year"] = calibration["forecast_kst_dtm"].dt.year
        calibration = calibration.loc[
            calibration["year"].isin(train_years)
        ].dropna(subset=["scada_ws_mean", "ws"])
        slope, intercept = fit_linear(
            calibration["ws"].to_numpy(float),
            calibration["scada_ws_mean"].to_numpy(float),
        )
        feature_parts.append(
            _build_selected_features(frame, turbine_id, slope, intercept)
        )

        turbine = metadata.loc[turbine_id]
        distance = float(
            haversine_km(
                float(turbine["latitude"]),
                float(turbine["longitude"]),
                float(chosen["latitude"]),
                float(chosen["longitude"]),
            )
        )
        selection_rows.append(
            {
                "group": group,
                "turbine_id": turbine_id,
                "train_years": ",".join(str(year) for year in train_years),
                "candidate": name,
                "source": chosen["source"],
                "grid_id": chosen["grid_id"],
                "level": chosen["level"],
                "distance_km": distance,
                "cv_ws_mae": cv_mae,
                "calibration_slope": slope,
                "calibration_intercept": intercept,
                "n_selection_rows": n_rows,
            }
        )
    return pd.concat(feature_parts, ignore_index=True), pd.DataFrame(selection_rows)


def apply_selection(
    catalog: dict[str, dict[str, object]],
    selection: pd.DataFrame,
) -> pd.DataFrame:
    parts = []
    for row in selection.itertuples(index=False):
        if row.candidate not in catalog:
            raise ValueError(f"Selected candidate is missing: {row.candidate}")
        parts.append(
            _build_selected_features(
                catalog[row.candidate]["frame"],
                row.turbine_id,
                float(row.calibration_slope),
                float(row.calibration_intercept),
            )
        )
    return pd.concat(parts, ignore_index=True)


def cache_paths(cache_dir: Path, group: str, pred_year: int) -> tuple[Path, Path]:
    return (
        cache_dir / f"{group}_pred{pred_year}_features.pkl",
        cache_dir / f"{group}_pred{pred_year}_selection.csv",
    )


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_root / OPTIMAL_GRID_CACHE_VERSION
    cache_dir.mkdir(parents=True, exist_ok=True)

    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    ldaps_test = pd.read_csv("data/test/ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv("data/test/gfs_test.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    scada_vestas = pd.read_csv(
        "data/train/scada_vestas_train.csv", encoding="utf-8-sig"
    )
    scada_unison = pd.read_csv(
        "data/train/scada_unison_train.csv", encoding="utf-8-sig"
    )
    scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }
    train_catalog = build_candidate_catalog(ldaps_train, gfs_train)
    test_catalog = build_candidate_catalog(ldaps_test, gfs_test)
    missing_test_candidates = sorted(set(train_catalog) - set(test_catalog))
    if missing_test_candidates:
        raise ValueError(f"Test catalog is incomplete: {missing_test_candidates}")
    print(f"candidate catalog={len(train_catalog)}", flush=True)

    for group in TARGET_COLS:
        targets = build_official_aligned_turbine_targets(
            scada_by_group[group], labels, group
        )
        for pred_year in YEARS:
            if (
                labels.loc[
                    labels["kst_dtm"].dt.year.eq(pred_year), group
                ].notna().sum()
                < 200
            ):
                continue
            feature_path, selection_path = cache_paths(
                cache_dir, group, pred_year
            )
            if feature_path.exists() and selection_path.exists() and not args.rebuild:
                print(f"reuse {group} pred={pred_year}", flush=True)
                continue
            train_years = [year for year in YEARS if year != pred_year]
            features, selection = select_and_build_optimal_grid(
                train_catalog, targets, group, train_years
            )
            features.to_pickle(feature_path)
            selection.to_csv(selection_path, index=False, encoding="utf-8-sig")
            print(
                f"built {group} pred={pred_year} turbines="
                f"{selection['turbine_id'].nunique()}",
                flush=True,
            )

        full_train_path = cache_dir / f"{group}_full_train_features.pkl"
        full_test_path = cache_dir / f"{group}_full_test_features.pkl"
        full_selection_path = cache_dir / f"{group}_full_selection.csv"
        full_paths = (full_train_path, full_test_path, full_selection_path)
        if all(path.exists() for path in full_paths) and not args.rebuild:
            print(f"reuse {group} full", flush=True)
            continue
        target_years = sorted(
            targets.loc[targets["scada_ws_mean"].notna(), "year"].unique().tolist()
        )
        train_features, selection = select_and_build_optimal_grid(
            train_catalog, targets, group, target_years
        )
        test_features = apply_selection(test_catalog, selection)
        train_features.to_pickle(full_train_path)
        test_features.to_pickle(full_test_path)
        selection.to_csv(full_selection_path, index=False, encoding="utf-8-sig")
        expected = len(GROUP_TURBINE_PREFIXES[group])
        if selection["turbine_id"].nunique() != expected:
            raise ValueError(f"Incomplete full selection for {group}")
        print(f"built {group} full turbines={expected}", flush=True)


if __name__ == "__main__":
    main()

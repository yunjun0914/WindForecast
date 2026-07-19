from __future__ import annotations

import argparse
import math
import re
import shutil
import tempfile
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd


SCADA_MIN_SAMPLES_PER_HOUR = 4
STRONG_WIND_THRESHOLD_MS = 10.0


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="터빈별 시간 평균 SCADA 바람과 모든 NWP 풍속 proxy/격자를 비교합니다."
    )
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "reports" / "eda")
    return parser.parse_args()


def write_csv(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False, encoding="utf-8-sig")


def dms_to_decimal(value: object) -> tuple[float, float]:
    match = re.search(
        r"(\d+)[^\d]+(\d+)[^\d]+([\d.]+)\"?([NS])\s+"
        r"(\d+)[^\d]+(\d+)[^\d]+([\d.]+)\"?([EW])",
        str(value),
    )
    if match is None:
        return np.nan, np.nan
    lat_d, lat_m, lat_s, lat_h, lon_d, lon_m, lon_s, lon_h = match.groups()
    lat = float(lat_d) + float(lat_m) / 60.0 + float(lat_s) / 3600.0
    lon = float(lon_d) + float(lon_m) / 60.0 + float(lon_s) / 3600.0
    if lat_h == "S":
        lat = -lat
    if lon_h == "W":
        lon = -lon
    return lat, lon


def load_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_excel(path, sheet_name="info", header=3, usecols="B:L")
    metadata = metadata.dropna(subset=["제작사", "호기"]).copy()
    metadata["KPX그룹"] = metadata["KPX그룹"].ffill().astype(int)
    metadata["그룹설비용량(MW)"] = metadata["그룹설비용량(MW)"].ffill()
    metadata["호기"] = metadata["호기"].astype(int)
    metadata["turbine_id"] = [
        f"{str(maker).strip().lower()}_wtg{unit:02d}"
        for maker, unit in zip(metadata["제작사"], metadata["호기"])
    ]
    coordinates = metadata["좌표(Google)"].map(dms_to_decimal)
    metadata["turbine_latitude"] = [item[0] for item in coordinates]
    metadata["turbine_longitude"] = [item[1] for item in coordinates]
    metadata["group"] = metadata["KPX그룹"].map(lambda value: f"kpx_group_{value}")
    return metadata[
        [
            "turbine_id",
            "group",
            "제작사",
            "모델명",
            "호기",
            "Hub Height(m)",
            "Rotor Diameter(m)",
            "설비용량(MW)",
            "turbine_latitude",
            "turbine_longitude",
        ]
    ].rename(
        columns={
            "제작사": "manufacturer",
            "모델명": "model",
            "호기": "unit",
            "Hub Height(m)": "hub_height_m",
            "Rotor Diameter(m)": "rotor_diameter_m",
            "설비용량(MW)": "capacity_mw",
        }
    )


def circular_hourly_mean(values: pd.Series, hour_end: pd.Series) -> pd.Series:
    radians = np.deg2rad(np.mod(values, 360.0))
    frame = pd.DataFrame(
        {
            "hour_end": hour_end,
            "sin": np.sin(radians),
            "cos": np.cos(radians),
            "valid": values.notna().astype(int),
        }
    )
    grouped = frame.groupby("hour_end", sort=True).agg(
        sin=("sin", "mean"), cos=("cos", "mean"), count=("valid", "sum")
    )
    direction = np.mod(
        np.rad2deg(np.arctan2(grouped["sin"], grouped["cos"])), 360.0
    )
    direction[grouped["count"] < SCADA_MIN_SAMPLES_PER_HOUR] = np.nan
    return direction


def load_scada_hourly(
    path: Path, prefix: str, turbine_count: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    turbines = [f"{prefix}_wtg{unit:02d}" for unit in range(1, turbine_count + 1)]
    usecols = ["kst_dtm"]
    for turbine in turbines:
        usecols.extend([f"{turbine}_ws", f"{turbine}_wd"])
    raw = pd.read_csv(path, usecols=usecols, encoding="utf-8-sig")
    raw["kst_dtm"] = pd.to_datetime(raw["kst_dtm"], errors="raise")
    raw = raw.sort_values("kst_dtm").reset_index(drop=True)
    raw["hour_end"] = raw["kst_dtm"].dt.ceil("h")

    hourly = pd.DataFrame(index=pd.Index(sorted(raw["hour_end"].unique()), name="kst_dtm"))
    summary_rows: list[dict[str, object]] = []
    for turbine in turbines:
        ws_col = f"{turbine}_ws"
        wd_col = f"{turbine}_wd"
        ws_grouped = raw.groupby("hour_end", sort=True)[ws_col]
        ws_hourly = ws_grouped.mean()
        ws_count = ws_grouped.count()
        ws_hourly[ws_count < SCADA_MIN_SAMPLES_PER_HOUR] = np.nan
        hourly[f"{turbine}__ws"] = ws_hourly.reindex(hourly.index)
        hourly[f"{turbine}__wd"] = circular_hourly_mean(
            raw[wd_col], raw["hour_end"]
        ).reindex(hourly.index)

        ws = pd.to_numeric(raw[ws_col], errors="coerce")
        wd = pd.to_numeric(raw[wd_col], errors="coerce")
        valid_hourly = int(hourly[f"{turbine}__ws"].notna().sum())
        summary_rows.append(
            {
                "turbine_id": turbine,
                "raw_rows": len(raw),
                "raw_start": raw["kst_dtm"].min(),
                "raw_end": raw["kst_dtm"].max(),
                "duplicate_timestamps": int(raw["kst_dtm"].duplicated().sum()),
                "ws_missing": int(ws.isna().sum()),
                "wd_missing": int(wd.isna().sum()),
                "ws_min": float(ws.min()),
                "ws_p01": float(ws.quantile(0.01)),
                "ws_p50": float(ws.quantile(0.50)),
                "ws_p99": float(ws.quantile(0.99)),
                "ws_max": float(ws.max()),
                "wd_outside_0_360": int(((wd < 0) | (wd >= 360)).sum()),
                "valid_hourly_ws": valid_hourly,
            }
        )
    return hourly, pd.DataFrame(summary_rows)


def speed_and_direction(u: pd.Series, v: pd.Series) -> tuple[pd.Series, pd.Series]:
    speed = np.hypot(u, v)
    direction_from = np.mod(np.rad2deg(np.arctan2(-u, -v)), 360.0)
    return speed, direction_from


def component_absmax(maximum: pd.Series, minimum: pd.Series) -> pd.Series:
    return pd.Series(
        np.where(maximum.abs() >= minimum.abs(), maximum, minimum),
        index=maximum.index,
    )


def load_ldaps_candidates(
    path: Path,
) -> tuple[dict[tuple[str, str, int], pd.DataFrame], pd.DataFrame, dict[str, object]]:
    cols = [
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "grid_id",
        "latitude",
        "longitude",
        "heightAboveGround_10_10u",
        "heightAboveGround_10_10v",
        "heightAboveGround_50_50MUmax",
        "heightAboveGround_50_50MUmin",
        "heightAboveGround_50_50MVmax",
        "heightAboveGround_50_50MVmin",
        "heightAboveGround_5_XBLWS",
        "heightAboveGround_5_YBLWS",
    ]
    frame = pd.read_csv(path, usecols=cols, encoding="utf-8-sig")
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    frame["data_available_kst_dtm"] = pd.to_datetime(frame["data_available_kst_dtm"])
    feature_pairs: dict[str, tuple[pd.Series, pd.Series]] = {
        "10m": (
            frame["heightAboveGround_10_10u"],
            frame["heightAboveGround_10_10v"],
        ),
        "50m_midpoint": (
            (
                frame["heightAboveGround_50_50MUmax"]
                + frame["heightAboveGround_50_50MUmin"]
            )
            / 2.0,
            (
                frame["heightAboveGround_50_50MVmax"]
                + frame["heightAboveGround_50_50MVmin"]
            )
            / 2.0,
        ),
        "50m_component_extreme_proxy": (
            component_absmax(
                frame["heightAboveGround_50_50MUmax"],
                frame["heightAboveGround_50_50MUmin"],
            ),
            component_absmax(
                frame["heightAboveGround_50_50MVmax"],
                frame["heightAboveGround_50_50MVmin"],
            ),
        ),
        "5m_boundary_layer": (
            frame["heightAboveGround_5_XBLWS"],
            frame["heightAboveGround_5_YBLWS"],
        ),
    }
    candidates: dict[tuple[str, str, int], pd.DataFrame] = {}
    for feature, (u, v) in feature_pairs.items():
        speed, direction = speed_and_direction(u, v)
        frame[f"{feature}__speed"] = speed
        frame[f"{feature}__direction"] = direction
    for grid_id, grid in frame.groupby("grid_id", sort=True):
        grid = grid.sort_values("forecast_kst_dtm").set_index("forecast_kst_dtm")
        for feature in feature_pairs:
            candidates[("ldaps", feature, int(grid_id))] = grid[
                [f"{feature}__speed", f"{feature}__direction"]
            ].rename(
                columns={
                    f"{feature}__speed": "speed",
                    f"{feature}__direction": "direction",
                }
            )
    grids = (
        frame[["grid_id", "latitude", "longitude"]]
        .drop_duplicates()
        .assign(source="ldaps")
    )
    structure = summarize_nwp_structure(frame, "ldaps")
    return candidates, grids, structure


def load_gfs_candidates(
    path: Path,
) -> tuple[dict[tuple[str, str, int], pd.DataFrame], pd.DataFrame, dict[str, object]]:
    pair_columns = {
        "10m": ("heightAboveGround_10_10u", "heightAboveGround_10_10v"),
        "80m": ("heightAboveGround_80_u", "heightAboveGround_80_v"),
        "100m": ("heightAboveGround_100_100u", "heightAboveGround_100_100v"),
        "pbl": ("planetaryBoundaryLayer_0_u", "planetaryBoundaryLayer_0_v"),
        "850hpa": ("isobaricInhPa_850_u", "isobaricInhPa_850_v"),
        "700hpa": ("isobaricInhPa_700_u", "isobaricInhPa_700_v"),
        "500hpa": ("isobaricInhPa_500_u", "isobaricInhPa_500_v"),
    }
    cols = [
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "grid_id",
        "latitude",
        "longitude",
    ]
    cols.extend(column for pair in pair_columns.values() for column in pair)
    frame = pd.read_csv(path, usecols=cols, encoding="utf-8-sig")
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    frame["data_available_kst_dtm"] = pd.to_datetime(frame["data_available_kst_dtm"])
    for feature, (u_col, v_col) in pair_columns.items():
        speed, direction = speed_and_direction(frame[u_col], frame[v_col])
        frame[f"{feature}__speed"] = speed
        frame[f"{feature}__direction"] = direction
    candidates: dict[tuple[str, str, int], pd.DataFrame] = {}
    for grid_id, grid in frame.groupby("grid_id", sort=True):
        grid = grid.sort_values("forecast_kst_dtm").set_index("forecast_kst_dtm")
        for feature in pair_columns:
            candidates[("gfs", feature, int(grid_id))] = grid[
                [f"{feature}__speed", f"{feature}__direction"]
            ].rename(
                columns={
                    f"{feature}__speed": "speed",
                    f"{feature}__direction": "direction",
                }
            )
    grids = (
        frame[["grid_id", "latitude", "longitude"]]
        .drop_duplicates()
        .assign(source="gfs")
    )
    structure = summarize_nwp_structure(frame, "gfs")
    return candidates, grids, structure


def summarize_nwp_structure(frame: pd.DataFrame, source: str) -> dict[str, object]:
    grid_counts = frame.groupby("forecast_kst_dtm")["grid_id"].nunique()
    lead_hours = (
        frame["forecast_kst_dtm"] - frame["data_available_kst_dtm"]
    ).dt.total_seconds() / 3600.0
    return {
        "source": source,
        "rows": len(frame),
        "forecast_hours": int(frame["forecast_kst_dtm"].nunique()),
        "forecast_start": frame["forecast_kst_dtm"].min(),
        "forecast_end": frame["forecast_kst_dtm"].max(),
        "grid_ids": int(frame["grid_id"].nunique()),
        "min_grids_per_hour": int(grid_counts.min()),
        "max_grids_per_hour": int(grid_counts.max()),
        "duplicate_time_grid_keys": int(
            frame.duplicated(["forecast_kst_dtm", "grid_id"]).sum()
        ),
        "min_lead_hours": float(lead_hours.min()),
        "max_lead_hours": float(lead_hours.max()),
        "future_available_rows": int(
            (frame["data_available_kst_dtm"] >= frame["forecast_kst_dtm"]).sum()
        ),
    }


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    value = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    return 2.0 * radius * math.asin(math.sqrt(value))


def build_distance_table(metadata: pd.DataFrame, grids: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for turbine in metadata.itertuples(index=False):
        for grid in grids.itertuples(index=False):
            rows.append(
                {
                    "turbine_id": turbine.turbine_id,
                    "source": grid.source,
                    "grid_id": int(grid.grid_id),
                    "distance_km": haversine_km(
                        turbine.turbine_latitude,
                        turbine.turbine_longitude,
                        grid.latitude,
                        grid.longitude,
                    ),
                }
            )
    distances = pd.DataFrame(rows)
    distances["distance_rank"] = distances.groupby(["turbine_id", "source"])[
        "distance_km"
    ].rank(method="dense")
    distances["is_nearest_grid"] = distances["distance_rank"].eq(1.0)
    return distances


def finite_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return np.nan
    return float(np.corrcoef(left, right)[0, 1])


def speed_metrics(scada: pd.Series, nwp: pd.Series) -> dict[str, float | int]:
    joined = pd.concat(
        [scada.rename("scada"), nwp.rename("nwp")], axis=1, join="inner"
    ).dropna()
    y = joined["scada"].to_numpy(float)
    x = joined["nwp"].to_numpy(float)
    error = x - y
    absolute = np.abs(error)
    strong = y >= STRONG_WIND_THRESHOLD_MS
    return {
        "n_overlap": len(joined),
        "l3_root_ms": float(np.mean(absolute**3) ** (1.0 / 3.0)),
        "mae_ms": float(np.mean(absolute)),
        "rmse_ms": float(np.sqrt(np.mean(error**2))),
        "bias_nwp_minus_scada_ms": float(np.mean(error)),
        "correlation": finite_correlation(y, x),
        "scada_mean_ms": float(np.mean(y)),
        "nwp_mean_ms": float(np.mean(x)),
        "scada_p90_ms": float(np.quantile(y, 0.90)),
        "nwp_p90_ms": float(np.quantile(x, 0.90)),
        "strong_wind_count": int(strong.sum()),
        "strong_wind_l3_root_ms": (
            float(np.mean(absolute[strong] ** 3) ** (1.0 / 3.0))
            if strong.any()
            else np.nan
        ),
    }


def direction_metrics(scada: pd.Series, nwp: pd.Series) -> dict[str, float | int]:
    joined = pd.concat(
        [scada.rename("scada"), nwp.rename("nwp")], axis=1, join="inner"
    ).dropna()
    y = np.mod(joined["scada"].to_numpy(float), 360.0)
    x = np.mod(joined["nwp"].to_numpy(float), 360.0)
    signed_error = (x - y + 180.0) % 360.0 - 180.0
    offset = float(
        np.rad2deg(
            np.arctan2(
                np.mean(np.sin(np.deg2rad(-signed_error))),
                np.mean(np.cos(np.deg2rad(-signed_error))),
            )
        )
    )
    corrected = (x + offset) % 360.0
    corrected_error = (corrected - y + 180.0) % 360.0 - 180.0
    return {
        "n_overlap": len(joined),
        "circular_mae_deg": float(np.mean(np.abs(signed_error))),
        "circular_p90_abs_error_deg": float(
            np.quantile(np.abs(signed_error), 0.90)
        ),
        "mean_cosine_agreement": float(np.mean(np.cos(np.deg2rad(signed_error)))),
        "estimated_scada_offset_deg": offset,
        "offset_corrected_circular_mae_deg": float(
            np.mean(np.abs(corrected_error))
        ),
        "offset_corrected_mean_cosine_agreement": float(
            np.mean(np.cos(np.deg2rad(corrected_error)))
        ),
    }


def evaluate_candidates(
    metadata: pd.DataFrame,
    scada_hourly: pd.DataFrame,
    candidates: dict[tuple[str, str, int], pd.DataFrame],
    distances: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    distance_lookup = distances.set_index(["turbine_id", "source", "grid_id"])
    speed_rows: list[dict[str, object]] = []
    direction_rows: list[dict[str, object]] = []
    for turbine in metadata["turbine_id"]:
        scada_ws = scada_hourly[f"{turbine}__ws"]
        scada_wd = scada_hourly[f"{turbine}__wd"]
        for (source, feature, grid_id), candidate in candidates.items():
            distance = distance_lookup.loc[(turbine, source, grid_id)]
            common = {
                "turbine_id": turbine,
                "source": source,
                "feature": feature,
                "grid_id": grid_id,
                "distance_km": float(distance["distance_km"]),
                "distance_rank": int(distance["distance_rank"]),
                "is_nearest_grid": bool(distance["is_nearest_grid"]),
            }
            speed_rows.append(
                {**common, **speed_metrics(scada_ws, candidate["speed"])}
            )
            direction_rows.append(
                {
                    **common,
                    **direction_metrics(scada_wd, candidate["direction"]),
                }
            )
    speed_table = pd.DataFrame(speed_rows)
    direction_table = pd.DataFrame(direction_rows)
    speed_table["overall_rank"] = speed_table.groupby("turbine_id")[
        "l3_root_ms"
    ].rank(method="min")
    speed_table["source_rank"] = speed_table.groupby(["turbine_id", "source"])[
        "l3_root_ms"
    ].rank(method="min")
    direction_table["raw_direction_rank"] = direction_table.groupby("turbine_id")[
        "circular_mae_deg"
    ].rank(method="min")
    direction_table["offset_corrected_direction_rank"] = direction_table.groupby(
        "turbine_id"
    )["offset_corrected_circular_mae_deg"].rank(method="min")
    return speed_table, direction_table


def build_best_table(
    metadata: pd.DataFrame,
    speed_table: pd.DataFrame,
    direction_table: pd.DataFrame,
) -> pd.DataFrame:
    best_speed = (
        speed_table.sort_values(
            ["turbine_id", "l3_root_ms", "mae_ms", "correlation"],
            ascending=[True, True, True, False],
        )
        .groupby("turbine_id", as_index=False)
        .first()
    )
    nearest_speed = (
        speed_table[speed_table["is_nearest_grid"]]
        .sort_values(["turbine_id", "l3_root_ms", "mae_ms"])
        .groupby("turbine_id", as_index=False)
        .first()
    )
    best_direction = (
        direction_table.sort_values(
            ["turbine_id", "offset_corrected_circular_mae_deg"]
        )
        .groupby("turbine_id", as_index=False)
        .first()
    )
    speed_cols = [
        "turbine_id",
        "source",
        "feature",
        "grid_id",
        "distance_km",
        "distance_rank",
        "is_nearest_grid",
        "n_overlap",
        "l3_root_ms",
        "mae_ms",
        "bias_nwp_minus_scada_ms",
        "correlation",
        "strong_wind_l3_root_ms",
    ]
    nearest_cols = [
        "turbine_id",
        "source",
        "feature",
        "grid_id",
        "l3_root_ms",
    ]
    direction_cols = [
        "turbine_id",
        "source",
        "feature",
        "grid_id",
        "circular_mae_deg",
        "estimated_scada_offset_deg",
        "offset_corrected_circular_mae_deg",
    ]
    best_speed = best_speed[speed_cols].rename(
        columns={column: f"best_speed_{column}" for column in speed_cols if column != "turbine_id"}
    )
    nearest_speed = nearest_speed[nearest_cols].rename(
        columns={
            column: f"best_nearest_{column}"
            for column in nearest_cols
            if column != "turbine_id"
        }
    )
    best_direction = best_direction[direction_cols].rename(
        columns={
            column: f"best_direction_{column}"
            for column in direction_cols
            if column != "turbine_id"
        }
    )
    result = (
        metadata.merge(best_speed, on="turbine_id", how="left")
        .merge(nearest_speed, on="turbine_id", how="left")
        .merge(best_direction, on="turbine_id", how="left")
    )
    for source in ["ldaps", "gfs"]:
        source_best = (
            speed_table[speed_table["source"] == source]
            .sort_values(["turbine_id", "l3_root_ms", "mae_ms"])
            .groupby("turbine_id", as_index=False)
            .first()[speed_cols]
            .rename(
                columns={
                    column: f"best_{source}_{column}"
                    for column in speed_cols
                    if column != "turbine_id"
                }
            )
        )
        result = result.merge(source_best, on="turbine_id", how="left")
    return result


def lag_diagnostics(
    best: pd.DataFrame,
    scada_hourly: pd.DataFrame,
    candidates: dict[tuple[str, str, int], pd.DataFrame],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for item in best.itertuples(index=False):
        key = (
            item.best_speed_source,
            item.best_speed_feature,
            int(item.best_speed_grid_id),
        )
        nwp = candidates[key]["speed"]
        scada = scada_hourly[f"{item.turbine_id}__ws"]
        for lag in range(-2, 3):
            shifted = nwp.copy()
            shifted.index = shifted.index + pd.Timedelta(hours=lag)
            metric = speed_metrics(scada, shifted)
            rows.append(
                {
                    "turbine_id": item.turbine_id,
                    "source": key[0],
                    "feature": key[1],
                    "grid_id": key[2],
                    "nwp_timestamp_shift_hours": lag,
                    **metric,
                }
            )
    table = pd.DataFrame(rows)
    table["lag_rank"] = table.groupby("turbine_id")["l3_root_ms"].rank(
        method="min"
    )
    return table


def combine_metric_tables(
    speed_table: pd.DataFrame, direction_table: pd.DataFrame
) -> pd.DataFrame:
    keys = [
        "turbine_id",
        "source",
        "feature",
        "grid_id",
        "distance_km",
        "distance_rank",
        "is_nearest_grid",
    ]
    return speed_table.merge(
        direction_table,
        on=keys,
        how="inner",
        validate="one_to_one",
        suffixes=("_speed", "_direction"),
    )


def attach_lag_summary(best: pd.DataFrame, lag_table: pd.DataFrame) -> pd.DataFrame:
    lag_zero = lag_table[lag_table["nwp_timestamp_shift_hours"] == 0][
        ["turbine_id", "l3_root_ms"]
    ].rename(columns={"l3_root_ms": "lag0_l3_root_ms"})
    lag_best = (
        lag_table.sort_values(["turbine_id", "l3_root_ms"])
        .groupby("turbine_id", as_index=False)
        .first()[
            [
                "turbine_id",
                "nwp_timestamp_shift_hours",
                "l3_root_ms",
            ]
        ]
        .rename(
            columns={
                "nwp_timestamp_shift_hours": "best_lag_hours",
                "l3_root_ms": "best_lag_l3_root_ms",
            }
        )
        .merge(lag_zero, on="turbine_id", how="left", validate="one_to_one")
    )
    lag_best["lag_l3_gain_ms"] = (
        lag_best["lag0_l3_root_ms"] - lag_best["best_lag_l3_root_ms"]
    )
    return best.merge(lag_best, on="turbine_id", how="left", validate="one_to_one")


def cleanup_superseded_outputs(output_dir: Path) -> None:
    superseded = [
        "turbine_metadata.csv",
        "scada_wind_sensor_summary.csv",
        "nwp_structure_summary.csv",
        "nearest_grid_by_turbine.csv",
        "wind_speed_metrics.csv",
        "wind_direction_metrics.csv",
        "wind_speed_lag_diagnostics.csv",
        "wind_alignment_summary.md",
    ]
    for name in superseded:
        path = output_dir / name
        if path.exists():
            path.unlink()
    charts = output_dir / "charts"
    if charts.exists():
        shutil.rmtree(charts)


def color_mix(start: tuple[int, int, int], end: tuple[int, int, int], value: float) -> str:
    value = min(1.0, max(0.0, value))
    rgb = tuple(round(left + (right - left) * value) for left, right in zip(start, end))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def render_metric_heatmap(
    speed_table: pd.DataFrame, metadata: pd.DataFrame, path: Path
) -> None:
    feature_order = [
        ("ldaps", "5m_boundary_layer"),
        ("ldaps", "10m"),
        ("ldaps", "50m_midpoint"),
        ("ldaps", "50m_component_extreme_proxy"),
        ("gfs", "10m"),
        ("gfs", "80m"),
        ("gfs", "100m"),
        ("gfs", "pbl"),
        ("gfs", "850hpa"),
        ("gfs", "700hpa"),
        ("gfs", "500hpa"),
    ]
    labels = [
        "LDAPS\n5m BL",
        "LDAPS\n10m",
        "LDAPS\n50m midpoint",
        "LDAPS\n50m extreme",
        "GFS\n10m",
        "GFS\n80m",
        "GFS\n100m",
        "GFS\nPBL",
        "GFS\n850hPa",
        "GFS\n700hPa",
        "GFS\n500hPa",
    ]
    best_feature = (
        speed_table.groupby(["turbine_id", "source", "feature"], as_index=False)[
            "l3_root_ms"
        ].min()
    )
    lookup = best_feature.set_index(["turbine_id", "source", "feature"])[
        "l3_root_ms"
    ]
    turbines = metadata["turbine_id"].tolist()
    values = np.array(
        [
            [float(lookup.loc[(turbine, source, feature)]) for source, feature in feature_order]
            for turbine in turbines
        ]
    )
    lower = float(np.quantile(values, 0.05))
    upper = float(np.quantile(values, 0.90))
    left, top = 170, 116
    cell_w, cell_h = 93, 31
    width = left + cell_w * len(feature_order) + 30
    height = top + cell_h * len(turbines) + 55
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#202124">NWP proxy별 최적 격자의 세제곱 풍속 오차</text>',
        f'<text x="24" y="58" font-family="Arial" font-size="13" fill="#5f6368">각 셀은 모든 격자 중 최솟값입니다. 파란색이 낮고 붉은색이 높습니다. 색상 범위: {lower:.2f}-{upper:.2f} m/s.</text>',
    ]
    for column, label in enumerate(labels):
        source, feature = label.split("\n", 1)
        x = left + column * cell_w + cell_w / 2
        elements.extend(
            [
                f'<text x="{x}" y="82" text-anchor="middle" font-family="Arial" font-size="11" font-weight="700" fill="#3c4043">{escape(source)}</text>',
                f'<text x="{x}" y="100" text-anchor="middle" font-family="Arial" font-size="10" fill="#5f6368">{escape(feature)}</text>',
            ]
        )
    for row, turbine in enumerate(turbines):
        y = top + row * cell_h
        elements.append(
            f'<text x="{left - 10}" y="{y + 20}" text-anchor="end" font-family="Arial" font-size="12" fill="#3c4043">{escape(turbine)}</text>'
        )
        for column, value in enumerate(values[row]):
            normalized = (value - lower) / max(upper - lower, 1e-9)
            fill = color_mix((220, 237, 255), (211, 65, 75), normalized)
            x = left + column * cell_w
            text_fill = "#ffffff" if normalized > 0.67 else "#202124"
            elements.extend(
                [
                    f'<rect x="{x + 1}" y="{y + 1}" width="{cell_w - 2}" height="{cell_h - 2}" fill="{fill}"/>',
                    f'<text x="{x + cell_w / 2}" y="{y + 20}" text-anchor="middle" font-family="Arial" font-size="11" fill="{text_fill}">{value:.2f}</text>',
                ]
            )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def render_source_comparison(
    speed_table: pd.DataFrame, metadata: pd.DataFrame, path: Path
) -> None:
    source_best = (
        speed_table.sort_values(["turbine_id", "source", "l3_root_ms"])
        .groupby(["turbine_id", "source"], as_index=False)
        .first()
    )
    pivot = source_best.pivot(index="turbine_id", columns="source", values="l3_root_ms")
    turbines = metadata["turbine_id"].tolist()
    maximum = float(pivot.loc[turbines, ["ldaps", "gfs"]].to_numpy().max())
    width, left, right, top, row_h = 960, 180, 75, 78, 35
    plot_w = width - left - right
    height = top + row_h * len(turbines) + 60
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#202124">터빈별 최적 LDAPS와 최적 GFS 비교</text>',
        '<rect x="630" y="22" width="14" height="14" fill="#2f80ed"/><text x="651" y="34" font-family="Arial" font-size="12" fill="#3c4043">LDAPS</text>',
        '<rect x="714" y="22" width="14" height="14" fill="#f2994a"/><text x="735" y="34" font-family="Arial" font-size="12" fill="#3c4043">GFS</text>',
    ]
    for row, turbine in enumerate(turbines):
        y = top + row * row_h
        ldaps = float(pivot.loc[turbine, "ldaps"])
        gfs = float(pivot.loc[turbine, "gfs"])
        ldaps_w = ldaps / maximum * plot_w
        gfs_w = gfs / maximum * plot_w
        elements.extend(
            [
                f'<text x="{left - 10}" y="{y + 17}" text-anchor="end" font-family="Arial" font-size="12" fill="#3c4043">{escape(turbine)}</text>',
                f'<rect x="{left}" y="{y + 2}" width="{ldaps_w:.2f}" height="11" fill="#2f80ed"/>',
                f'<rect x="{left}" y="{y + 17}" width="{gfs_w:.2f}" height="11" fill="#f2994a"/>',
                f'<text x="{left + ldaps_w + 6:.2f}" y="{y + 12}" font-family="Arial" font-size="10" fill="#2f80ed">{ldaps:.2f}</text>',
                f'<text x="{left + gfs_w + 6:.2f}" y="{y + 27}" font-family="Arial" font-size="10" fill="#a95510">{gfs:.2f}</text>',
            ]
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def render_spatial_map(
    metadata: pd.DataFrame, grids: pd.DataFrame, best: pd.DataFrame, path: Path
) -> None:
    width, height = 980, 700
    left, right, top, bottom = 75, 40, 70, 65
    all_lon = pd.concat([metadata["turbine_longitude"], grids["longitude"]])
    all_lat = pd.concat([metadata["turbine_latitude"], grids["latitude"]])
    lon_min, lon_max = float(all_lon.min()), float(all_lon.max())
    lat_min, lat_max = float(all_lat.min()), float(all_lat.max())

    def xy(lat: float, lon: float) -> tuple[float, float]:
        x = left + (lon - lon_min) / (lon_max - lon_min) * (width - left - right)
        y = top + (lat_max - lat) / (lat_max - lat_min) * (height - top - bottom)
        return x, y

    grid_lookup = grids.set_index(["source", "grid_id"])
    group_colors = {"kpx_group_1": "#2f80ed", "kpx_group_2": "#27ae60", "kpx_group_3": "#f2994a"}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#202124">터빈 위치, NWP 격자와 선택된 최적 풍속 격자</text>',
        '<text x="24" y="55" font-family="Arial" font-size="12" fill="#5f6368">선은 각 터빈과 세제곱 오차가 가장 낮은 격자를 연결합니다. 옅은 사각형과 십자 표시는 전체 LDAPS/GFS 격자입니다.</text>',
    ]
    for grid in grids.itertuples(index=False):
        x, y = xy(grid.latitude, grid.longitude)
        if grid.source == "ldaps":
            elements.append(
                f'<rect x="{x - 4:.2f}" y="{y - 4:.2f}" width="8" height="8" fill="#dbeafe" stroke="#2f80ed" stroke-width="1"/>'
            )
        else:
            elements.extend(
                [
                    f'<line x1="{x - 4:.2f}" y1="{y - 4:.2f}" x2="{x + 4:.2f}" y2="{y + 4:.2f}" stroke="#f2994a" stroke-width="1.5"/>',
                    f'<line x1="{x - 4:.2f}" y1="{y + 4:.2f}" x2="{x + 4:.2f}" y2="{y - 4:.2f}" stroke="#f2994a" stroke-width="1.5"/>',
                ]
            )
    best_lookup = best.set_index("turbine_id")
    for turbine in metadata.itertuples(index=False):
        selected = best_lookup.loc[turbine.turbine_id]
        grid = grid_lookup.loc[(selected.best_speed_source, int(selected.best_speed_grid_id))]
        tx, ty = xy(turbine.turbine_latitude, turbine.turbine_longitude)
        gx, gy = xy(float(grid.latitude), float(grid.longitude))
        color = group_colors[turbine.group]
        elements.extend(
            [
                f'<line x1="{tx:.2f}" y1="{ty:.2f}" x2="{gx:.2f}" y2="{gy:.2f}" stroke="{color}" stroke-width="1" opacity="0.42"/>',
                f'<circle cx="{tx:.2f}" cy="{ty:.2f}" r="5" fill="{color}" stroke="white" stroke-width="1.5"/>',
                f'<text x="{tx + 7:.2f}" y="{ty - 6:.2f}" font-family="Arial" font-size="10" fill="#202124">{escape(turbine.turbine_id.replace("vestas_wtg", "V").replace("unison_wtg", "U"))}</text>',
            ]
        )
    elements.extend(
        [
            '<rect x="720" y="610" width="10" height="10" fill="#dbeafe" stroke="#2f80ed"/><text x="738" y="619" font-family="Arial" font-size="11">LDAPS 격자</text>',
            '<line x1="720" y1="637" x2="730" y2="647" stroke="#f2994a"/><line x1="720" y1="647" x2="730" y2="637" stroke="#f2994a"/><text x="738" y="646" font-family="Arial" font-size="11">GFS 격자</text>',
        ]
    )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def svg_polyline(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x0: float,
    y0: float,
    width: float,
    height: float,
    y_max: float,
    color: str,
) -> str:
    if len(x_values) == 0:
        return ""
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    points = []
    for x_value, y_value in zip(x_values, y_values):
        x = x0 + (x_value - x_min) / max(x_max - x_min, 1e-9) * width
        y = y0 + height - y_value / max(y_max, 1e-9) * height
        points.append(f"{x:.2f},{y:.2f}")
    return f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>'


def render_turbine_panel(
    turbine_id: str,
    scada_hourly: pd.DataFrame,
    candidates: dict[tuple[str, str, int], pd.DataFrame],
    speed_table: pd.DataFrame,
    path: Path,
) -> None:
    turbine_metrics = speed_table[speed_table["turbine_id"] == turbine_id]
    source_best = (
        turbine_metrics.sort_values(["source", "l3_root_ms"])
        .groupby("source", as_index=False)
        .first()
        .set_index("source")
    )
    ldaps_row = source_best.loc["ldaps"]
    gfs_row = source_best.loc["gfs"]
    ldaps_key = ("ldaps", ldaps_row.feature, int(ldaps_row.grid_id))
    gfs_key = ("gfs", gfs_row.feature, int(gfs_row.grid_id))
    joined = pd.concat(
        [
            scada_hourly[f"{turbine_id}__ws"].rename("scada"),
            candidates[ldaps_key]["speed"].rename("ldaps"),
            candidates[gfs_key]["speed"].rename("gfs"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    climatology = joined.groupby(joined.index.month).mean()
    robust_max = float(np.quantile(joined.to_numpy(), 0.995))
    robust_max = max(robust_max, 1.0)
    sample_step = max(1, len(joined) // 1300)
    sample = joined.iloc[::sample_step]
    bins = np.linspace(0.0, robust_max, 25)
    centers = (bins[:-1] + bins[1:]) / 2.0
    histograms = {
        column: np.histogram(joined[column].clip(0, robust_max), bins=bins, density=True)[0]
        for column in joined.columns
    }

    width, height = 1240, 405
    panel_y, panel_h, panel_w = 105, 245, 345
    panel_x = [65, 447, 829]
    colors = {"scada": "#202124", "ldaps": "#2f80ed", "gfs": "#f2994a"}
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="32" font-family="Arial" font-size="22" font-weight="700" fill="#202124">{escape(turbine_id)}</text>',
        f'<text x="24" y="56" font-family="Arial" font-size="12" fill="#5f6368">LDAPS {escape(str(ldaps_row.feature))} 격자 {int(ldaps_row.grid_id)}: L3 {ldaps_row.l3_root_ms:.2f}, 상관 {ldaps_row.correlation:.2f} | GFS {escape(str(gfs_row.feature))} 격자 {int(gfs_row.grid_id)}: L3 {gfs_row.l3_root_ms:.2f}, 상관 {gfs_row.correlation:.2f}</text>',
        '<line x1="850" y1="30" x2="875" y2="30" stroke="#202124" stroke-width="3"/><text x="882" y="34" font-family="Arial" font-size="11">SCADA</text>',
        '<line x1="950" y1="30" x2="975" y2="30" stroke="#2f80ed" stroke-width="3"/><text x="982" y="34" font-family="Arial" font-size="11">LDAPS</text>',
        '<line x1="1050" y1="30" x2="1075" y2="30" stroke="#f2994a" stroke-width="3"/><text x="1082" y="34" font-family="Arial" font-size="11">GFS</text>',
    ]
    titles = ["월별 평균 풍속", "시간별 SCADA-NWP 관계", "풍속 분포"]
    for x, title in zip(panel_x, titles):
        elements.extend(
            [
                f'<text x="{x}" y="91" font-family="Arial" font-size="14" font-weight="700" fill="#3c4043">{escape(title)}</text>',
                f'<rect x="{x}" y="{panel_y}" width="{panel_w}" height="{panel_h}" fill="#fafafa" stroke="#dadce0"/>',
            ]
        )

    months = climatology.index.to_numpy(float)
    monthly_max = float(climatology.to_numpy().max()) * 1.08
    for column in ["scada", "ldaps", "gfs"]:
        elements.append(
            svg_polyline(
                months,
                climatology[column].to_numpy(float),
                panel_x[0] + 34,
                panel_y + 15,
                panel_w - 50,
                panel_h - 42,
                monthly_max,
                colors[column],
            )
        )
    for month in [1, 4, 7, 10, 12]:
        x = panel_x[0] + 34 + (month - 1) / 11 * (panel_w - 50)
        elements.append(
            f'<text x="{x:.2f}" y="{panel_y + panel_h - 8}" text-anchor="middle" font-family="Arial" font-size="10" fill="#5f6368">{month}</text>'
        )

    scatter_x, scatter_y, scatter_w, scatter_h = panel_x[1] + 34, panel_y + 15, panel_w - 52, panel_h - 45
    identity = f'{scatter_x:.2f},{scatter_y + scatter_h:.2f} {scatter_x + scatter_w:.2f},{scatter_y:.2f}'
    elements.append(
        f'<polyline points="{identity}" fill="none" stroke="#9aa0a6" stroke-width="1" stroke-dasharray="4 4"/>'
    )
    for column in ["ldaps", "gfs"]:
        for scada_value, nwp_value in zip(sample["scada"], sample[column]):
            x = scatter_x + min(scada_value, robust_max) / robust_max * scatter_w
            y = scatter_y + scatter_h - min(nwp_value, robust_max) / robust_max * scatter_h
            elements.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.4" fill="{colors[column]}" opacity="0.24"/>'
            )
    elements.extend(
        [
            f'<text x="{scatter_x + scatter_w / 2:.2f}" y="{panel_y + panel_h - 7}" text-anchor="middle" font-family="Arial" font-size="10" fill="#5f6368">SCADA 풍속 (m/s)</text>',
            f'<text x="{scatter_x - 22:.2f}" y="{scatter_y + scatter_h / 2:.2f}" text-anchor="middle" transform="rotate(-90 {scatter_x - 22:.2f} {scatter_y + scatter_h / 2:.2f})" font-family="Arial" font-size="10" fill="#5f6368">NWP 풍속 (m/s)</text>',
        ]
    )

    hist_max = max(float(values.max()) for values in histograms.values()) * 1.05
    for column in ["scada", "ldaps", "gfs"]:
        elements.append(
            svg_polyline(
                centers,
                histograms[column],
                panel_x[2] + 34,
                panel_y + 15,
                panel_w - 50,
                panel_h - 42,
                hist_max,
                colors[column],
            )
        )
    elements.append(
        f'<text x="{panel_x[2] + panel_w / 2:.2f}" y="{panel_y + panel_h - 7}" text-anchor="middle" font-family="Arial" font-size="10" fill="#5f6368">풍속 (m/s)</text>'
    )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def render_direction_summary(
    direction_table: pd.DataFrame, metadata: pd.DataFrame, path: Path
) -> None:
    raw = (
        direction_table.sort_values(["turbine_id", "circular_mae_deg"])
        .groupby("turbine_id", as_index=False)
        .first()
        .set_index("turbine_id")
    )
    corrected = (
        direction_table.sort_values(
            ["turbine_id", "offset_corrected_circular_mae_deg"]
        )
        .groupby("turbine_id", as_index=False)
        .first()
        .set_index("turbine_id")
    )
    turbines = metadata["turbine_id"].tolist()
    width, left, top, row_h = 980, 185, 78, 32
    plot_w = 650
    maximum = max(float(raw["circular_mae_deg"].max()), 50.0)
    height = top + len(turbines) * row_h + 55
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700" fill="#202124">터빈별 풍향 일치도</text>',
        '<rect x="650" y="23" width="14" height="14" fill="#d9485f"/><text x="671" y="35" font-family="Arial" font-size="11">원시 최적</text>',
        '<rect x="750" y="23" width="14" height="14" fill="#27ae60"/><text x="771" y="35" font-family="Arial" font-size="11">offset 보정 최적</text>',
    ]
    for row, turbine in enumerate(turbines):
        y = top + row * row_h
        raw_value = float(raw.loc[turbine, "circular_mae_deg"])
        corrected_value = float(corrected.loc[turbine, "offset_corrected_circular_mae_deg"])
        raw_w = raw_value / maximum * plot_w
        corrected_w = corrected_value / maximum * plot_w
        elements.extend(
            [
                f'<text x="{left - 10}" y="{y + 17}" text-anchor="end" font-family="Arial" font-size="12" fill="#3c4043">{escape(turbine)}</text>',
                f'<rect x="{left}" y="{y + 2}" width="{raw_w:.2f}" height="10" fill="#d9485f"/>',
                f'<rect x="{left}" y="{y + 16}" width="{corrected_w:.2f}" height="10" fill="#27ae60"/>',
                f'<text x="{left + raw_w + 5:.2f}" y="{y + 11}" font-family="Arial" font-size="10" fill="#9d2438">{raw_value:.1f}</text>',
                f'<text x="{left + corrected_w + 5:.2f}" y="{y + 25}" font-family="Arial" font-size="10" fill="#18723d">{corrected_value:.1f}</text>',
            ]
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def write_dashboard(
    output_dir: Path,
    metadata: pd.DataFrame,
    best: pd.DataFrame,
    speed_table: pd.DataFrame,
    structures: pd.DataFrame,
    scada_summary: pd.DataFrame,
    svg_assets: dict[str, str],
) -> None:
    source_best = (
        speed_table.sort_values(["turbine_id", "source", "l3_root_ms"])
        .groupby(["turbine_id", "source"], as_index=False)
        .first()
    )
    mean_by_source = source_best.groupby("source")["l3_root_ms"].mean()
    mean_corr = float(best["best_speed_correlation"].mean())
    nearest_count = int(best["best_speed_is_nearest_grid"].sum())
    mean_advantage = float(mean_by_source["gfs"] - mean_by_source["ldaps"])
    turbine_images = "\n".join(
        f'<section id="turbine-{escape(turbine)}">{svg_assets[turbine]}</section>'
        for turbine in metadata["turbine_id"]
    )
    structure_html = structures.rename(
        columns={
            "source": "자료원",
            "rows": "행 수",
            "forecast_hours": "예보 시각 수",
            "forecast_start": "시작 시각",
            "forecast_end": "종료 시각",
            "grid_ids": "격자 수",
            "min_grids_per_hour": "시간당 최소 격자",
            "max_grids_per_hour": "시간당 최대 격자",
            "duplicate_time_grid_keys": "중복 키",
            "min_lead_hours": "최소 lead(h)",
            "max_lead_hours": "최대 lead(h)",
            "future_available_rows": "가용시각 위반",
        }
    ).to_html(index=False, border=0, float_format=lambda x: f"{x:.2f}")
    sensor_columns = [
        "turbine_id",
        "ws_missing",
        "wd_missing",
        "ws_p50",
        "ws_p99",
        "ws_max",
        "valid_hourly_ws",
    ]
    sensor_html = scada_summary[sensor_columns].rename(
        columns={
            "turbine_id": "터빈",
            "ws_missing": "풍속 결측",
            "wd_missing": "풍향 결측",
            "ws_p50": "풍속 P50",
            "ws_p99": "풍속 P99",
            "ws_max": "최대 풍속",
            "valid_hourly_ws": "유효 시간",
        }
    ).to_html(index=False, border=0, float_format=lambda x: f"{x:.2f}")
    selection_columns = [
        "turbine_id",
        "best_speed_source",
        "best_speed_feature",
        "best_speed_grid_id",
        "best_speed_l3_root_ms",
        "best_speed_correlation",
        "best_lag_hours",
        "lag_l3_gain_ms",
    ]
    selection_html = best[selection_columns].rename(
        columns={
            "turbine_id": "터빈",
            "best_speed_source": "최적 자료원",
            "best_speed_feature": "최적 feature",
            "best_speed_grid_id": "최적 격자",
            "best_speed_l3_root_ms": "L3 풍속 오차",
            "best_speed_correlation": "상관계수",
            "best_lag_hours": "진단 최적 lag(h)",
            "lag_l3_gain_ms": "lag 개선량",
        }
    ).to_html(index=False, border=0, float_format=lambda x: f"{x:.3f}")
    html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NWP-SCADA 바람 정합성 EDA</title>
<style>
body {{ margin: 0; font-family: Arial, sans-serif; color: #202124; background: #f5f6f7; }}
header {{ background: #ffffff; border-bottom: 1px solid #dadce0; padding: 24px 32px; }}
h1 {{ font-size: 28px; margin: 0 0 8px; }}
p {{ margin: 0; color: #5f6368; line-height: 1.5; }}
main {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
.metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1px; background: #dadce0; border: 1px solid #dadce0; margin-bottom: 24px; }}
.metric {{ background: white; padding: 18px; min-height: 72px; }}
.metric strong {{ display: block; font-size: 24px; margin-top: 8px; }}
.metric span {{ color: #5f6368; font-size: 12px; }}
section {{ background: white; border-bottom: 1px solid #dadce0; margin-bottom: 20px; overflow-x: auto; }}
section svg {{ display: block; width: 100%; height: auto; min-width: 820px; }}
h2 {{ font-size: 20px; margin: 32px 0 12px; }}
.method {{ background: white; border-left: 4px solid #2f80ed; padding: 18px 22px; margin-bottom: 22px; line-height: 1.55; }}
.method code {{ background: #eef2f6; padding: 2px 5px; }}
.table-wrap {{ background: white; overflow-x: auto; margin-bottom: 22px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
th, td {{ border-bottom: 1px solid #e5e7eb; padding: 8px 10px; text-align: right; white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f8f9fa; color: #3c4043; }}
@media (max-width: 760px) {{ .metrics {{ grid-template-columns: 1fr 1fr; }} main {{ padding: 12px; }} }}
</style>
</head>
<body>
<header><h1>NWP-SCADA 바람 정합성 EDA</h1><p>모든 LDAPS/GFS 바람 proxy와 격자를 터빈별 시간 평균 SCADA와 원시 값으로 비교했습니다. 주 지표는 세제곱 절대오차의 세제곱근입니다.</p></header>
<main>
<div class="metrics">
<div class="metric"><span>최적 LDAPS 평균 L3</span><strong>{mean_by_source['ldaps']:.2f} m/s</strong></div>
<div class="metric"><span>최적 GFS 평균 L3</span><strong>{mean_by_source['gfs']:.2f} m/s</strong></div>
<div class="metric"><span>LDAPS 평균 우위</span><strong>{mean_advantage:.2f} m/s</strong></div>
<div class="metric"><span>선택 proxy 평균 상관 / 최근접 일치</span><strong>{mean_corr:.2f} / {nearest_count}/17</strong></div>
</div>
<div class="method"><strong>비교 기준</strong><br>SCADA는 각 시각에 끝나는 1시간 구간에서 유효한 10분 관측이 {SCADA_MIN_SAMPLES_PER_HOUR}개 이상일 때 평균했습니다. 모든 NWP proxy는 보정 없이 같은 예보 대상 시각에 비교했습니다. 풍속 순위는 <code>mean(abs(SCADA-NWP)^3)^(1/3)</code>, 풍향은 원형 오차를 사용했습니다. offset과 -2~+2시간 lag는 시간 정합성 진단일 뿐 모델 입력 규칙이 아닙니다.</div>
<h2>데이터 무결성</h2>
<div class="table-wrap">{structure_html}</div>
<h2>전체 터빈 비교</h2>
<section>{svg_assets['heatmap']}</section>
<section>{svg_assets['source_comparison']}</section>
<section>{svg_assets['spatial_map']}</section>
<section>{svg_assets['direction']}</section>
<h2>터빈별 canonical 선택 결과</h2>
<div class="table-wrap">{selection_html}</div>
<h2>SCADA 센서 가용성</h2>
<div class="table-wrap">{sensor_html}</div>
<h2>터빈별 진단</h2>
{turbine_images}
</main>
</body>
</html>
"""
    (output_dir / "wind_eda_dashboard.html").write_text(html, encoding="utf-8")


def validate_outputs(
    metadata: pd.DataFrame,
    speed_table: pd.DataFrame,
    best: pd.DataFrame,
    structures: pd.DataFrame,
) -> None:
    expected_candidates = 16 * 4 + 9 * 7
    expected_rows = len(metadata) * expected_candidates
    if len(speed_table) != expected_rows:
        raise ValueError(
            f"Unexpected speed metric rows: {len(speed_table)} != {expected_rows}"
        )
    if best["turbine_id"].nunique() != len(metadata):
        raise ValueError("Best table does not contain exactly one row per turbine")
    if speed_table["n_overlap"].min() < 1000:
        raise ValueError("A candidate has insufficient SCADA-NWP overlap")
    if not structures["duplicate_time_grid_keys"].eq(0).all():
        raise ValueError("Duplicate NWP forecast-time/grid keys detected")
    if not structures["future_available_rows"].eq(0).all():
        raise ValueError("NWP availability is not strictly before forecast time")
    recomputed = (
        speed_table.groupby("turbine_id")["l3_root_ms"].min().sort_index()
    )
    selected = best.set_index("turbine_id")["best_speed_l3_root_ms"].sort_index()
    if not np.allclose(recomputed.to_numpy(), selected.to_numpy(), rtol=0, atol=1e-12):
        raise ValueError("Best table does not reproduce detailed speed minima")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(args.data_dir / "info.xlsx")
    vestas_hourly, vestas_summary = load_scada_hourly(
        args.data_dir / "train" / "scada_vestas_train.csv", "vestas", 12
    )
    unison_hourly, unison_summary = load_scada_hourly(
        args.data_dir / "train" / "scada_unison_train.csv", "unison", 5
    )
    scada_hourly = vestas_hourly.join(unison_hourly, how="outer").sort_index()
    scada_summary = pd.concat([vestas_summary, unison_summary], ignore_index=True)

    ldaps_candidates, ldaps_grids, ldaps_structure = load_ldaps_candidates(
        args.data_dir / "train" / "ldaps_train.csv"
    )
    gfs_candidates, gfs_grids, gfs_structure = load_gfs_candidates(
        args.data_dir / "train" / "gfs_train.csv"
    )
    candidates = {**ldaps_candidates, **gfs_candidates}
    grids = pd.concat([ldaps_grids, gfs_grids], ignore_index=True)
    structures = pd.DataFrame([ldaps_structure, gfs_structure])
    distances = build_distance_table(metadata, grids)

    speed_table, direction_table = evaluate_candidates(
        metadata, scada_hourly, candidates, distances
    )
    best = build_best_table(metadata, speed_table, direction_table)
    lag_table = lag_diagnostics(best, scada_hourly, candidates)
    best = attach_lag_summary(best, lag_table)
    metrics = combine_metric_tables(speed_table, direction_table)

    validate_outputs(metadata, speed_table, best, structures)
    cleanup_superseded_outputs(args.output_dir)
    write_csv(metrics, args.output_dir / "wind_proxy_metrics.csv")
    write_csv(best, args.output_dir / "best_wind_proxy_by_turbine.csv")

    with tempfile.TemporaryDirectory(prefix="wind_eda_") as temp_name:
        temp_dir = Path(temp_name)
        overview_paths = {
            "heatmap": temp_dir / "heatmap.svg",
            "source_comparison": temp_dir / "source_comparison.svg",
            "spatial_map": temp_dir / "spatial_map.svg",
            "direction": temp_dir / "direction.svg",
        }
        render_metric_heatmap(speed_table, metadata, overview_paths["heatmap"])
        render_source_comparison(
            speed_table, metadata, overview_paths["source_comparison"]
        )
        render_spatial_map(metadata, grids, best, overview_paths["spatial_map"])
        render_direction_summary(
            direction_table, metadata, overview_paths["direction"]
        )
        svg_assets = {
            name: path.read_text(encoding="utf-8")
            for name, path in overview_paths.items()
        }
        for turbine_id in metadata["turbine_id"]:
            turbine_path = temp_dir / f"{turbine_id}.svg"
            render_turbine_panel(
                turbine_id,
                scada_hourly,
                candidates,
                speed_table,
                turbine_path,
            )
            svg_assets[turbine_id] = turbine_path.read_text(encoding="utf-8")
        write_dashboard(
            args.output_dir,
            metadata,
            best,
            speed_table,
            structures,
            scada_summary,
            svg_assets,
        )

    print(f"Saved EDA outputs to {args.output_dir}")
    print(
        best[
            [
                "turbine_id",
                "best_speed_source",
                "best_speed_feature",
                "best_speed_grid_id",
                "best_speed_l3_root_ms",
                "best_speed_correlation",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

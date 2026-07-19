from __future__ import annotations

import argparse
import html
import math
import re
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor


TARGET_GROUPS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
GROUP_CAPACITY_KWH = {
    "kpx_group_1": 21_600.0,
    "kpx_group_2": 21_600.0,
    "kpx_group_3": 21_000.0,
}
FEATURE_NAMES = [
    "wind_u",
    "wind_v",
    "wind_speed",
    "lead_hour",
    "hour_sin",
    "hour_cos",
    "doy_sin",
    "doy_cos",
]
SCADA_MIN_WIND_SAMPLES = 4
POWER_UPPER_TOLERANCE = 1.02
SHUTDOWN_WIND_MS = 5.0
SHUTDOWN_POWER_RATIO = 0.01
SHUTDOWN_OTHER_RATIO = 0.10
RANDOM_SEED = 20260719


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="터빈별 LightGBM으로 최근접 10m와 EDA 선택 바람을 OOF 비교합니다."
    )
    parser.add_argument("--data-dir", type=Path, default=root / "data")
    parser.add_argument(
        "--output-dir", type=Path, default=root / "results" / "per_turbine_lgbm"
    )
    parser.add_argument("--max-rounds", type=int, default=1600)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser.parse_args()


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def interval_year(index: pd.DatetimeIndex) -> pd.Index:
    """Map an interval-ending timestamp to the year containing that interval."""
    return (index - pd.Timedelta(nanoseconds=1)).year


def dms_to_decimal(value: object) -> tuple[float, float]:
    match = re.search(
        r"(\d+)[^\d]+(\d+)[^\d]+([\d.]+)\"?([NS])\s+"
        r"(\d+)[^\d]+(\d+)[^\d]+([\d.]+)\"?([EW])",
        str(value),
    )
    if match is None:
        return np.nan, np.nan
    lat_d, lat_m, lat_s, lat_h, lon_d, lon_m, lon_s, lon_h = match.groups()
    latitude = float(lat_d) + float(lat_m) / 60.0 + float(lat_s) / 3600.0
    longitude = float(lon_d) + float(lon_m) / 60.0 + float(lon_s) / 3600.0
    if lat_h == "S":
        latitude = -latitude
    if lon_h == "W":
        longitude = -longitude
    return latitude, longitude


def load_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_excel(path, sheet_name="info", header=3, usecols="B:L")
    metadata.columns = [
        "stage",
        "name",
        "manufacturer",
        "model",
        "unit",
        "coordinates",
        "group_number",
        "hub_height_m",
        "rotor_diameter_m",
        "capacity_mw",
        "group_capacity_mw",
    ]
    metadata = metadata.dropna(subset=["manufacturer", "unit"]).copy()
    metadata["group_number"] = metadata["group_number"].ffill().astype(int)
    metadata["unit"] = metadata["unit"].astype(int)
    metadata["turbine_id"] = [
        f"{str(maker).strip().lower()}_wtg{unit:02d}"
        for maker, unit in zip(metadata["manufacturer"], metadata["unit"])
    ]
    coordinates = metadata["coordinates"].map(dms_to_decimal)
    metadata["latitude"] = [item[0] for item in coordinates]
    metadata["longitude"] = [item[1] for item in coordinates]
    metadata["group"] = metadata["group_number"].map(
        lambda value: f"kpx_group_{value}"
    )
    columns = [
        "turbine_id",
        "group",
        "manufacturer",
        "model",
        "unit",
        "capacity_mw",
        "latitude",
        "longitude",
    ]
    result = metadata[columns].reset_index(drop=True)
    if len(result) != 17 or result["turbine_id"].nunique() != 17:
        raise ValueError("터빈 메타데이터는 중복 없이 17행이어야 합니다.")
    return result


def component_absmax(maximum: pd.Series, minimum: pd.Series) -> np.ndarray:
    return np.where(maximum.abs() >= minimum.abs(), maximum, minimum)


def load_ldaps(path: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    columns = [
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
    frame = pd.read_csv(path, usecols=columns, encoding="utf-8-sig")
    frame["forecast_kst_dtm"] = pd.to_datetime(frame["forecast_kst_dtm"])
    frame["data_available_kst_dtm"] = pd.to_datetime(
        frame["data_available_kst_dtm"]
    )
    frame["grid_id"] = frame["grid_id"].astype(int)
    feature_pairs: dict[str, tuple[object, object]] = {
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
    for feature, (u_value, v_value) in feature_pairs.items():
        frame[f"{feature}__u"] = np.asarray(u_value, dtype=float)
        frame[f"{feature}__v"] = np.asarray(v_value, dtype=float)
        frame[f"{feature}__speed"] = np.hypot(
            frame[f"{feature}__u"], frame[f"{feature}__v"]
        )
    grids = frame[["grid_id", "latitude", "longitude"]].drop_duplicates()
    if len(grids) != 16:
        raise ValueError("LDAPS 격자는 16개여야 합니다.")
    if frame.duplicated(["forecast_kst_dtm", "grid_id"]).any():
        raise ValueError("LDAPS 예보시각-격자 키가 중복되었습니다.")
    indexed = frame.set_index(["forecast_kst_dtm", "grid_id"]).sort_index()
    speed_matrices = {
        feature: indexed[f"{feature}__speed"].unstack("grid_id")
        for feature in feature_pairs
    }
    return indexed, speed_matrices


def load_scada(
    data_dir: Path, metadata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_series: dict[str, pd.Series] = {}
    wind_series: dict[str, pd.Series] = {}
    audit_rows: list[dict[str, object]] = []
    metadata_index = metadata.set_index("turbine_id")
    sources = [
        ("vestas", data_dir / "train" / "scada_vestas_train.csv", 12),
        ("unison", data_dir / "train" / "scada_unison_train.csv", 5),
    ]
    for prefix, path, turbine_count in sources:
        raw = pd.read_csv(path, encoding="utf-8-sig")
        raw["kst_dtm"] = pd.to_datetime(raw["kst_dtm"])
        if raw["kst_dtm"].duplicated().any():
            raise ValueError(f"{path.name}에 중복 시각이 있습니다.")
        hour_end = raw["kst_dtm"].dt.ceil("h")
        for unit in range(1, turbine_count + 1):
            turbine = f"{prefix}_wtg{unit:02d}"
            capacity_kwh = float(metadata_index.loc[turbine, "capacity_mw"]) * 1000.0
            power_column = f"{turbine}_power_kw10m"
            wind_column = f"{turbine}_ws"

            raw_power = pd.to_numeric(raw[power_column], errors="coerce")
            upper = capacity_kwh / 6.0 * POWER_UPPER_TOLERANCE
            invalid_power = raw_power.lt(0.0) | raw_power.gt(upper)
            clean_power = raw_power.mask(invalid_power)
            power_group = clean_power.groupby(hour_end, sort=True)
            hourly_power = power_group.sum(min_count=6)
            hourly_power[power_group.count() < 6] = np.nan
            hourly_power.index.name = "kst_dtm"
            target_series[turbine] = hourly_power

            clean_wind = pd.to_numeric(raw[wind_column], errors="coerce")
            clean_wind = clean_wind.mask(clean_wind.lt(0.0) | clean_wind.gt(50.0))
            wind_group = clean_wind.groupby(hour_end, sort=True)
            hourly_wind = wind_group.mean()
            hourly_wind[wind_group.count() < SCADA_MIN_WIND_SAMPLES] = np.nan
            hourly_wind.index.name = "kst_dtm"
            wind_series[turbine] = hourly_wind

            audit_rows.append(
                {
                    "turbine_id": turbine,
                    "invalid_power_10m_rows": int(invalid_power.sum()),
                    "valid_hourly_target_rows": int(hourly_power.notna().sum()),
                    "valid_hourly_wind_rows": int(hourly_wind.notna().sum()),
                }
            )

    targets = pd.DataFrame(target_series).sort_index()
    winds = pd.DataFrame(wind_series).sort_index()
    shutdown = pd.DataFrame(False, index=targets.index, columns=TARGET_GROUPS)
    for group in TARGET_GROUPS:
        group_meta = metadata[metadata["group"].eq(group)]
        turbines = group_meta["turbine_id"].tolist()
        capacities = group_meta.set_index("turbine_id")["capacity_mw"] * 1000.0
        ratios = targets[turbines].div(capacities, axis=1)
        turbine_shutdown = pd.DataFrame(False, index=targets.index, columns=turbines)
        for turbine in turbines:
            others = [item for item in turbines if item != turbine]
            other_ratio = ratios[others].median(axis=1, skipna=True)
            turbine_shutdown[turbine] = (
                winds[turbine].ge(SHUTDOWN_WIND_MS)
                & ratios[turbine].le(SHUTDOWN_POWER_RATIO)
                & other_ratio.ge(SHUTDOWN_OTHER_RATIO)
            )
        shutdown[group] = turbine_shutdown.any(axis=1)
    audit = pd.DataFrame(audit_rows)
    return targets, winds, shutdown, audit


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


def nearest_grid_mapping(
    metadata: pd.DataFrame, ldaps: pd.DataFrame
) -> dict[str, dict[str, object]]:
    grids = (
        ldaps.reset_index()[["grid_id", "latitude", "longitude"]]
        .drop_duplicates()
        .sort_values("grid_id")
    )
    mapping: dict[str, dict[str, object]] = {}
    for turbine in metadata.itertuples(index=False):
        distances = [
            (
                int(grid.grid_id),
                haversine_km(
                    float(turbine.latitude),
                    float(turbine.longitude),
                    float(grid.latitude),
                    float(grid.longitude),
                ),
            )
            for grid in grids.itertuples(index=False)
        ]
        grid_id, distance = min(distances, key=lambda item: item[1])
        mapping[turbine.turbine_id] = {
            "feature": "10m",
            "grid_id": grid_id,
            "distance_km": distance,
        }
    return mapping


def mapping_l3(
    wind: pd.Series,
    matrix: pd.DataFrame,
    grid_id: int,
    train_years: list[int],
) -> tuple[float, int]:
    common = matrix.index.intersection(wind.index)
    selected = common[interval_year(common).isin(train_years)]
    actual = wind.reindex(selected).to_numpy(float)
    forecast = matrix.loc[selected, grid_id].to_numpy(float)
    valid = np.isfinite(actual) & np.isfinite(forecast)
    if int(valid.sum()) < 1000:
        raise ValueError("NWP-SCADA mapping 평가 표본이 1,000개 미만입니다.")
    l3 = float(np.mean(np.abs(forecast[valid] - actual[valid]) ** 3) ** (1.0 / 3.0))
    return l3, int(valid.sum())


def select_candidate_mapping(
    turbine: str,
    train_years: list[int],
    winds: pd.DataFrame,
    speed_matrices: dict[str, pd.DataFrame],
) -> dict[str, object]:
    rows: list[tuple[float, str, int, int]] = []
    for feature, matrix in speed_matrices.items():
        for grid_id in matrix.columns:
            l3, count = mapping_l3(
                winds[turbine], matrix, int(grid_id), train_years
            )
            rows.append((l3, feature, int(grid_id), count))
    l3, feature, grid_id, count = min(rows, key=lambda item: (item[0], item[1], item[2]))
    return {
        "feature": feature,
        "grid_id": grid_id,
        "mapping_l3_ms": l3,
        "mapping_overlap": count,
    }


def build_features(
    ldaps: pd.DataFrame, feature: str, grid_id: int
) -> pd.DataFrame:
    selected = ldaps.xs(int(grid_id), level="grid_id").copy()
    result = pd.DataFrame(index=selected.index)
    result["wind_u"] = selected[f"{feature}__u"].astype(float)
    result["wind_v"] = selected[f"{feature}__v"].astype(float)
    result["wind_speed"] = selected[f"{feature}__speed"].astype(float)
    result["lead_hour"] = (
        result.index.to_series() - selected["data_available_kst_dtm"]
    ).dt.total_seconds() / 3600.0
    hour_angle = 2.0 * np.pi * result.index.hour / 24.0
    result["hour_sin"] = np.sin(hour_angle)
    result["hour_cos"] = np.cos(hour_angle)
    year_days = np.where(result.index.is_leap_year, 366.0, 365.0)
    day_position = result.index.dayofyear - 1 + result.index.hour / 24.0
    doy_angle = 2.0 * np.pi * day_position / year_days
    result["doy_sin"] = np.sin(doy_angle)
    result["doy_cos"] = np.cos(doy_angle)
    if result[FEATURE_NAMES].isna().any().any():
        raise ValueError(f"LDAPS feature에 결측이 있습니다: {feature}, grid={grid_id}")
    return result[FEATURE_NAMES]


def make_inner_folds(valid_index: pd.DatetimeIndex) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    years_for_index = interval_year(valid_index)
    years = sorted(set(years_for_index.tolist()))
    folds: list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]] = []
    if len(years) >= 2:
        for validation_year in years:
            validation = valid_index[years_for_index == validation_year]
            train = valid_index[years_for_index != validation_year]
            folds.append((train, validation))
    else:
        ordered = valid_index.sort_values()
        for block in np.array_split(np.arange(len(ordered)), 4):
            validation = ordered[block]
            train = ordered[~np.isin(np.arange(len(ordered)), block)]
            folds.append((train, validation))
    if any(len(train) < 1000 or len(validation) < 500 for train, validation in folds):
        raise ValueError("inner validation fold의 학습 또는 검증 표본이 부족합니다.")
    return folds


def model_parameters(n_jobs: int, n_estimators: int) -> dict[str, object]:
    return {
        "objective": "regression_l1",
        "learning_rate": 0.03,
        "n_estimators": int(n_estimators),
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 100,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "random_state": RANDOM_SEED,
        "n_jobs": n_jobs,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }


def estimate_best_iteration(
    features: pd.DataFrame,
    target: pd.Series,
    valid_index: pd.DatetimeIndex,
    args: argparse.Namespace,
) -> tuple[int, list[int]]:
    iterations: list[int] = []
    for train_index, validation_index in make_inner_folds(valid_index):
        model = LGBMRegressor(
            **model_parameters(args.n_jobs, args.max_rounds)
        )
        model.fit(
            features.loc[train_index],
            target.loc[train_index],
            eval_X=features.loc[validation_index],
            eval_y=target.loc[validation_index],
            eval_metric="l1",
            callbacks=[
                lgb.early_stopping(
                    args.early_stopping_rounds, verbose=False
                )
            ],
        )
        best = int(model.best_iteration_ or args.max_rounds)
        iterations.append(max(1, min(best, args.max_rounds)))
    median_iteration = int(np.rint(np.median(iterations)))
    return max(1, median_iteration), iterations


def train_outer_model(
    features: pd.DataFrame,
    target: pd.Series,
    group_shutdown: pd.Series,
    train_years: list[int],
    validation_index: pd.DatetimeIndex,
    capacity_kwh: float,
    args: argparse.Namespace,
) -> tuple[np.ndarray, int, list[int], int]:
    target = target.reindex(features.index)
    shutdown = group_shutdown.reindex(features.index, fill_value=False)
    train_mask = (
        interval_year(features.index).isin(train_years)
        & target.notna().to_numpy()
        & ~shutdown.to_numpy(bool)
    )
    valid_train_index = features.index[train_mask]
    best_iteration, inner_iterations = estimate_best_iteration(
        features, target, valid_train_index, args
    )
    model = LGBMRegressor(
        **model_parameters(args.n_jobs, best_iteration)
    )
    model.fit(features.loc[valid_train_index], target.loc[valid_train_index])
    prediction = model.predict(features.loc[validation_index])
    prediction = np.clip(prediction, 0.0, capacity_kwh)
    return prediction, best_iteration, inner_iterations, len(valid_train_index)


def group_metric(
    actual: np.ndarray, prediction: np.ndarray, capacity: float
) -> dict[str, float | int]:
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(actual) & np.isfinite(prediction) & (actual >= 0.10 * capacity)
    error_rate = np.abs(prediction[valid] - actual[valid]) / capacity
    nmae = float(np.mean(error_rate))
    unit_price = np.select(
        [error_rate <= 0.06, error_rate <= 0.08], [4.0, 3.0], default=0.0
    )
    ficr = float(np.sum(actual[valid] * unit_price) / np.sum(actual[valid] * 4.0))
    return {
        "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
        "nmae": nmae,
        "ficr": ficr,
        "n_rows": int(len(actual)),
        "evaluated_rows": int(valid.sum()),
    }


def build_metrics(oof: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, variant_frame in oof.groupby("variant", sort=True):
        group_results: list[dict[str, float | int]] = []
        for group in TARGET_GROUPS:
            group_frame = variant_frame[variant_frame["group"].eq(group)]
            metric = group_metric(
                group_frame["actual"],
                group_frame["prediction"],
                GROUP_CAPACITY_KWH[group],
            )
            group_results.append(metric)
            rows.append(
                {
                    "scope": "group",
                    "variant": variant,
                    "group": group,
                    "year": np.nan,
                    **metric,
                }
            )
            for year, year_frame in group_frame.groupby("validation_year", sort=True):
                year_metric = group_metric(
                    year_frame["actual"],
                    year_frame["prediction"],
                    GROUP_CAPACITY_KWH[group],
                )
                rows.append(
                    {
                        "scope": "group_year",
                        "variant": variant,
                        "group": group,
                        "year": int(year),
                        **year_metric,
                    }
                )
        mean_nmae = float(np.mean([item["nmae"] for item in group_results]))
        mean_ficr = float(np.mean([item["ficr"] for item in group_results]))
        rows.append(
            {
                "scope": "overall",
                "variant": variant,
                "group": "all",
                "year": np.nan,
                "score": 0.5 * (1.0 - mean_nmae) + 0.5 * mean_ficr,
                "nmae": mean_nmae,
                "ficr": mean_ficr,
                "n_rows": int(sum(item["n_rows"] for item in group_results)),
                "evaluated_rows": int(
                    sum(item["evaluated_rows"] for item in group_results)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["scope", "variant", "group", "year"], na_position="first"
    )


def score_chart(metrics: pd.DataFrame) -> str:
    overall = metrics[metrics["scope"].eq("overall")].set_index("variant")
    colors = {"baseline": "#64748b", "candidate": "#0f766e"}
    labels = {"baseline": "최근접 LDAPS 10m", "candidate": "EDA 선택 바람"}
    width, height = 760, 230
    elements = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="전체 OOF 점수 비교">',
        '<line x1="90" y1="190" x2="730" y2="190" stroke="#cbd5e1"/>',
    ]
    for index, variant in enumerate(["baseline", "candidate"]):
        score = float(overall.loc[variant, "score"])
        x = 180 + index * 310
        bar_height = max(1.0, score * 160.0)
        y = 190.0 - bar_height
        elements.extend(
            [
                f'<rect x="{x}" y="{y:.1f}" width="150" height="{bar_height:.1f}" fill="{colors[variant]}"/>',
                f'<text x="{x + 75}" y="{y - 10:.1f}" text-anchor="middle" font-size="22" font-weight="700">{score:.6f}</text>',
                f'<text x="{x + 75}" y="216" text-anchor="middle" font-size="14">{labels[variant]}</text>',
            ]
        )
    elements.append("</svg>")
    return "".join(elements)


def render_table(frame: pd.DataFrame, columns: list[str], labels: list[str]) -> str:
    header = "".join(f"<th>{html.escape(label)}</th>" for label in labels)
    rows = []
    for item in frame[columns].itertuples(index=False, name=None):
        cells = []
        for value in item:
            if isinstance(value, float):
                text = "" if np.isnan(value) else f"{value:.6f}"
            else:
                text = str(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def write_dashboard(
    path: Path,
    metrics: pd.DataFrame,
    mappings: pd.DataFrame,
    audit: pd.DataFrame,
    shutdown: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    overall = metrics[metrics["scope"].eq("overall")].set_index("variant")
    baseline_score = float(overall.loc["baseline", "score"])
    candidate_score = float(overall.loc["candidate", "score"])
    delta = candidate_score - baseline_score
    group_metrics = metrics[metrics["scope"].eq("group")].copy()
    group_metrics["variant"] = group_metrics["variant"].map(
        {"baseline": "최근접 10m", "candidate": "EDA 선택"}
    )
    year_metrics = metrics[metrics["scope"].eq("group_year")].copy()
    year_metrics["variant"] = year_metrics["variant"].map(
        {"baseline": "최근접 10m", "candidate": "EDA 선택"}
    )
    candidate_mappings = mappings[mappings["variant"].eq("candidate")].copy()
    shutdown_rows = []
    for group in TARGET_GROUPS:
        for year, values in shutdown[group].groupby(interval_year(shutdown.index)):
            if int(year) in (2022, 2023, 2024):
                shutdown_rows.append(
                    {"group": group, "year": int(year), "hours": int(values.sum())}
                )
    shutdown_table = pd.DataFrame(shutdown_rows)
    parameters = model_parameters(args.n_jobs, args.max_rounds)
    parameter_text = ", ".join(
        f"{key}={value}"
        for key, value in parameters.items()
        if key not in {"n_jobs", "verbosity", "force_col_wise", "deterministic"}
    )
    document = f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>터빈별 LightGBM 바람 입력 비교</title>
<style>
body{{margin:0;background:#f8fafc;color:#172033;font-family:Arial,'Malgun Gothic',sans-serif}}
main{{max-width:1180px;margin:auto;padding:32px 28px 64px}}
h1{{font-size:30px;margin:0 0 8px}} h2{{font-size:21px;margin:34px 0 12px}}
p{{line-height:1.65;color:#475569}} .metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:22px 0}}
.metric{{background:white;border:1px solid #dbe3ec;border-radius:6px;padding:16px}} .metric span{{display:block;color:#64748b;font-size:13px}}
.metric strong{{display:block;font-size:25px;margin-top:7px}} .positive{{color:#0f766e}} .negative{{color:#b42318}}
.panel{{background:white;border:1px solid #dbe3ec;border-radius:6px;padding:18px;margin-top:12px;overflow:auto}}
table{{border-collapse:collapse;width:100%;font-size:12px}} th,td{{border-bottom:1px solid #e2e8f0;padding:8px;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} th{{background:#f1f5f9;color:#334155;position:sticky;top:0}}
code{{background:#eef2f6;padding:2px 5px;border-radius:3px}} .method{{border-left:4px solid #0f766e;padding:10px 14px;background:#ecfdf5;color:#334155}}
</style></head><body><main>
<h1>터빈별 LightGBM 바람 입력 비교</h1>
<p>같은 target, 운영정지 mask, fold, 구조 하이퍼파라미터를 사용하고 LDAPS 바람 선택 방식만 바꾼 paired outer OOF 결과입니다.</p>
<div class="metrics">
<div class="metric"><span>최근접 LDAPS 10m</span><strong>{baseline_score:.6f}</strong></div>
<div class="metric"><span>EDA 선택 바람</span><strong>{candidate_score:.6f}</strong></div>
<div class="metric"><span>Candidate - Baseline</span><strong class="{'positive' if delta >= 0 else 'negative'}">{delta:+.6f}</strong></div>
</div>
<div class="panel">{score_chart(metrics)}</div>
<h2>실험 계약</h2>
<div class="method">10분 SCADA power는 터빈 정격 범위 밖 값을 결측 처리한 뒤 종료시각 기준 1시간에 6개를 합산했습니다. 실제 정지 후보는 SCADA 풍속 ≥ {SHUTDOWN_WIND_MS:.1f} m/s, 해당 터빈 발전률 ≤ {SHUTDOWN_POWER_RATIO:.0%}, 다른 가용 터빈 중앙 발전률 ≥ {SHUTDOWN_OTHER_RATIO:.0%}일 때이며 outer-train에서만 제외했습니다. Candidate mapping은 각 outer-train의 SCADA 풍속만 사용해 <code>mean(abs(SCADA-NWP)^3)^(1/3)</code>가 최소인 LDAPS feature/grid를 다시 선택했습니다.</div>
<p>공유 하파: <code>{html.escape(parameter_text)}</code>. 터빈별 epoch는 inner blocked validation의 best iteration 중앙값입니다.</p>
<h2>그룹별 pooled OOF</h2><div class="panel">{render_table(group_metrics, ['variant','group','score','nmae','ficr','evaluated_rows'], ['입력','그룹','점수','NMAE','FiCR','평가 행'])}</div>
<h2>그룹·연도별 안정성</h2><div class="panel">{render_table(year_metrics, ['variant','group','year','score','nmae','ficr'], ['입력','그룹','연도','점수','NMAE','FiCR'])}</div>
<h2>Outer fold별 Candidate mapping과 epoch</h2><div class="panel">{render_table(candidate_mappings, ['validation_year','turbine_id','feature','grid_id','mapping_l3_ms','best_iteration','inner_best_iterations','train_rows'], ['검증연도','터빈','LDAPS feature','격자','학습구간 L3','중앙 epoch','inner epoch','학습 행'])}</div>
<h2>SCADA target 품질</h2><div class="panel">{render_table(audit, ['turbine_id','invalid_power_10m_rows','valid_hourly_target_rows','valid_hourly_wind_rows'], ['터빈','제외한 10분 power','유효 시간 target','유효 시간 wind'])}</div>
<h2>그룹 운영정지 제외 현황</h2><div class="panel">{render_table(shutdown_table, ['group','year','hours'], ['그룹','연도','학습 제외 시간'])}</div>
</main></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")


def validate_outputs(
    oof: pd.DataFrame, metrics: pd.DataFrame, mappings: pd.DataFrame
) -> None:
    if oof[["variant", "group", "kst_dtm"]].duplicated().any():
        raise ValueError("OOF variant-group-time 키가 중복되었습니다.")
    if oof[["actual", "prediction"]].isna().any().any():
        raise ValueError("OOF actual 또는 prediction에 결측이 있습니다.")
    counts = oof.groupby("variant").size()
    if len(counts) != 2 or counts.nunique() != 1:
        raise ValueError("Baseline과 Candidate OOF 행 수가 다릅니다.")
    overall = metrics[metrics["scope"].eq("overall")]
    if set(overall["variant"]) != {"baseline", "candidate"}:
        raise ValueError("두 variant의 전체 metric이 필요합니다.")
    expected_mappings = 2 * (12 * 3 + 5 * 2)
    if len(mappings) != expected_mappings:
        raise ValueError(
            f"mapping 행 수가 예상과 다릅니다: {len(mappings)} != {expected_mappings}"
        )


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata = load_metadata(args.data_dir / "info.xlsx")
    labels = pd.read_csv(
        args.data_dir / "train" / "train_labels.csv",
        encoding="utf-8-sig",
        parse_dates=["kst_dtm"],
    ).set_index("kst_dtm")
    ldaps, speed_matrices = load_ldaps(
        args.data_dir / "train" / "ldaps_train.csv"
    )
    targets, winds, shutdown, audit = load_scada(args.data_dir, metadata)
    baseline_mapping = nearest_grid_mapping(metadata, ldaps)
    feature_cache: dict[tuple[str, int], pd.DataFrame] = {}

    def get_features(feature: str, grid_id: int) -> pd.DataFrame:
        key = (feature, int(grid_id))
        if key not in feature_cache:
            feature_cache[key] = build_features(ldaps, feature, int(grid_id))
        return feature_cache[key]

    prediction_rows: list[pd.DataFrame] = []
    mapping_rows: list[dict[str, object]] = []
    audit_index = audit.set_index("turbine_id")
    for group in TARGET_GROUPS:
        group_metadata = metadata[metadata["group"].eq(group)].copy()
        validation_years = sorted(
            interval_year(labels.index[labels[group].notna()]).unique().tolist()
        )
        for validation_year in validation_years:
            train_years = [year for year in validation_years if year != validation_year]
            validation_index = labels.index[
                (interval_year(labels.index) == validation_year) & labels[group].notna()
            ]
            candidate_mapping = {
                turbine: select_candidate_mapping(
                    turbine, train_years, winds, speed_matrices
                )
                for turbine in group_metadata["turbine_id"]
            }
            print(
                f"[{group} / {validation_year}] train={train_years}, "
                f"validation_rows={len(validation_index)}",
                flush=True,
            )
            for variant in ["baseline", "candidate"]:
                group_prediction = np.zeros(len(validation_index), dtype=float)
                mapping = baseline_mapping if variant == "baseline" else candidate_mapping
                for turbine_row in group_metadata.itertuples(index=False):
                    turbine = turbine_row.turbine_id
                    selection = mapping[turbine]
                    feature = str(selection["feature"])
                    grid_id = int(selection["grid_id"])
                    features = get_features(feature, grid_id)
                    if variant == "baseline":
                        mapping_l3_value, mapping_overlap = mapping_l3(
                            winds[turbine],
                            speed_matrices[feature],
                            grid_id,
                            train_years,
                        )
                    else:
                        mapping_l3_value = float(selection["mapping_l3_ms"])
                        mapping_overlap = int(selection["mapping_overlap"])
                    prediction, best_iteration, inner_iterations, train_rows = (
                        train_outer_model(
                            features,
                            targets[turbine],
                            shutdown[group],
                            train_years,
                            validation_index,
                            float(turbine_row.capacity_mw) * 1000.0,
                            args,
                        )
                    )
                    group_prediction += prediction
                    audit_row = audit_index.loc[turbine]
                    mapping_rows.append(
                        {
                            "variant": variant,
                            "group": group,
                            "validation_year": validation_year,
                            "train_years": "|".join(map(str, train_years)),
                            "turbine_id": turbine,
                            "feature": feature,
                            "grid_id": grid_id,
                            "mapping_l3_ms": mapping_l3_value,
                            "mapping_overlap": mapping_overlap,
                            "best_iteration": best_iteration,
                            "inner_best_iterations": "|".join(map(str, inner_iterations)),
                            "train_rows": train_rows,
                            "invalid_power_10m_rows": int(
                                audit_row["invalid_power_10m_rows"]
                            ),
                            "valid_hourly_target_rows": int(
                                audit_row["valid_hourly_target_rows"]
                            ),
                            "group_shutdown_hours": int(
                                shutdown.loc[
                                    interval_year(shutdown.index).isin(train_years), group
                                ].sum()
                            ),
                        }
                    )
                group_prediction = np.clip(
                    group_prediction, 0.0, GROUP_CAPACITY_KWH[group]
                )
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "variant": variant,
                            "group": group,
                            "validation_year": validation_year,
                            "kst_dtm": validation_index,
                            "actual": labels.loc[validation_index, group].to_numpy(float),
                            "prediction": group_prediction,
                        }
                    )
                )
    oof = pd.concat(prediction_rows, ignore_index=True)
    metrics = build_metrics(oof)
    mappings = pd.DataFrame(mapping_rows).sort_values(
        ["variant", "validation_year", "turbine_id"]
    )
    validate_outputs(oof, metrics, mappings)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(oof, args.output_dir / "oof_predictions.csv")
    write_csv(metrics, args.output_dir / "oof_metrics.csv")
    write_csv(mappings, args.output_dir / "feature_mappings.csv")
    write_dashboard(
        args.output_dir / "dashboard.html",
        metrics,
        mappings,
        audit,
        shutdown,
        args,
    )
    return oof, metrics, mappings


def main() -> None:
    args = parse_args()
    _, metrics, _ = run(args)
    overall = metrics[metrics["scope"].eq("overall")][
        ["variant", "score", "nmae", "ficr"]
    ]
    print("\n전체 pooled OOF")
    print(overall.to_string(index=False))
    print(f"\n결과 경로: {args.output_dir}")


if __name__ == "__main__":
    main()

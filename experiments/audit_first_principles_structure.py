from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # noqa: F401
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr
from utils.per_turbine_scada import build_turbine_scada_hourly, turbine_capacity_kwh
from utils.seq_dataset import build_seqnn_weather


YEARS = [2022, 2023, 2024]
WIND_COLUMNS = ["ldaps_ws50_max_speed", "gfs_ws100_speed", "gfs_ws850_speed"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction-file",
        type=Path,
        default=Path("results/per_turbine_issue24_tcn_oof_v1_predictions.csv"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--stem", default="first_principles_audit_v1")
    return parser.parse_args()


def fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 20:
        return np.nan, np.nan
    design = np.column_stack([x[valid], np.ones(int(valid.sum()))])
    slope, intercept = np.linalg.lstsq(design, y[valid], rcond=None)[0]
    return float(slope), float(intercept)


def score_arrays(actual: np.ndarray, prediction: np.ndarray, group: str) -> dict[str, float]:
    nmae, ficr = group_nmae_ficr(actual, prediction, GROUP_CAPACITY_KWH[group])
    return {
        "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
        "nmae": nmae,
        "ficr": ficr,
    }


def prepare_weather_labels(
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    weather = build_seqnn_weather(ldaps, gfs, TARGET_COLS[0])
    weather["forecast_kst_dtm"] = pd.to_datetime(weather["forecast_kst_dtm"])
    weather["data_available_kst_dtm"] = pd.to_datetime(
        weather["data_available_kst_dtm"]
    )
    label_table = labels.rename(columns={"kst_dtm": "forecast_kst_dtm"}).copy()
    label_table["forecast_kst_dtm"] = pd.to_datetime(label_table["forecast_kst_dtm"])
    table = weather.merge(
        label_table[["forecast_kst_dtm", *TARGET_COLS]],
        on="forecast_kst_dtm",
        how="left",
        validate="one_to_one",
    ).sort_values(["data_available_kst_dtm", "forecast_kst_dtm"])
    table["forecast_year"] = table["forecast_kst_dtm"].dt.year
    table["wind_direction_deg"] = (
        np.degrees(
            np.arctan2(table["gfs_ws100_dir_sin"], table["gfs_ws100_dir_cos"])
        )
        + 360.0
    ) % 360.0
    table["direction_bin"] = np.floor(table["wind_direction_deg"] / 30.0).astype(int)
    return table.reset_index(drop=True)


def add_issue_time_offsets(
    table: pd.DataFrame,
    wind_columns: list[str] = WIND_COLUMNS,
    offsets: tuple[int, ...] = (-2, -1, 0, 1, 2),
) -> pd.DataFrame:
    out = table.sort_values(
        ["data_available_kst_dtm", "forecast_kst_dtm"]
    ).copy()
    grouped = out.groupby("data_available_kst_dtm", sort=False)
    for column in wind_columns:
        for offset in offsets:
            # At target t, positive offset means the NWP valid at t+offset.
            out[f"{column}_offset_{offset:+d}"] = grouped[column].shift(-offset)
    return out


def phase_alignment_audit(table: pd.DataFrame) -> pd.DataFrame:
    shifted = add_issue_time_offsets(table)
    rows = []
    for group in TARGET_COLS:
        capacity = GROUP_CAPACITY_KWH[group]
        power_ratio = pd.to_numeric(shifted[group], errors="coerce") / capacity
        target = np.cbrt(power_ratio.clip(lower=0.0)).to_numpy(float)
        mid_power = power_ratio.between(0.10, 0.80).to_numpy()
        for source in WIND_COLUMNS:
            for pred_year in YEARS:
                val_year = shifted["forecast_year"].eq(pred_year).to_numpy()
                if int((val_year & mid_power).sum()) < 200:
                    continue
                train_year = shifted["forecast_year"].isin(
                    [year for year in YEARS if year != pred_year]
                ).to_numpy()
                for offset in [-2, -1, 0, 1, 2]:
                    feature_column = f"{source}_offset_{offset:+d}"
                    wind = pd.to_numeric(
                        shifted[feature_column], errors="coerce"
                    ).to_numpy(float)
                    fit = train_year & mid_power & np.isfinite(wind) & np.isfinite(target)
                    evaluate = val_year & mid_power & np.isfinite(wind) & np.isfinite(target)
                    slope, intercept = fit_affine(wind[fit], target[fit])
                    prediction = slope * wind[evaluate] + intercept
                    error = np.abs(prediction - target[evaluate])
                    correlation = (
                        float(np.corrcoef(wind[evaluate], target[evaluate])[0, 1])
                        if int(evaluate.sum()) > 1
                        else np.nan
                    )
                    rows.append(
                        {
                            "group": group,
                            "source": source,
                            "pred_year": pred_year,
                            "offset_hours": offset,
                            "power_eq_mae": float(error.mean()),
                            "power_eq_corr": correlation,
                            "slope": slope,
                            "intercept": intercept,
                            "n_train": int(fit.sum()),
                            "n_val": int(evaluate.sum()),
                        }
                    )
    return pd.DataFrame(rows)


def share_columns(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    out["total_12"] = out["kpx_group_1"] + out["kpx_group_2"]
    out["share_1_of_12"] = out["kpx_group_1"] / out["total_12"].replace(0.0, np.nan)
    out["total_123"] = out[TARGET_COLS].sum(axis=1, min_count=3)
    out["share_3_of_123"] = out["kpx_group_3"] / out["total_123"].replace(
        0.0, np.nan
    )
    return out


def share_stability_audit(table: pd.DataFrame) -> pd.DataFrame:
    work = share_columns(table)
    rows = []
    for share in ["share_1_of_12", "share_3_of_123"]:
        for (year, direction_bin), part in work.groupby(
            ["forecast_year", "direction_bin"], sort=True
        ):
            values = pd.to_numeric(part[share], errors="coerce").dropna()
            values = values.loc[values.between(0.0, 1.0)]
            if len(values) < 20:
                continue
            rows.append(
                {
                    "share": share,
                    "forecast_year": int(year),
                    "direction_bin": int(direction_bin),
                    "direction_start_deg": int(direction_bin) * 30,
                    "n_rows": len(values),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=0)),
                    "p10": float(values.quantile(0.10)),
                    "p50": float(values.quantile(0.50)),
                    "p90": float(values.quantile(0.90)),
                }
            )
    return pd.DataFrame(rows)


def fit_share_lookup(
    train: pd.DataFrame,
    share_column: str,
    pseudo_count: float = 100.0,
) -> tuple[float, pd.Series]:
    valid = train.dropna(subset=[share_column, "direction_bin"]).copy()
    valid = valid.loc[valid[share_column].between(0.0, 1.0)]
    global_share = float(valid[share_column].mean())
    grouped = valid.groupby("direction_bin")[share_column].agg(["sum", "count"])
    lookup = (grouped["sum"] + pseudo_count * global_share) / (
        grouped["count"] + pseudo_count
    )
    return global_share, lookup


def predict_share(
    table: pd.DataFrame,
    global_share: float,
    lookup: pd.Series | None,
) -> np.ndarray:
    if lookup is None:
        return np.full(len(table), global_share, dtype=float)
    return (
        table["direction_bin"].map(lookup).fillna(global_share).to_numpy(float)
    )


def share_oracle_audit(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = share_columns(table)
    prediction_rows = []
    for pred_year in YEARS:
        train = work.loc[work["forecast_year"].ne(pred_year)]
        val = work.loc[work["forecast_year"].eq(pred_year)].copy()
        if val["share_1_of_12"].notna().sum() < 200:
            continue
        global_12, lookup_12 = fit_share_lookup(train, "share_1_of_12")
        for variant, lookup in [("static", None), ("direction", lookup_12)]:
            share_1 = predict_share(val, global_12, lookup)
            prediction_rows.extend(
                [
                    pd.DataFrame(
                        {
                            "forecast_kst_dtm": val["forecast_kst_dtm"],
                            "variant": f"oracle_s12_{variant}",
                            "group": "kpx_group_1",
                            "pred_year": pred_year,
                            "actual": val["kpx_group_1"],
                            "pred": val["total_12"] * share_1,
                        }
                    ),
                    pd.DataFrame(
                        {
                            "forecast_kst_dtm": val["forecast_kst_dtm"],
                            "variant": f"oracle_s12_{variant}",
                            "group": "kpx_group_2",
                            "pred_year": pred_year,
                            "actual": val["kpx_group_2"],
                            "pred": val["total_12"] * (1.0 - share_1),
                        }
                    ),
                ]
            )

        if val["share_3_of_123"].notna().sum() < 200:
            continue
        global_3, lookup_3 = fit_share_lookup(train, "share_3_of_123")
        for variant, lookups in [
            ("static", (None, None)),
            ("direction", (lookup_12, lookup_3)),
        ]:
            share_1 = predict_share(val, global_12, lookups[0])
            share_3 = predict_share(val, global_3, lookups[1])
            total_123 = val["total_123"].to_numpy(float)
            remaining = total_123 * (1.0 - share_3)
            predictions = {
                "kpx_group_1": remaining * share_1,
                "kpx_group_2": remaining * (1.0 - share_1),
                "kpx_group_3": total_123 * share_3,
            }
            for group in TARGET_COLS:
                prediction_rows.append(
                    pd.DataFrame(
                        {
                            "forecast_kst_dtm": val["forecast_kst_dtm"],
                            "variant": f"oracle_s123_{variant}",
                            "group": group,
                            "pred_year": pred_year,
                            "actual": val[group],
                            "pred": predictions[group],
                        }
                    )
                )

    predictions = pd.concat(prediction_rows, ignore_index=True).dropna(
        subset=["actual", "pred"]
    )
    score_rows = []
    for (variant, group), part in predictions.groupby(["variant", "group"]):
        stats = score_arrays(part["actual"], part["pred"], group)
        score_rows.append(
            {
                "variant": variant,
                "group": group,
                **stats,
                "n_rows": len(part),
            }
        )
    scores = pd.DataFrame(score_rows)
    summary = scores.groupby("variant", as_index=False).agg(
        mean_score=("score", "mean"),
        mean_nmae=("nmae", "mean"),
        mean_ficr=("ficr", "mean"),
        n_groups=("group", "nunique"),
    )
    return scores, summary


def power_equivalent_audit(
    scada_by_group: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for group in TARGET_COLS:
        hourly = build_turbine_scada_hourly(scada_by_group[group], group)
        capacity = turbine_capacity_kwh(group)
        hourly["year"] = pd.to_datetime(hourly["forecast_kst_dtm"]).dt.year
        hourly["power_ratio"] = hourly["scada_power_kwh"] / capacity
        hourly["power_eq"] = np.cbrt(hourly["power_ratio"].clip(lower=0.0))
        for (turbine, year), part in hourly.groupby(["turbine_id", "year"]):
            part = part.loc[
                part["power_ratio"].between(0.10, 0.80)
                & part["scada_ws_cubic"].between(0.0, 40.0)
            ].dropna(subset=["power_eq", "scada_ws_cubic"])
            if len(part) < 200:
                continue
            wind = part["scada_ws_cubic"].to_numpy(float)
            power_eq = part["power_eq"].to_numpy(float)
            slope, intercept = fit_affine(wind, power_eq)
            prediction = slope * wind + intercept
            rows.append(
                {
                    "group": group,
                    "turbine_id": turbine,
                    "year": int(year),
                    "n_rows": len(part),
                    "slope": slope,
                    "intercept": intercept,
                    "mae": float(np.abs(prediction - power_eq).mean()),
                    "corr": float(np.corrcoef(wind, power_eq)[0, 1]),
                    "median_power_eq_per_ms": float(
                        np.median(power_eq / np.maximum(wind, 1e-6))
                    ),
                }
            )
    detail = pd.DataFrame(rows)
    stability = (
        detail.groupby(["group", "turbine_id"], as_index=False)
        .agg(
            n_years=("year", "nunique"),
            mean_corr=("corr", "mean"),
            mean_mae=("mae", "mean"),
            slope_mean=("slope", "mean"),
            slope_std=("slope", lambda values: values.std(ddof=0)),
            coefficient_mean=("median_power_eq_per_ms", "mean"),
            coefficient_std=(
                "median_power_eq_per_ms",
                lambda values: values.std(ddof=0),
            ),
        )
    )
    stability["slope_cv"] = stability["slope_std"] / stability["slope_mean"].abs()
    stability["coefficient_cv"] = (
        stability["coefficient_std"] / stability["coefficient_mean"].abs()
    )
    return detail, stability


def prediction_oracle_decomposition(prediction_file: Path) -> pd.DataFrame:
    if not prediction_file.exists():
        return pd.DataFrame()
    predictions = pd.read_csv(prediction_file, encoding="utf-8-sig")
    required = ["forecast_kst_dtm", "group", "pred", "official_target"]
    missing = [column for column in required if column not in predictions]
    if missing:
        raise ValueError(f"Prediction oracle input missing columns: {missing}")
    predictions["forecast_kst_dtm"] = pd.to_datetime(
        predictions["forecast_kst_dtm"]
    )
    actual = predictions.pivot_table(
        index="forecast_kst_dtm", columns="group", values="official_target", aggfunc="first"
    )
    forecast = predictions.pivot_table(
        index="forecast_kst_dtm", columns="group", values="pred", aggfunc="first"
    )
    common = actual.dropna(subset=TARGET_COLS).index.intersection(
        forecast.dropna(subset=TARGET_COLS).index
    )
    actual = actual.loc[common, TARGET_COLS]
    forecast = forecast.loc[common, TARGET_COLS]
    actual_total = actual.sum(axis=1)
    forecast_total = forecast.sum(axis=1)
    actual_share = actual.div(actual_total.replace(0.0, np.nan), axis=0)
    forecast_share = forecast.div(forecast_total.replace(0.0, np.nan), axis=0)
    variants = {
        "base": forecast,
        "oracle_total_keep_pred_share": forecast_share.mul(actual_total, axis=0),
        "oracle_share_keep_pred_total": actual_share.mul(forecast_total, axis=0),
    }
    rows = []
    for variant, variant_prediction in variants.items():
        for group in TARGET_COLS:
            stats = score_arrays(actual[group], variant_prediction[group], group)
            rows.append(
                {
                    "variant": variant,
                    "group": group,
                    **stats,
                    "n_rows": len(actual),
                }
            )
    scores = pd.DataFrame(rows)
    return scores.merge(
        scores.groupby("variant", as_index=False).agg(mean_score=("score", "mean")),
        on="variant",
        how="left",
    )


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }
    table = prepare_weather_labels(ldaps, gfs, labels)
    prefix = args.results_dir / args.stem

    phase = phase_alignment_audit(table)
    phase.to_csv(f"{prefix}_phase_alignment.csv", index=False, encoding="utf-8-sig")
    phase_summary = (
        phase.groupby(["group", "source", "offset_hours"], as_index=False)
        .agg(
            mean_power_eq_mae=("power_eq_mae", "mean"),
            mean_corr=("power_eq_corr", "mean"),
            n_folds=("pred_year", "count"),
        )
        .sort_values(["group", "source", "mean_power_eq_mae"])
    )
    phase_summary.to_csv(
        f"{prefix}_phase_summary.csv", index=False, encoding="utf-8-sig"
    )

    share_stability = share_stability_audit(table)
    share_stability.to_csv(
        f"{prefix}_share_stability.csv", index=False, encoding="utf-8-sig"
    )
    share_scores, share_summary = share_oracle_audit(table)
    share_scores.to_csv(
        f"{prefix}_share_oracle_scores.csv", index=False, encoding="utf-8-sig"
    )
    share_summary.to_csv(
        f"{prefix}_share_oracle_summary.csv", index=False, encoding="utf-8-sig"
    )

    power_eq, power_eq_stability = power_equivalent_audit(scada_by_group)
    power_eq.to_csv(
        f"{prefix}_power_equivalent.csv", index=False, encoding="utf-8-sig"
    )
    power_eq_stability.to_csv(
        f"{prefix}_power_equivalent_stability.csv",
        index=False,
        encoding="utf-8-sig",
    )

    oracle = prediction_oracle_decomposition(args.prediction_file)
    oracle.to_csv(
        f"{prefix}_prediction_oracle.csv", index=False, encoding="utf-8-sig"
    )

    print("\n=== phase best offsets ===")
    print(phase_summary.groupby(["group", "source"], as_index=False).first().to_string(index=False))
    print("\n=== share oracle ===")
    print(share_summary.sort_values("mean_score", ascending=False).to_string(index=False))
    print("\n=== power-equivalent stability ===")
    print(
        power_eq_stability.groupby("group", as_index=False)
        .agg(
            mean_corr=("mean_corr", "mean"),
            mean_mae=("mean_mae", "mean"),
            mean_coefficient_cv=("coefficient_cv", "mean"),
        )
        .to_string(index=False)
    )
    if len(oracle):
        print("\n=== prediction total/share oracle ===")
        print(
            oracle.groupby("variant", as_index=False)["mean_score"].first()
            .sort_values("mean_score", ascending=False)
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()

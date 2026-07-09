import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import _bootstrap  # noqa: F401
from predict_power_lgbm_best import clean_params, sample_weight
from predict_tree_compact_physics_v2 import build_all_meteo_compact_v2
from tune_power_lgbm_hyperparams import filter_scada_years, filter_weather_years, parse_list, score_one
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature_oof
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_group_dataset


RESULTS_DIR = Path("results")
YEARS = [2022, 2023, 2024]
EPS = 0.1


WIND_RAMP_COLS = [
    "gfs_ws850_speed",
    "gfs_ws100_speed",
    "gfs_ws10_speed",
    "ldaps_ws50_max_speed",
    "ldaps_ws10_speed",
    "phys_gfs_ws850_near_speed",
    "phys_gfs_ws100_near_speed",
    "phys_ldaps_ws50max_grid_max",
    "phys_ldaps_ws10_grid_p90",
]


WEATHER_RAMP_COLS = [
    "phys_gfs_gust",
    "phys_gfs_surface_pressure",
    "phys_ldaps_surface_pressure",
    "phys_gfs_pbl_vrate",
    "phys_gfs_cloud_sum",
    "phys_ldaps_cloud_sum",
    "phys_gfs_air_density_x_gfs_ws850_speed_cube",
    "phys_gfs_air_density_x_gfs_ws100_speed_cube",
    "phys_ldaps_air_density_x_ldaps_ws50_max_speed_cube",
    "phys_ldaps_air_density_x_ldaps_ws10_speed_cube",
]


DIRECTION_PAIRS = [
    ("gfs850", "gfs_isobaricInhPa_850_u", "gfs_isobaricInhPa_850_v"),
    ("gfs100", "gfs_heightAboveGround_100_100u", "gfs_heightAboveGround_100_100v"),
    ("ldaps50max", "ldaps_heightAboveGround_50_50MUmax", "ldaps_heightAboveGround_50_50MVmax"),
]


def _past_mean(series, window):
    return series.shift(1).rolling(window, min_periods=1).mean()


def _add_scalar_ramp(out, col, prefix, add_power=False):
    new_cols = []
    s = out[col].astype(float)
    for lag in [6, 12]:
        name = f"{prefix}_{col}_diff{lag}"
        out[name] = s.diff(lag)
        new_cols.append(name)
    for lag in [3, 6]:
        name = f"{prefix}_{col}_absdiff{lag}"
        out[name] = s.diff(lag).abs()
        new_cols.append(name)
    for window in [3, 6]:
        name = f"{prefix}_{col}_minus_prev{window}mean"
        out[name] = s - _past_mean(s, window)
        new_cols.append(name)
    rel = f"{prefix}_{col}_reldiff3"
    out[rel] = s.diff(3) / (s.abs() + EPS)
    new_cols.append(rel)

    if add_power:
        ws3 = s.clip(lower=0.0) ** 3
        for lag in [3, 6]:
            name = f"{prefix}_{col}_cube_diff{lag}"
            out[name] = ws3.diff(lag)
            new_cols.append(name)
        name = f"{prefix}_{col}_cube_minus_prev6mean"
        out[name] = ws3 - _past_mean(ws3, 6)
        new_cols.append(name)
    return new_cols


def _angle_from_uv(u, v):
    return np.arctan2(np.asarray(v, dtype=float), np.asarray(u, dtype=float))


def _angle_diff(angle, lag):
    prev = angle.shift(lag)
    return np.arctan2(np.sin(angle - prev), np.cos(angle - prev))


def _add_direction_ramp(out):
    new_cols = []
    for name, u_col, v_col in DIRECTION_PAIRS:
        if u_col not in out.columns or v_col not in out.columns:
            continue
        angle = pd.Series(_angle_from_uv(out[u_col], out[v_col]), index=out.index)
        for lag in [3, 6]:
            diff = _angle_diff(angle, lag)
            sin_col = f"phys_ramp_{name}_turn{lag}_sin"
            cos_col = f"phys_ramp_{name}_turn{lag}_cos"
            abs_col = f"phys_ramp_{name}_turn{lag}_abs"
            out[sin_col] = np.sin(diff)
            out[cos_col] = np.cos(diff)
            out[abs_col] = np.abs(diff)
            new_cols.extend([sin_col, cos_col, abs_col])
    return new_cols


def add_weather_ramp_features(weather, mode):
    out = weather.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    out = out.sort_values("forecast_kst_dtm").reset_index(drop=True)
    new_cols = []

    if mode in {"ramp_wind", "ramp_full"}:
        for col in WIND_RAMP_COLS:
            if col in out.columns:
                new_cols.extend(_add_scalar_ramp(out, col, "phys_ramp", add_power=True))

    if mode in {"ramp_weather", "ramp_full"}:
        for col in WEATHER_RAMP_COLS:
            if col in out.columns:
                new_cols.extend(_add_scalar_ramp(out, col, "phys_ramp", add_power=False))
        new_cols.extend(_add_direction_ramp(out))

    out.attrs["ramp_feature_cols"] = [col for col in dict.fromkeys(new_cols) if col in out.columns]
    return out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)


def load_data():
    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    return ldaps, gfs, labels, {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }


def feature_columns(weather):
    return [col for col in weather.columns if col not in TIME_KEY_COLS]


def score_predictions(scores):
    fold_means = (
        scores.groupby(["variant", "pred_year"], as_index=False)
        .agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"))
    )
    summary = (
        fold_means.groupby("variant", as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            worst_fold=("score", "min"),
            std_score=("score", "std"),
        )
        .sort_values("mean_score", ascending=False)
    )
    group_summary = (
        scores.groupby(["variant", "group"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            worst_fold=("score", "min"),
            n_features=("n_features", "max"),
            n_folds=("score", "count"),
        )
    )
    return summary, group_summary, fold_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--best-csv", default="results/power_lgbm_hyperparams_v2_l1_20_best.csv")
    parser.add_argument("--variants", default="baseline,ramp_wind,ramp_weather,ramp_full")
    parser.add_argument("--stem", default="weather_ramp_features_v1")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    groups = parse_list(args.groups)
    variants = parse_list(args.variants)
    best = pd.read_csv(args.best_csv, encoding="utf-8-sig")
    ldaps, gfs, labels, scada_by_group = load_data()

    feature_cache = {}
    ramp_cols = {}
    for group in groups:
        base = build_all_meteo_compact_v2(ldaps, gfs, group)
        feature_cache[(group, "baseline")] = base
        ramp_cols[(group, "baseline")] = []
        for variant in ["ramp_wind", "ramp_weather", "ramp_full"]:
            ramped = add_weather_ramp_features(base, variant)
            feature_cache[(group, variant)] = ramped
            ramp_cols[(group, variant)] = ramped.attrs.get("ramp_feature_cols", [])
        print(
            f"{group}: ramp_wind={len(ramp_cols[(group, 'ramp_wind')])} "
            f"ramp_weather={len(ramp_cols[(group, 'ramp_weather')])} "
            f"ramp_full={len(ramp_cols[(group, 'ramp_full')])}"
        )

    score_rows = []
    pred_parts = []
    feature_rows = []
    for group in groups:
        row = best[best["group"] == group]
        if row.empty:
            continue
        best_row = row.iloc[0]
        params = clean_params(best_row)
        policy = {
            "min_output_ratio": float(best_row["min_output_ratio"]),
            "weight_policy": best_row["weight_policy"],
        }
        for pred_year in YEARS:
            train_years = [year for year in YEARS if year != pred_year]
            scada = filter_scada_years(scada_by_group[group], train_years)
            if len(scada) == 0:
                continue
            for variant in variants:
                base_weather = feature_cache[(group, variant)]
                train_weather_base = filter_weather_years(base_weather, train_years)
                val_weather_base = filter_weather_years(base_weather, [pred_year])
                train_weather, val_weather = add_power_curve_feature_oof(
                    train_weather_base,
                    val_weather_base,
                    scada,
                    group,
                    HUB_HEIGHT_PROXY_COL,
                    GROUP_N_TURBINES[group],
                    out_col="power_curve_est",
                )

                x_train_full, y_train = build_group_dataset(train_weather, labels, group)
                x_val_full, y_val = build_group_dataset(val_weather, labels, group)
                if len(x_train_full) < 1000 or len(x_val_full) < 200:
                    continue
                val_times = (
                    val_weather.merge(labels[["kst_dtm", group]], left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner")
                    .dropna(subset=[group])["forecast_kst_dtm"]
                    .reset_index(drop=True)
                )
                train_mask = y_train.to_numpy(float) >= GROUP_CAPACITY_KWH[group] * policy["min_output_ratio"]
                x_train = x_train_full.loc[train_mask].reset_index(drop=True)
                y_used = y_train.loc[train_mask].reset_index(drop=True)
                feature_cols = feature_columns(train_weather)
                x_train = x_train.reindex(columns=feature_cols, fill_value=0)
                x_val = x_val_full.reindex(columns=feature_cols, fill_value=0)

                model = LGBMRegressor(**params)
                weights = sample_weight(y_used, group, policy["weight_policy"])
                model.fit(x_train, y_used, sample_weight=weights)
                pred = np.clip(model.predict(x_val), 0.0, GROUP_CAPACITY_KWH[group])
                score, nmae, ficr = score_one(y_val, pred, group)
                score_rows.append(
                    {
                        "variant": variant,
                        "group": group,
                        "pred_year": pred_year,
                        "train_years": ",".join(map(str, train_years)),
                        "score": score,
                        "nmae": nmae,
                        "ficr": ficr,
                        "n_features": len(feature_cols),
                        "n_ramp_features": len(ramp_cols[(group, variant)]),
                        "n_used": len(y_used),
                        "n_val": len(y_val),
                    }
                )
                pred_parts.append(
                    pd.DataFrame(
                        {
                            "forecast_kst_dtm": pd.to_datetime(val_times).to_numpy(),
                            "pred_year": pred_year,
                            "train_years": ",".join(map(str, train_years)),
                            "variant": variant,
                            "group": group,
                            "actual": y_val.to_numpy(float),
                            "pred": pred,
                        }
                    )
                )
                print(f"{group} {variant} pred_year={pred_year}: score={score:.5f} nmae={nmae:.5f} ficr={ficr:.5f} features={len(feature_cols)}")

        for variant in variants:
            for col in ramp_cols.get((group, variant), []):
                feature_rows.append({"group": group, "variant": variant, "feature": col})

    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    features = pd.DataFrame(feature_rows).drop_duplicates()
    summary, group_summary, fold_means = score_predictions(scores)

    scores.to_csv(RESULTS_DIR / f"{args.stem}_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS_DIR / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(RESULTS_DIR / f"{args.stem}_group_summary.csv", index=False, encoding="utf-8-sig")
    fold_means.to_csv(RESULTS_DIR / f"{args.stem}_fold_means.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULTS_DIR / f"{args.stem}_predictions.csv", index=False, encoding="utf-8-sig")
    features.to_csv(RESULTS_DIR / f"{args.stem}_features.csv", index=False, encoding="utf-8-sig")

    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print("\n=== group summary ===")
    print(group_summary.sort_values(["group", "mean_score"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()

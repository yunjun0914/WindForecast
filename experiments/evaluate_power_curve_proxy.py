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
DEFAULT_CANDIDATES = [
    HUB_HEIGHT_PROXY_COL,
    "gfs_ws850_speed",
    "ldaps_ws50_max_speed",
    "phys_ldaps_ws50max_grid_max",
    "phys_ldaps_ws50max_grid_p90",
    "phys_ldaps_ws10_grid_max",
    "phys_ldaps_ws10_grid_p90",
    "phys_gfs_ws850_near_speed",
    "phys_gfs_ws850_grid_max",
    "phys_gfs_ws850_upwind_speed",
]


def load_data():
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
    return ldaps, gfs, labels, scada_by_group


def feature_columns(weather):
    return [col for col in weather.columns if col not in TIME_KEY_COLS]


def evaluate_candidate(group, base_weather, labels, scada_all, best_row, proxy_col, mode):
    if proxy_col not in base_weather.columns:
        return [], []

    rows = []
    pred_rows = []
    for pred_year in YEARS:
        train_years = [year for year in YEARS if year != pred_year]
        train_weather_base = filter_weather_years(base_weather, train_years)
        val_weather_base = filter_weather_years(base_weather, [pred_year])
        scada = filter_scada_years(scada_all, train_years)
        if len(scada) == 0:
            continue

        if mode == "replace":
            train_weather, val_weather = add_power_curve_feature_oof(
                train_weather_base,
                val_weather_base,
                scada,
                group,
                proxy_col,
                GROUP_N_TURBINES[group],
                out_col="power_curve_est",
            )
        elif mode == "add":
            train_weather, val_weather = add_power_curve_feature_oof(
                train_weather_base,
                val_weather_base,
                scada,
                group,
                HUB_HEIGHT_PROXY_COL,
                GROUP_N_TURBINES[group],
                out_col="power_curve_est",
            )
            out_col = f"power_curve_{proxy_col}"
            train_weather, val_weather = add_power_curve_feature_oof(
                train_weather,
                val_weather,
                scada,
                group,
                proxy_col,
                GROUP_N_TURBINES[group],
                out_col=out_col,
            )
        else:
            raise ValueError(f"unknown mode: {mode}")

        x_train, y_train = build_group_dataset(train_weather, labels, group)
        x_val, y_val = build_group_dataset(val_weather, labels, group)
        val_times = (
            val_weather.merge(labels[["kst_dtm", group]], left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner")
            .dropna(subset=[group])["forecast_kst_dtm"]
            .reset_index(drop=True)
        )
        if len(x_train) < 1000 or len(x_val) < 200:
            continue

        min_output_ratio = float(best_row["min_output_ratio"])
        train_mask = y_train.to_numpy(float) >= GROUP_CAPACITY_KWH[group] * min_output_ratio
        x_train = x_train.loc[train_mask].reset_index(drop=True)
        y_used = y_train.loc[train_mask].reset_index(drop=True)

        cols = feature_columns(train_weather)
        x_train = x_train.reindex(columns=cols, fill_value=0)
        x_val = x_val.reindex(columns=cols, fill_value=0)
        model = LGBMRegressor(**clean_params(best_row))
        weights = sample_weight(y_used, group, best_row["weight_policy"])
        model.fit(x_train, y_used, sample_weight=weights)
        pred = model.predict(x_val)
        score, nmae, ficr = score_one(y_val, pred, group)

        rows.append(
            {
                "mode": mode,
                "proxy_col": proxy_col,
                "group": group,
                "pred_year": pred_year,
                "train_years": ",".join(map(str, train_years)),
                "score": score,
                "nmae": nmae,
                "ficr": ficr,
                "n_train": len(y_used),
                "n_val": len(y_val),
                "n_features": len(cols),
            }
        )
        pred_rows.append(
            pd.DataFrame(
                {
                    "forecast_kst_dtm": pd.to_datetime(val_times).to_numpy(),
                    "pred_year": pred_year,
                    "train_years": ",".join(map(str, train_years)),
                    "mode": mode,
                    "proxy_col": proxy_col,
                    "group": group,
                    "actual": y_val.to_numpy(float),
                    "pred": pred,
                }
            )
        )
    return rows, pred_rows


def summarize(scores):
    group_summary = (
        scores.groupby(["mode", "proxy_col", "group"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            worst_fold=("score", "min"),
            std_score=("score", "std"),
            n_folds=("score", "count"),
        )
    )
    fold_means = (
        scores.groupby(["mode", "proxy_col", "pred_year"], as_index=False)
        .agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"))
    )
    overall = (
        fold_means.groupby(["mode", "proxy_col"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            worst_fold=("score", "min"),
            std_score=("score", "std"),
        )
        .sort_values("mean_score", ascending=False)
    )
    return overall, group_summary, fold_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--candidates", default=",".join(DEFAULT_CANDIDATES))
    parser.add_argument("--modes", default="replace,add")
    parser.add_argument("--best-csv", default="results/power_lgbm_hyperparams_v2_l1_20_best.csv")
    parser.add_argument("--stem", default="power_curve_proxy_v1")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    groups = parse_list(args.groups)
    candidates = parse_list(args.candidates)
    modes = parse_list(args.modes)
    best = pd.read_csv(args.best_csv, encoding="utf-8-sig")
    ldaps, gfs, labels, scada_by_group = load_data()

    score_rows = []
    pred_parts = []
    available_rows = []
    feature_cache = {}
    for group in groups:
        print(f"build all_meteo_compact_v2 {group}")
        feature_cache[group] = build_all_meteo_compact_v2(ldaps, gfs, group)
        available = [col for col in candidates if col in feature_cache[group].columns]
        missing = [col for col in candidates if col not in feature_cache[group].columns]
        print(f"{group}: available={available}")
        if missing:
            print(f"{group}: missing={missing}")
        for col in candidates:
            available_rows.append({"group": group, "proxy_col": col, "available": col in feature_cache[group].columns})

    for group in groups:
        row = best[best["group"] == group]
        if row.empty:
            print(f"{group}: skip no best params")
            continue
        best_row = row.iloc[0]
        for mode in modes:
            for proxy_col in candidates:
                if proxy_col not in feature_cache[group].columns:
                    continue
                print(f"\n=== {group} mode={mode} proxy={proxy_col} ===")
                rows, preds = evaluate_candidate(
                    group,
                    feature_cache[group],
                    labels,
                    scada_by_group[group],
                    best_row,
                    proxy_col,
                    mode,
                )
                score_rows.extend(rows)
                pred_parts.extend(preds)

    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    overall, group_summary, fold_means = summarize(scores)
    available_df = pd.DataFrame(available_rows)

    scores.to_csv(RESULTS_DIR / f"{args.stem}_scores.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(RESULTS_DIR / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(RESULTS_DIR / f"{args.stem}_group_summary.csv", index=False, encoding="utf-8-sig")
    fold_means.to_csv(RESULTS_DIR / f"{args.stem}_fold_means.csv", index=False, encoding="utf-8-sig")
    available_df.to_csv(RESULTS_DIR / f"{args.stem}_availability.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULTS_DIR / f"{args.stem}_predictions.csv", index=False, encoding="utf-8-sig")

    print("\n=== overall top ===")
    print(overall.head(20).to_string(index=False))
    print("\n=== group top by score ===")
    print(
        group_summary.sort_values(["group", "mean_score"], ascending=[True, False])
        .groupby("group", as_index=False)
        .head(5)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

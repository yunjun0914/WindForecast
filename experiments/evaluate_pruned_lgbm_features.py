import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import _bootstrap  # noqa: F401
from predict_power_lgbm_best import clean_params, sample_weight
from tune_power_lgbm_hyperparams import (
    filter_scada_years,
    filter_weather_years,
    parse_list,
    score_one,
)
from predict_tree_compact_physics_v2 import build_all_meteo_compact_v2
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature_oof
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_group_dataset


RESULTS_DIR = Path("results")
YEARS = [2022, 2023, 2024]
MANDATORY_FEATURES = ["power_curve_est", "sin_doy", "cos_doy", "sin_hod", "cos_hod"]


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


def parse_int_list(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def base_feature_cols(weather):
    return [col for col in weather.columns if col not in TIME_KEY_COLS]


def load_top_features(path, group, k, available_cols, include_mandatory=True):
    importance = pd.read_csv(path, encoding="utf-8-sig")
    part = importance[importance["group"] == group].copy()
    if part.empty:
        raise ValueError(f"no importance rows for {group} in {path}")
    part = part.sort_values("gain_rank")
    selected = [feature for feature in part["feature"].tolist() if feature in available_cols][:k]
    if include_mandatory:
        for feature in MANDATORY_FEATURES:
            if feature in available_cols and feature not in selected:
                selected.append(feature)
    return selected


def evaluate_fold(train_weather, val_weather, labels, group, best_row, feature_cols):
    x_train, y_train = build_group_dataset(train_weather, labels, group)
    x_val, y_val = build_group_dataset(val_weather, labels, group)
    val_times = (
        val_weather.merge(labels[["kst_dtm", group]], left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner")
        .dropna(subset=[group])["forecast_kst_dtm"]
        .reset_index(drop=True)
    )
    min_output_ratio = float(best_row["min_output_ratio"])
    mask = y_train.to_numpy(float) >= GROUP_CAPACITY_KWH[group] * min_output_ratio
    x_train = x_train.loc[mask].reset_index(drop=True)
    y_used = y_train.loc[mask].reset_index(drop=True)
    x_train = x_train.reindex(columns=feature_cols, fill_value=0)
    x_val = x_val.reindex(columns=feature_cols, fill_value=0)

    model = LGBMRegressor(**clean_params(best_row))
    weights = sample_weight(y_used, group, best_row["weight_policy"])
    model.fit(x_train, y_used, sample_weight=weights)
    pred = model.predict(x_val)
    score, nmae, ficr = score_one(y_val, pred, group)
    return score, nmae, ficr, y_val, pred, val_times, len(y_used)


def summarize(scores):
    group_summary = (
        scores.groupby(["variant", "group"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            worst_fold=("score", "min"),
            std_score=("score", "std"),
            n_features=("n_features", "max"),
            n_folds=("score", "count"),
        )
    )
    fold_means = (
        scores.groupby(["variant", "pred_year"], as_index=False)
        .agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"))
    )
    overall = (
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
    return overall, group_summary, fold_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--top-k", default="30,50,80,120,180,260,511")
    parser.add_argument("--best-csv", default="results/power_lgbm_hyperparams_v2_l1_20_best.csv")
    parser.add_argument("--importance-csv", default="results/feature_importance_v1_tree_lgbm_best.csv")
    parser.add_argument("--no-mandatory", action="store_true")
    parser.add_argument("--stem", default="pruned_lgbm_features_v1")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    groups = parse_list(args.groups)
    top_ks = parse_int_list(args.top_k)
    best = pd.read_csv(args.best_csv, encoding="utf-8-sig")
    ldaps, gfs, labels, scada_by_group = load_data()

    feature_cache = {}
    for group in groups:
        print(f"build all_meteo_compact_v2 {group}")
        feature_cache[group] = build_all_meteo_compact_v2(ldaps, gfs, group)

    score_rows = []
    pred_parts = []
    feature_rows = []
    for group in groups:
        best_row = best[best["group"] == group]
        if best_row.empty:
            print(f"{group}: skip no best params")
            continue
        best_row = best_row.iloc[0]
        base_weather = feature_cache[group]

        for k in top_ks:
            variant = f"top{k}"
            print(f"\n=== {group} {variant} ===")
            for pred_year in YEARS:
                train_years = [year for year in YEARS if year != pred_year]
                scada = filter_scada_years(scada_by_group[group], train_years)
                if len(scada) == 0:
                    continue
                train_weather_base = filter_weather_years(base_weather, train_years)
                val_weather_base = filter_weather_years(base_weather, [pred_year])
                train_weather, val_weather = add_power_curve_feature_oof(
                    train_weather_base,
                    val_weather_base,
                    scada,
                    group,
                    HUB_HEIGHT_PROXY_COL,
                    GROUP_N_TURBINES[group],
                )
                available = base_feature_cols(train_weather)
                feature_cols = load_top_features(
                    args.importance_csv,
                    group,
                    min(k, len(available)),
                    available,
                    include_mandatory=not args.no_mandatory,
                )
                score, nmae, ficr, y_val, pred, val_times, n_used = evaluate_fold(
                    train_weather,
                    val_weather,
                    labels,
                    group,
                    best_row,
                    feature_cols,
                )
                score_rows.append(
                    {
                        "variant": variant,
                        "top_k": k,
                        "group": group,
                        "pred_year": pred_year,
                        "train_years": ",".join(map(str, train_years)),
                        "score": score,
                        "nmae": nmae,
                        "ficr": ficr,
                        "n_used": n_used,
                        "n_val": len(y_val),
                        "n_features": len(feature_cols),
                    }
                )
                pred_parts.append(
                    pd.DataFrame(
                        {
                            "forecast_kst_dtm": pd.to_datetime(val_times).to_numpy(),
                            "pred_year": pred_year,
                            "train_years": ",".join(map(str, train_years)),
                            "variant": variant,
                            "top_k": k,
                            "group": group,
                            "actual": y_val.to_numpy(float),
                            "pred": pred,
                        }
                    )
                )
                for rank, feature in enumerate(feature_cols, start=1):
                    feature_rows.append(
                        {"variant": variant, "top_k": k, "group": group, "feature_rank_used": rank, "feature": feature}
                    )
                print(f"{group} {variant} pred_year={pred_year}: score={score:.5f} ficr={ficr:.5f} n_features={len(feature_cols)}")

    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    features = pd.DataFrame(feature_rows).drop_duplicates()
    overall, group_summary, fold_means = summarize(scores)

    scores.to_csv(RESULTS_DIR / f"{args.stem}_scores.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(RESULTS_DIR / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(RESULTS_DIR / f"{args.stem}_group_summary.csv", index=False, encoding="utf-8-sig")
    fold_means.to_csv(RESULTS_DIR / f"{args.stem}_fold_means.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULTS_DIR / f"{args.stem}_predictions.csv", index=False, encoding="utf-8-sig")
    features.to_csv(RESULTS_DIR / f"{args.stem}_features.csv", index=False, encoding="utf-8-sig")

    print("\n=== overall ===")
    print(overall.to_string(index=False))
    print("\n=== group best ===")
    print(
        group_summary.sort_values(["group", "mean_score"], ascending=[True, False])
        .groupby("group", as_index=False)
        .head(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import _bootstrap  # noqa: F401
from analyze_feature_families import classify_feature
from evaluate_family_pruned_lgbm import VARIANTS, build_family_map, load_data, select_features
from predict_power_lgbm_best import clean_params, sample_weight
from predict_tree_compact_physics_v2 import build_all_meteo_compact_v2
from tune_power_lgbm_hyperparams import filter_scada_years, filter_weather_years, parse_list, score_one
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature_oof
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_group_dataset


RESULTS_DIR = Path("results")
YEARS = [2022, 2023, 2024]


def feature_columns(weather):
    return [col for col in weather.columns if col not in TIME_KEY_COLS]


def parse_int_list(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def seeded_params(best_row, seed):
    params = clean_params(best_row).copy()
    params["random_state"] = seed
    params["bagging_seed"] = seed + 101
    params["feature_fraction_seed"] = seed + 202
    params["data_random_seed"] = seed + 303
    return params


def summarize(scores):
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
            n_seeds=("n_seeds", "max"),
            n_folds=("score", "count"),
        )
    )
    return summary, group_summary, fold_means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--variants", default="full,drop_selected_low")
    parser.add_argument("--seeds", default="2026070801,2026070802,2026070803,2026070804,2026070805")
    parser.add_argument("--best-csv", default="results/power_lgbm_hyperparams_v2_l1_20_best.csv")
    parser.add_argument("--family-csv", default="results/feature_family_v2_tagged_features.csv")
    parser.add_argument("--stem", default="lgbm_seed_bagging_v1")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    groups = parse_list(args.groups)
    variant_names = parse_list(args.variants)
    seeds = parse_int_list(args.seeds)
    best = pd.read_csv(args.best_csv, encoding="utf-8-sig")
    family_map = build_family_map(args.family_csv)
    ldaps, gfs, labels, scada_by_group = load_data()

    feature_cache = {}
    for group in groups:
        print(f"build all_meteo_compact_v2 {group}")
        feature_cache[group] = build_all_meteo_compact_v2(ldaps, gfs, group)

    score_rows = []
    pred_parts = []
    for group in groups:
        row = best[best["group"] == group]
        if row.empty:
            continue
        best_row = row.iloc[0]
        base_weather = feature_cache[group]
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
            x_train_full, y_train = build_group_dataset(train_weather, labels, group)
            x_val_full, y_val = build_group_dataset(val_weather, labels, group)
            if len(x_train_full) < 1000 or len(x_val_full) < 200:
                print(f"{group} pred_year={pred_year}: skip insufficient rows train={len(x_train_full)} val={len(x_val_full)}")
                continue
            val_times = (
                val_weather.merge(labels[["kst_dtm", group]], left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner")
                .dropna(subset=[group])["forecast_kst_dtm"]
                .reset_index(drop=True)
            )
            min_output_ratio = float(best_row["min_output_ratio"])
            train_mask = y_train.to_numpy(float) >= GROUP_CAPACITY_KWH[group] * min_output_ratio
            y_used = y_train.loc[train_mask].reset_index(drop=True)
            all_features = feature_columns(train_weather)

            for variant in variant_names:
                if variant not in VARIANTS:
                    raise ValueError(f"unknown variant: {variant}. choices={list(VARIANTS)}")
                selected, _ = select_features(all_features, VARIANTS[variant], family_map)
                x_train = x_train_full.loc[train_mask].reset_index(drop=True).reindex(columns=selected, fill_value=0)
                x_val = x_val_full.reindex(columns=selected, fill_value=0)
                seed_preds = []
                for seed in seeds:
                    model = LGBMRegressor(**seeded_params(best_row, seed))
                    weights = sample_weight(y_used, group, best_row["weight_policy"])
                    model.fit(x_train, y_used, sample_weight=weights)
                    seed_preds.append(model.predict(x_val))
                pred = np.mean(seed_preds, axis=0)
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
                        "n_features": len(selected),
                        "n_used": len(y_used),
                        "n_val": len(y_val),
                        "n_seeds": len(seeds),
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
                print(f"{group} {variant} pred_year={pred_year}: score={score:.5f} nmae={nmae:.5f} ficr={ficr:.5f}")

    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    summary, group_summary, fold_means = summarize(scores)
    scores.to_csv(RESULTS_DIR / f"{args.stem}_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(RESULTS_DIR / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(RESULTS_DIR / f"{args.stem}_group_summary.csv", index=False, encoding="utf-8-sig")
    fold_means.to_csv(RESULTS_DIR / f"{args.stem}_fold_means.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULTS_DIR / f"{args.stem}_predictions.csv", index=False, encoding="utf-8-sig")

    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print("\n=== group summary ===")
    print(group_summary.sort_values(["group", "mean_score"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()

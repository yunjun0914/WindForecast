import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import _bootstrap  # noqa: F401
from predict_power_lgbm_best import clean_params, sample_weight
from predict_tree_compact_physics_v2 import build_all_meteo_compact_v2
from tune_power_lgbm_hyperparams import filter_scada_years, filter_weather_years, score_one
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS
from utils.pinn_effective_pipeline import (
    SCADA_DIRECTION_TARGETS,
    SCADA_WS_TARGETS,
    apply_extended_teacher_crossfit,
)
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature_oof
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_group_dataset


RESULTS_DIR = Path("results")
YEARS = [2022, 2023, 2024]
GROUPS = TARGET_COLS
TEACHER_V_MODE = {
    "kpx_group_1": "cubic",
    "kpx_group_2": "p90",
    "kpx_group_3": "p90",
}


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


def evaluate_variant(group, base_weather, labels, scada_all, best_row, variant):
    rows = []
    pred_parts = []
    for pred_year in YEARS:
        train_years = [year for year in YEARS if year != pred_year]
        train_base = filter_weather_years(base_weather, train_years)
        val_base = filter_weather_years(base_weather, [pred_year])
        scada = filter_scada_years(scada_all, train_years)
        if len(scada) == 0:
            continue

        if variant == "scada_wd_teacher":
            train_base, val_base = apply_extended_teacher_crossfit(
                train_base,
                val_base,
                scada,
                group,
                TEACHER_V_MODE[group],
                backend="lgbm_time_oof",
            )
        elif variant != "baseline":
            raise ValueError(f"unknown variant: {variant}")

        train_weather, val_weather = add_power_curve_feature_oof(
            train_base,
            val_base,
            scada,
            group,
            HUB_HEIGHT_PROXY_COL,
            GROUP_N_TURBINES[group],
        )

        x_train, y_train = build_group_dataset(train_weather, labels, group)
        x_val, y_val = build_group_dataset(val_weather, labels, group)
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
        pred = np.clip(model.predict(x_val), 0.0, GROUP_CAPACITY_KWH[group])
        score, nmae, ficr = score_one(y_val, pred, group)

        rows.append(
            {
                "variant": variant,
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
        pred_parts.append(
            pd.DataFrame(
                {
                    "variant": variant,
                    "group": group,
                    "pred_year": pred_year,
                    "actual": y_val.to_numpy(float),
                    "pred": pred,
                }
            )
        )
    return rows, pred_parts


def summarize(scores):
    group_summary = (
        scores.groupby(["variant", "group"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            worst_fold=("score", "min"),
            std_score=("score", "std"),
            n_features=("n_features", "mean"),
        )
        .sort_values(["group", "mean_score"], ascending=[True, False])
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
    parser.add_argument("--best-csv", default="results/power_lgbm_hyperparams_v2_l1_20_best.csv")
    parser.add_argument("--stem", default="scada_wd_teacher_lgbm_v1")
    parser.add_argument("--variants", default="baseline,scada_wd_teacher")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    best = pd.read_csv(args.best_csv, encoding="utf-8-sig")
    variants = [part.strip() for part in args.variants.split(",") if part.strip()]
    ldaps, gfs, labels, scada_by_group = load_data()

    score_rows = []
    pred_parts = []
    target_rows = []
    feature_cache = {}
    for group in GROUPS:
        print(f"build all_meteo_compact_v2 {group}")
        feature_cache[group] = build_all_meteo_compact_v2(ldaps, gfs, group)
        for target in SCADA_WS_TARGETS:
            target_rows.append({"group": group, "scada_wd_feature": target})
        for target in SCADA_DIRECTION_TARGETS:
            target_rows.append({"group": group, "scada_wd_feature": target})

    for group in GROUPS:
        row = best[best["group"] == group]
        if row.empty:
            print(f"{group}: skip missing best params")
            continue
        best_row = row.iloc[0]
        for variant in variants:
            print(f"\n=== {group} variant={variant} ===")
            rows, preds = evaluate_variant(
                group,
                feature_cache[group],
                labels,
                scada_by_group[group],
                best_row,
                variant,
            )
            score_rows.extend(rows)
            pred_parts.extend(preds)

    scores = pd.DataFrame(score_rows)
    predictions = pd.concat(pred_parts, ignore_index=True) if pred_parts else pd.DataFrame()
    overall, group_summary, fold_means = summarize(scores)

    scores.to_csv(RESULTS_DIR / f"{args.stem}_scores.csv", index=False, encoding="utf-8-sig")
    overall.to_csv(RESULTS_DIR / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")
    group_summary.to_csv(RESULTS_DIR / f"{args.stem}_group_summary.csv", index=False, encoding="utf-8-sig")
    fold_means.to_csv(RESULTS_DIR / f"{args.stem}_fold_means.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(RESULTS_DIR / f"{args.stem}_predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(target_rows).to_csv(RESULTS_DIR / f"{args.stem}_features.csv", index=False, encoding="utf-8-sig")

    print("\n=== overall ===")
    print(overall.to_string(index=False))
    print("\n=== group summary ===")
    print(group_summary.to_string(index=False))
    print(f"added unified SCADA teacher features: {', '.join([*SCADA_WS_TARGETS, *SCADA_DIRECTION_TARGETS])}")
    return overall, group_summary, fold_means


if __name__ == "__main__":
    main()

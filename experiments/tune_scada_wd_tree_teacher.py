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
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature_oof
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_group_dataset
from utils.scada_direction_features import SCADA_DIRECTION_TARGETS, add_scada_direction_teacher_oof


RESULTS_DIR = Path("results")
YEARS = [2022, 2023, 2024]
GROUPS = TARGET_COLS


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


def parse_configs(text):
    presets = {
        "baseline": {"variant": "baseline", "backend": None, "params": {}, "n_splits": 0},
        "et80_d18_l20_f070_s3": {
            "variant": "et80_d18_l20_f070_s3",
            "backend": "extra_trees",
            "params": {"n_estimators": 80, "max_depth": 18, "min_samples_leaf": 20, "max_features": 0.70},
            "n_splits": 3,
        },
        "et120_d18_l10_f070_s3": {
            "variant": "et120_d18_l10_f070_s3",
            "backend": "extra_trees",
            "params": {"n_estimators": 120, "max_depth": 18, "min_samples_leaf": 10, "max_features": 0.70},
            "n_splits": 3,
        },
        "et120_d24_l20_f080_s3": {
            "variant": "et120_d24_l20_f080_s3",
            "backend": "extra_trees",
            "params": {"n_estimators": 120, "max_depth": 24, "min_samples_leaf": 20, "max_features": 0.80},
            "n_splits": 3,
        },
        "et160_none_l30_f060_s3": {
            "variant": "et160_none_l30_f060_s3",
            "backend": "extra_trees",
            "params": {"n_estimators": 160, "max_depth": None, "min_samples_leaf": 30, "max_features": 0.60},
            "n_splits": 3,
        },
    }
    names = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in presets]
    if unknown:
        raise ValueError(f"unknown config(s): {unknown}; available={sorted(presets)}")
    return [presets[name] for name in names]


def apply_variant(train_base, val_base, scada, group, config):
    if config["variant"] == "baseline":
        return train_base, val_base
    return add_scada_direction_teacher_oof(
        train_base,
        val_base,
        scada,
        group,
        n_splits=config["n_splits"],
        backend=config["backend"],
        teacher_params=config["params"],
    )


def evaluate_config(group, base_weather, labels, scada_all, best_row, config):
    rows = []
    pred_parts = []
    for pred_year in YEARS:
        train_years = [year for year in YEARS if year != pred_year]
        train_base = filter_weather_years(base_weather, train_years)
        val_base = filter_weather_years(base_weather, [pred_year])
        scada = filter_scada_years(scada_all, train_years)
        if len(scada) == 0:
            continue

        train_base, val_base = apply_variant(train_base, val_base, scada, group, config)
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
                "variant": config["variant"],
                "group": group,
                "pred_year": pred_year,
                "train_years": ",".join(map(str, train_years)),
                "score": score,
                "nmae": nmae,
                "ficr": ficr,
                "n_train": len(y_used),
                "n_val": len(y_val),
                "n_features": len(cols),
                "teacher_backend": config["backend"] or "",
                "teacher_n_splits": config["n_splits"],
                **{f"teacher_{k}": v for k, v in config["params"].items()},
            }
        )
        pred_parts.append(
            pd.DataFrame(
                {
                    "variant": config["variant"],
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
    parser.add_argument("--stem", default="scada_wd_tree_teacher_tune_v1")
    parser.add_argument(
        "--configs",
        default="baseline,et80_d18_l20_f070_s3,et120_d18_l10_f070_s3,et120_d24_l20_f080_s3,et160_none_l30_f060_s3",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    configs = parse_configs(args.configs)
    best = pd.read_csv(args.best_csv, encoding="utf-8-sig")
    ldaps, gfs, labels, scada_by_group = load_data()

    feature_cache = {}
    for group in GROUPS:
        print(f"build all_meteo_compact_v2 {group}")
        feature_cache[group] = build_all_meteo_compact_v2(ldaps, gfs, group)

    score_rows = []
    pred_parts = []
    for group in GROUPS:
        row = best[best["group"] == group]
        if row.empty:
            print(f"{group}: skip missing best params")
            continue
        best_row = row.iloc[0]
        for config in configs:
            print(f"\n=== {group} variant={config['variant']} ===")
            rows, preds = evaluate_config(group, feature_cache[group], labels, scada_by_group[group], best_row, config)
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

    print("\n=== overall ===")
    print(overall.to_string(index=False))
    print("\n=== group summary ===")
    print(group_summary.to_string(index=False))
    print(f"teacher targets: {', '.join(SCADA_DIRECTION_TARGETS)}")
    return overall, group_summary, fold_means


if __name__ == "__main__":
    main()

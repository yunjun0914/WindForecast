import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict

from evaluate_multi_year_generalization import MIN_TRAIN_ROWS, MIN_VAL_ROWS, score_arrays, time_split_arrays
from evaluate_tree_meteo_feature_blocks_fast import add_meteo_block, build_meteo_features
from predict_tree_feature_block import CV_SPLITTER, MODELS, ensemble_fit_predict
from utils.metrics import GROUP_CAPACITY_KWH
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, build_weather_features

YEARLY_FOLDS = [
    {"fold": "year_2023", "fold_type": "yearly", "val_start": "2023-01-01 01:00:00", "val_end": "2024-01-01 01:00:00"},
    {"fold": "year_2024", "fold_type": "yearly", "val_start": "2024-01-01 01:00:00", "val_end": "2025-01-01 01:00:00"},
]
QUARTER_FOLDS = [
    {"fold": "q2024_1", "fold_type": "quarter", "val_start": "2024-01-01 01:00:00", "val_end": "2024-04-01 01:00:00"},
    {"fold": "q2024_2", "fold_type": "quarter", "val_start": "2024-04-01 01:00:00", "val_end": "2024-07-01 01:00:00"},
    {"fold": "q2024_3", "fold_type": "quarter", "val_start": "2024-07-01 01:00:00", "val_end": "2024-10-01 01:00:00"},
    {"fold": "q2024_4", "fold_type": "quarter", "val_start": "2024-10-01 01:00:00", "val_end": "2025-01-01 01:00:00"},
]
BLOCKS = ["baseline", "all_meteo"]
RESULTS_PATH = "results/tree_meteo_multi_year_scores.csv"
SUMMARY_PATH = "results/tree_meteo_multi_year_summary.csv"


def filter_scada_before(scada_df, val_start):
    scada = scada_df.copy()
    scada["kst_dtm"] = pd.to_datetime(scada["kst_dtm"])
    return scada[scada["kst_dtm"] < pd.Timestamp(val_start)].reset_index(drop=True)


def evaluate_fold_block(fold, weather, labels, scada_by_group):
    per_group = {}
    pooled_oof_pred_pct, pooled_oof_actual_pct = [], []

    for group, capacity in GROUP_CAPACITY_KWH.items():
        scada = filter_scada_before(scada_by_group[group], fold["val_start"])
        if len(scada) == 0:
            continue
        curve_fn = fit_group_power_curve(scada, group)
        group_weather = add_power_curve_feature(weather, HUB_HEIGHT_PROXY_COL, curve_fn, GROUP_N_TURBINES[group])
        x_train, x_val, y_train, y_val, val_time = time_split_arrays(
            group_weather, labels, group, fold["val_start"], fold["val_end"]
        )
        if len(x_train) < MIN_TRAIN_ROWS or len(x_val) < MIN_VAL_ROWS:
            continue

        oof = np.zeros(len(y_train))
        for model in MODELS.values():
            oof += cross_val_predict(clone(model), x_train, y_train, cv=CV_SPLITTER, n_jobs=1)
        oof /= len(MODELS)
        raw_val = ensemble_fit_predict(x_train, y_train, x_val)
        per_group[group] = {
            "time": val_time,
            "y": y_val.to_numpy(),
            "raw": np.clip(raw_val, 0, capacity),
            "capacity": capacity,
        }
        pooled_oof_pred_pct.append(oof / capacity)
        pooled_oof_actual_pct.append(y_train.to_numpy() / capacity)

    if pooled_oof_pred_pct:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(np.concatenate(pooled_oof_pred_pct), np.concatenate(pooled_oof_actual_pct))
        for group, data in per_group.items():
            capacity = data["capacity"]
            data["pooled_isotonic"] = np.clip(calibrator.predict(data["raw"] / capacity) * capacity, 0, capacity)
    return per_group


def append_rows(rows, fold, block, pred_by_group):
    for group, data in pred_by_group.items():
        for variant in ["raw", "pooled_isotonic"]:
            score, nmae, ficr = score_arrays(data["y"], data[variant], data["capacity"])
            rows.append(
                {
                    **fold,
                    "feature_block": block,
                    "variant": variant,
                    "group": group,
                    "score": score,
                    "nmae": nmae,
                    "ficr": ficr,
                    "n": len(data["y"]),
                }
            )


def add_mean_rows(results):
    means = (
        results.groupby(["fold", "fold_type", "val_start", "val_end", "feature_block", "variant"], as_index=False)
        .agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"), n=("n", "sum"))
    )
    means["group"] = "mean"
    return pd.concat([results, means[results.columns]], ignore_index=True)


def main():
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

    weather_base = build_weather_features(ldaps, gfs)
    meteo = build_meteo_features(ldaps, gfs)
    weather_by_block = {block: add_meteo_block(weather_base, meteo, block) for block in BLOCKS}

    rows = []
    for fold in YEARLY_FOLDS + QUARTER_FOLDS:
        print(f"\n=== {fold['fold']} ===")
        for block, weather in weather_by_block.items():
            pred_by_group = evaluate_fold_block(fold, weather, labels, scada_by_group)
            append_rows(rows, fold, block, pred_by_group)
            if pred_by_group:
                tmp = pd.DataFrame(rows)
                current = tmp[
                    (tmp["fold"] == fold["fold"])
                    & (tmp["feature_block"] == block)
                    & (tmp["group"] != "mean")
                ]
                means = current.groupby("variant")["score"].mean().sort_values(ascending=False)
                print(f"{block}: {means.round(4).to_dict()}")

    results = add_mean_rows(pd.DataFrame(rows))
    summary = (
        results[results["group"] == "mean"]
        .groupby(["fold_type", "feature_block", "variant"], as_index=False)
        .agg(mean_score=("score", "mean"), worst_fold=("score", "min"), std_score=("score", "std"), n_folds=("score", "count"))
        .sort_values(["fold_type", "variant", "mean_score"], ascending=[True, True, False])
    )
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    return results, summary


if __name__ == "__main__":
    main()

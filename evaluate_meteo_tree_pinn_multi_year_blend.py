import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict

from evaluate_group3_effective_teacher_mix import blend_weather
from evaluate_multi_year_generalization import train_pinn_fold
from evaluate_pinn_effective_wind_teacher import (
    apply_extended_teacher,
    build_extended_pinn_weather,
    fit_extended_teacher,
)
from evaluate_tree_meteo_feature_blocks_fast import add_meteo_block, build_meteo_features
from predict_tree_feature_block import CV_SPLITTER, MODELS, ensemble_fit_predict
from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr
from utils.power_curve import GROUP_N_TURBINES as TREE_GROUP_N_TURBINES
from utils.power_curve import add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_weather_features

GROUPS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]
FOLDS = [
    {"fold": "year_2023", "val_start": "2023-01-01 01:00:00", "val_end": "2024-01-01 01:00:00"},
    {"fold": "year_2024", "val_start": "2024-01-01 01:00:00", "val_end": "2025-01-01 01:00:00"},
]
GLOBAL_WEIGHTS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 1.0]
HOLDOUT_BEST_WEIGHTS = {"kpx_group_1": 0.55, "kpx_group_2": 0.55, "kpx_group_3": 0.70}
RESULTS_PATH = "results/meteo_tree_pinn_multi_year_blend_scores.csv"
SUMMARY_PATH = "results/meteo_tree_pinn_multi_year_blend_summary.csv"


def score_one(y, pred, capacity):
    pred = np.clip(pred, 0, capacity)
    nmae, ficr = group_nmae_ficr(y, pred, capacity)
    return 0.5 * (1 - nmae) + 0.5 * ficr, nmae, ficr


def time_split(weather_df, labels_df, group, val_start, val_end):
    weather = weather_df.copy()
    weather["forecast_kst_dtm"] = pd.to_datetime(weather["forecast_kst_dtm"])
    labels = labels_df[["kst_dtm", group]].copy()
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    merged = weather.merge(labels, left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner")
    merged = merged.dropna(subset=[group]).reset_index(drop=True)
    val_start_ts = pd.Timestamp(val_start)
    val_end_ts = pd.Timestamp(val_end)
    train_mask = merged["forecast_kst_dtm"] < val_start_ts
    val_mask = (merged["forecast_kst_dtm"] >= val_start_ts) & (merged["forecast_kst_dtm"] < val_end_ts)
    feature_cols = [c for c in weather.columns if c not in TIME_KEY_COLS]
    return (
        merged.loc[train_mask, feature_cols].reset_index(drop=True),
        merged.loc[val_mask, feature_cols].reset_index(drop=True),
        merged.loc[train_mask, group].reset_index(drop=True),
        merged.loc[val_mask, group].reset_index(drop=True),
        merged.loc[val_mask, "forecast_kst_dtm"].reset_index(drop=True),
    )


def filter_scada(scada_df, val_start):
    scada = scada_df.copy()
    scada["kst_dtm"] = pd.to_datetime(scada["kst_dtm"])
    return scada[scada["kst_dtm"] < pd.Timestamp(val_start)].reset_index(drop=True)


def build_tree_predictions(fold, weather, labels, scada_vestas, scada_unison):
    scada_by_group = {
        "kpx_group_1": filter_scada(scada_vestas, fold["val_start"]),
        "kpx_group_2": filter_scada(scada_vestas, fold["val_start"]),
        "kpx_group_3": filter_scada(scada_unison, fold["val_start"]),
    }
    per_group = {}
    pooled_oof_pred_pct, pooled_oof_actual_pct = [], []
    for group, capacity in GROUP_CAPACITY_KWH.items():
        if len(scada_by_group[group]) == 0:
            continue
        curve_fn = fit_group_power_curve(scada_by_group[group], group)
        group_weather = add_power_curve_feature(weather, HUB_HEIGHT_PROXY_COL, curve_fn, TREE_GROUP_N_TURBINES[group])
        x_train, x_val, y_train, y_val, val_time = time_split(
            group_weather, labels, group, fold["val_start"], fold["val_end"]
        )
        if len(x_train) < 1000 or len(x_val) < 200:
            continue
        oof = np.zeros(len(y_train))
        for model in MODELS.values():
            oof += cross_val_predict(clone(model), x_train, y_train, cv=CV_SPLITTER, n_jobs=1)
        oof /= len(MODELS)
        raw = ensemble_fit_predict(x_train, y_train, x_val)
        per_group[group] = {
            "time": pd.to_datetime(val_time).reset_index(drop=True),
            "y": y_val.to_numpy(),
            "raw": np.clip(raw, 0, capacity),
            "capacity": capacity,
        }
        pooled_oof_pred_pct.append(oof / capacity)
        pooled_oof_actual_pct.append(y_train.to_numpy() / capacity)

    if pooled_oof_pred_pct:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(np.concatenate(pooled_oof_pred_pct), np.concatenate(pooled_oof_actual_pct))
        for group, data in per_group.items():
            capacity = data["capacity"]
            data["calibrated"] = np.clip(calibrator.predict(data["raw"] / capacity) * capacity, 0, capacity)
    return per_group


def teacher_weather(weather, scada_df, group, mode, val_start):
    teacher = fit_extended_teacher(weather, scada_df, group, fit_before=val_start)
    return apply_extended_teacher(weather, teacher, mode)


def build_pinn_weather_by_group(fold, weather, scada_vestas, scada_unison):
    g1 = teacher_weather(weather, scada_vestas, "kpx_group_1", "cubic", fold["val_start"])
    g2 = teacher_weather(weather, scada_vestas, "kpx_group_2", "p90", fold["val_start"])
    g3_unison = teacher_weather(weather, scada_unison, "kpx_group_3", "p90", fold["val_start"])
    g3_vestas = teacher_weather(weather, scada_vestas, "kpx_group_2", "p90", fold["val_start"])
    g3 = blend_weather("group3_meteo_p90_mix", g3_unison, g3_vestas, 0.30)
    return {"kpx_group_1": g1, "kpx_group_2": g2, "kpx_group_3": g3}


def predict_pinn_fold(fold, weather_by_group, labels):
    vestas_pred = train_pinn_fold(
        "vestas",
        {"kpx_group_1": weather_by_group["kpx_group_1"], "kpx_group_2": weather_by_group["kpx_group_2"]},
        labels,
        fold["val_start"],
        fold["val_end"],
    )
    unison_pred = train_pinn_fold(
        "unison",
        {"kpx_group_3": weather_by_group["kpx_group_3"]},
        labels,
        fold["val_start"],
        fold["val_end"],
    )
    return {**vestas_pred, **unison_pred}


def append_candidate(rows, fold, candidate, tree, pinn, weights):
    group_scores = []
    for group in sorted(set(tree) & set(pinn)):
        if not tree[group]["time"].equals(pinn[group]["time"]):
            raise ValueError(f"time mismatch {fold['fold']} {group}")
        capacity = tree[group]["capacity"]
        pred = np.clip((1 - weights[group]) * tree[group]["calibrated"] + weights[group] * pinn[group]["pred"], 0, capacity)
        score, nmae, ficr = score_one(tree[group]["y"], pred, capacity)
        rows.append(
            {
                "fold": fold["fold"],
                "candidate": candidate,
                "group": group,
                "pinn_weight": weights[group],
                "score": score,
                "nmae": nmae,
                "ficr": ficr,
                "n": len(tree[group]["y"]),
            }
        )
        group_scores.append(score)
    if group_scores:
        rows.append(
            {
                "fold": fold["fold"],
                "candidate": candidate,
                "group": "mean",
                "pinn_weight": float(np.mean([weights[g] for g in sorted(set(tree) & set(pinn))])),
                "score": float(np.mean(group_scores)),
                "nmae": np.nan,
                "ficr": np.nan,
                "n": sum(len(tree[g]["y"]) for g in sorted(set(tree) & set(pinn))),
            }
        )


def main():
    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")

    weather_tree = add_meteo_block(build_weather_features(ldaps, gfs), build_meteo_features(ldaps, gfs), "all_meteo")
    weather_pinn = add_meteo_block(build_extended_pinn_weather(ldaps, gfs), build_meteo_features(ldaps, gfs), "all_meteo")

    rows = []
    for fold in FOLDS:
        print(f"\n=== {fold['fold']} ===")
        tree = build_tree_predictions(fold, weather_tree, labels, scada_vestas, scada_unison)
        pinn_weather = build_pinn_weather_by_group(fold, weather_pinn, scada_vestas, scada_unison)
        pinn = predict_pinn_fold(fold, pinn_weather, labels)
        for weight in GLOBAL_WEIGHTS:
            append_candidate(rows, fold, f"global_pinn_{weight:.2f}", tree, pinn, {g: weight for g in GROUPS})
        append_candidate(rows, fold, "holdout_best_weights", tree, pinn, HOLDOUT_BEST_WEIGHTS)

        current = pd.DataFrame(rows)
        means = current[(current["fold"] == fold["fold"]) & (current["group"] == "mean")]
        print(means.sort_values("score", ascending=False).to_string(index=False))

    results = pd.DataFrame(rows)
    summary = (
        results[results["group"] == "mean"]
        .groupby("candidate", as_index=False)
        .agg(mean_score=("score", "mean"), worst_fold=("score", "min"), best_fold=("score", "max"), std_score=("score", "std"), n_folds=("score", "count"))
        .sort_values("mean_score", ascending=False)
    )
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    return results, summary


if __name__ == "__main__":
    main()

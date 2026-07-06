import argparse

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict

from predict_tree_feature_block import CV_SPLITTER, MODELS, add_feature_block, ensemble_fit_predict
from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_weather_features
from evaluate_pinn_effective_wind_teacher import build_extended_pinn_weather

VAL_START = "2024-01-01 01:00:00"
RESULTS_PATH = "results/tree_feature_block_submission_style_scores.csv"


def split_time(weather, labels, group, val_start):
    weather = weather.copy()
    weather["forecast_kst_dtm"] = pd.to_datetime(weather["forecast_kst_dtm"])
    labels = labels.copy()
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    merged = weather.merge(labels[["kst_dtm", group]], left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner")
    merged = merged.dropna(subset=[group]).reset_index(drop=True)
    is_val = merged["forecast_kst_dtm"] >= pd.Timestamp(val_start)
    feature_cols = [c for c in weather.columns if c not in TIME_KEY_COLS]
    return (
        merged.loc[~is_val, feature_cols].reset_index(drop=True),
        merged.loc[is_val, feature_cols].reset_index(drop=True),
        merged.loc[~is_val, group].reset_index(drop=True),
        merged.loc[is_val, group].reset_index(drop=True),
    )


def score_one(y, pred, capacity):
    pred = np.clip(pred, 0, capacity)
    nmae, ficr = group_nmae_ficr(y, pred, capacity)
    return 0.5 * (1 - nmae) + 0.5 * ficr, nmae, ficr


def evaluate_block(block, weather_base, weather_ext, labels, scada_by_group):
    weather = add_feature_block(weather_base, weather_ext, block)
    per_group = {}
    pooled_oof_pred_pct, pooled_oof_actual_pct = [], []

    for group, capacity in GROUP_CAPACITY_KWH.items():
        curve_fn = fit_group_power_curve(scada_by_group[group], group)
        group_weather = add_power_curve_feature(weather, HUB_HEIGHT_PROXY_COL, curve_fn, GROUP_N_TURBINES[group])
        x_train, x_val, y_train, y_val = split_time(group_weather, labels, group, VAL_START)

        oof = np.zeros(len(y_train))
        for model in MODELS.values():
            oof += cross_val_predict(clone(model), x_train, y_train, cv=CV_SPLITTER, n_jobs=1)
        oof /= len(MODELS)

        raw_val = ensemble_fit_predict(x_train, y_train, x_val)
        per_group[group] = {"y": y_val.to_numpy(), "raw": raw_val, "capacity": capacity}
        pooled_oof_pred_pct.append(oof / capacity)
        pooled_oof_actual_pct.append(y_train.to_numpy() / capacity)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.concatenate(pooled_oof_pred_pct), np.concatenate(pooled_oof_actual_pct))

    rows = []
    for group, data in per_group.items():
        capacity = data["capacity"]
        for variant, pred in {
            "raw": data["raw"],
            "pooled_isotonic": calibrator.predict(data["raw"] / capacity) * capacity,
        }.items():
            score, nmae, ficr = score_one(data["y"], pred, capacity)
            rows.append({"feature_block": block, "variant": variant, "group": group, "score": score, "nmae": nmae, "ficr": ficr})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", default="baseline,lead_ldaps_gfs")
    args = parser.parse_args()

    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    scada_by_group = {
        "kpx_group_1": pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig"),
        "kpx_group_2": pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig"),
        "kpx_group_3": pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig"),
    }
    weather_base = build_weather_features(ldaps, gfs)
    weather_ext = build_extended_pinn_weather(ldaps, gfs)

    rows = []
    for block in [b.strip() for b in args.blocks.split(",") if b.strip()]:
        print(f"\n=== {block} ===")
        block_rows = evaluate_block(block, weather_base, weather_ext, labels, scada_by_group)
        rows.extend(block_rows)
        tmp = pd.DataFrame(block_rows)
        means = tmp.groupby("variant")["score"].mean().sort_values(ascending=False)
        print(means.round(4).to_string())

    results = pd.DataFrame(rows)
    means = results.groupby(["feature_block", "variant"], as_index=False).agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"))
    results = pd.concat([results, means.assign(group="mean")[results.columns]], ignore_index=True)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    print("\n=== summary ===")
    print(means.sort_values("score", ascending=False).to_string(index=False))
    return results


if __name__ == "__main__":
    main()

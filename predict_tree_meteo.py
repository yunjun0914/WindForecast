import argparse

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict

from evaluate_tree_meteo_feature_blocks_fast import add_meteo_block, build_meteo_features
from predict_tree_feature_block import CV_SPLITTER, MODELS, ensemble_fit_predict
from utils.metrics import GROUP_CAPACITY_KWH
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_group_dataset, build_weather_features

DEFAULT_OUTPUT = "results/submission_tree_meteo.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", default="all_meteo", choices=["baseline", "thermo", "radiation_cloud", "all_meteo", "lead_all_meteo"])
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }

    ldaps_test = pd.read_csv("data/test/ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv("data/test/gfs_test.csv", encoding="utf-8-sig")

    weather_train_base = build_weather_features(ldaps_train, gfs_train)
    weather_test_base = build_weather_features(ldaps_test, gfs_test)
    meteo_train = build_meteo_features(ldaps_train, gfs_train)
    meteo_test = build_meteo_features(ldaps_test, gfs_test)
    weather_train = add_meteo_block(weather_train_base, meteo_train, args.block)
    weather_test = add_meteo_block(weather_test_base, meteo_test, args.block)

    pooled_oof_pred_pct, pooled_oof_actual_pct = [], []
    per_group = {}
    for group, capacity in GROUP_CAPACITY_KWH.items():
        curve_fn = fit_group_power_curve(scada_by_group[group], group)
        n_turbines = GROUP_N_TURBINES[group]
        group_weather_train = add_power_curve_feature(weather_train, HUB_HEIGHT_PROXY_COL, curve_fn, n_turbines)
        group_weather_test = add_power_curve_feature(weather_test, HUB_HEIGHT_PROXY_COL, curve_fn, n_turbines)

        x_train, y_train = build_group_dataset(group_weather_train, labels, group)
        feature_cols = [c for c in group_weather_test.columns if c not in TIME_KEY_COLS]
        x_test = group_weather_test[feature_cols]

        oof = np.zeros(len(y_train))
        for model in MODELS.values():
            oof += cross_val_predict(clone(model), x_train, y_train, cv=CV_SPLITTER, n_jobs=1)
        oof /= len(MODELS)

        raw_test = ensemble_fit_predict(x_train, y_train, x_test)
        per_group[group] = {
            "time": pd.to_datetime(group_weather_test["forecast_kst_dtm"]).reset_index(drop=True),
            "raw": raw_test,
            "capacity": capacity,
        }
        pooled_oof_pred_pct.append(oof / capacity)
        pooled_oof_actual_pct.append(y_train.to_numpy() / capacity)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.concatenate(pooled_oof_pred_pct), np.concatenate(pooled_oof_actual_pct))

    prediction = pd.DataFrame({"forecast_kst_dtm": per_group["kpx_group_1"]["time"]})
    for group, data in per_group.items():
        capacity = data["capacity"]
        prediction[group] = np.clip(calibrator.predict(data["raw"] / capacity) * capacity, 0, capacity)

    submission = pd.read_csv("data/sample_submission.csv", encoding="utf-8-sig")
    submission["forecast_kst_dtm"] = pd.to_datetime(submission["forecast_kst_dtm"])
    merged = submission[["forecast_id", "forecast_kst_dtm"]].merge(prediction, on="forecast_kst_dtm", how="left")
    if merged[["kpx_group_1", "kpx_group_2", "kpx_group_3"]].isna().any().any():
        raise ValueError("submission has missing predictions")

    merged = merged[["forecast_id", "forecast_kst_dtm", "kpx_group_1", "kpx_group_2", "kpx_group_3"]]
    merged.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"saved {args.output}: {merged.shape}, block={args.block}")
    print(merged.head())
    print(merged.tail())
    return merged


if __name__ == "__main__":
    main()

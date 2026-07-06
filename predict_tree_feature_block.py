import argparse

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

from evaluate_pinn_effective_wind_teacher import build_extended_pinn_weather
from utils.metrics import GROUP_CAPACITY_KWH
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_group_dataset, build_weather_features

MODELS = {
    "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "lgbm": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
    "xgb": XGBRegressor(random_state=42, n_jobs=-1),
}
CV_SPLITTER = KFold(n_splits=3, shuffle=True, random_state=42)
DEFAULT_OUTPUT = "results/submission_tree_feature_block.csv"


def add_feature_block(weather, weather_ext, block):
    out = weather.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    ext = weather_ext.copy()
    ext["forecast_kst_dtm"] = pd.to_datetime(ext["forecast_kst_dtm"])

    if block == "baseline":
        return out

    cols = ["forecast_kst_dtm"]
    if "lead" in block:
        cols += ["lead_hour"]
    if "ldaps" in block:
        cols += [c for c in ext.columns if c.startswith("ldaps_") and ("grid_" in c or c.endswith("_p75"))]
    if "gfs" in block:
        cols += [c for c in ext.columns if c.startswith("gfs_") and ("grid_" in c or c.endswith("_p75"))]

    cols = [c for c in dict.fromkeys(cols) if c in ext.columns]
    return out.merge(ext[cols], on="forecast_kst_dtm", how="left").sort_values("forecast_kst_dtm").ffill().fillna(0)


def ensemble_fit_predict(x_train, y_train, x_pred):
    preds = []
    for model in MODELS.values():
        fitted = clone(model)
        fitted.fit(x_train, y_train)
        preds.append(fitted.predict(x_pred))
    return sum(preds) / len(preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block", default="lead_ldaps_gfs", choices=["baseline", "lead", "ldaps", "gfs", "lead_ldaps_gfs"])
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
    weather_train_ext = build_extended_pinn_weather(ldaps_train, gfs_train)
    weather_test_ext = build_extended_pinn_weather(ldaps_test, gfs_test)
    weather_train = add_feature_block(weather_train_base, weather_train_ext, args.block)
    weather_test = add_feature_block(weather_test_base, weather_test_ext, args.block)

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

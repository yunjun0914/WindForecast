import argparse

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from predict_tree_compact_physics_v2 import GROUPS, build_all_meteo_compact_v2
from utils.metrics import GROUP_CAPACITY_KWH
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature_oof
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_group_dataset


DEFAULT_OUTPUT = "results/submission_tree_compact_v2_metric_valid_lgbm_mean.csv"


def fit_lgbm_tuned(seed=42):
    return LGBMRegressor(
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        n_estimators=700,
        learning_rate=0.04,
        num_leaves=48,
        min_child_samples=80,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
    )


def fit_lgbm_regularized(seed=43):
    return LGBMRegressor(
        random_state=seed,
        n_jobs=-1,
        verbose=-1,
        n_estimators=900,
        learning_rate=0.025,
        num_leaves=32,
        min_child_samples=150,
        subsample=0.80,
        colsample_bytree=0.75,
        reg_alpha=0.2,
        reg_lambda=4.0,
    )


def predict_group(train_weather, test_weather, labels, scada, group):
    train_weather, test_weather = add_power_curve_feature_oof(
        train_weather,
        test_weather,
        scada,
        group,
        HUB_HEIGHT_PROXY_COL,
        GROUP_N_TURBINES[group],
    )

    x_train, y_train = build_group_dataset(train_weather, labels, group)
    valid_metric_mask = y_train >= GROUP_CAPACITY_KWH[group] * 0.10
    x_train = x_train.loc[valid_metric_mask].reset_index(drop=True)
    y_train = y_train.loc[valid_metric_mask].reset_index(drop=True)

    feature_cols = [c for c in train_weather.columns if c not in TIME_KEY_COLS]
    x_train = x_train.reindex(columns=feature_cols, fill_value=0)
    x_test = test_weather.reindex(columns=feature_cols, fill_value=0)

    preds = []
    for model in [fit_lgbm_tuned(), fit_lgbm_regularized()]:
        model.fit(x_train, y_train)
        preds.append(model.predict(x_test))
    pred = np.mean(np.column_stack(preds), axis=1)
    return np.clip(pred, 0, GROUP_CAPACITY_KWH[group]), len(feature_cols), len(x_train)


def main():
    parser = argparse.ArgumentParser()
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
    submission = pd.read_csv("data/sample_submission.csv", encoding="utf-8-sig")
    submission["forecast_kst_dtm"] = pd.to_datetime(submission["forecast_kst_dtm"])
    prediction = submission[["forecast_id", "forecast_kst_dtm"]].copy()

    for group in GROUPS:
        print(f"build features and fit metric-valid LGBM mean {group}")
        train_weather = build_all_meteo_compact_v2(ldaps_train, gfs_train, group)
        test_weather = build_all_meteo_compact_v2(ldaps_test, gfs_test, group)
        pred, n_features, n_train = predict_group(train_weather, test_weather, labels, scada_by_group[group], group)
        prediction[group] = pred
        print(f"{group}: metric_valid_train_rows={n_train}, features={n_features}, min={pred.min():.2f}, max={pred.max():.2f}")

    prediction = prediction[["forecast_id", "forecast_kst_dtm", *GROUPS]]
    if prediction[GROUPS].isna().any().any():
        raise ValueError("submission has missing predictions")
    prediction.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"saved {args.output}: {prediction.shape}")
    print(prediction[GROUPS].agg(["min", "max", "mean"]).to_string())
    return prediction


if __name__ == "__main__":
    main()

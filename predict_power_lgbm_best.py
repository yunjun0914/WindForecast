import argparse

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from predict_tree_compact_physics_v2 import GROUPS, build_all_meteo_compact_v2
from utils.metrics import GROUP_CAPACITY_KWH
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature_oof
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_group_dataset
from utils.scada_direction_features import SCADA_DIRECTION_TARGETS, add_scada_direction_teacher_oof


DEFAULT_OUTPUT = "results/submission_tree_lgbm_best_v2_l1.csv"
PARAM_COLS = [
    "random_state",
    "n_jobs",
    "verbose",
    "objective",
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "min_split_gain",
]


def clean_params(row):
    params = {col: row[col] for col in PARAM_COLS}
    for col in ["random_state", "n_jobs", "verbose", "n_estimators", "num_leaves", "max_depth", "min_child_samples"]:
        params[col] = int(params[col])
    for col in ["learning_rate", "subsample", "colsample_bytree", "reg_alpha", "reg_lambda", "min_split_gain"]:
        params[col] = float(params[col])
    return params


def sample_weight(y, group, weight_policy):
    if weight_policy == "none":
        return None
    capacity = GROUP_CAPACITY_KWH[group]
    y = np.asarray(y, dtype=float)
    if weight_policy == "metric_x2":
        return 1.0 + 2.0 * (y >= capacity * 0.10)
    if weight_policy == "actual_sqrt":
        return 0.5 + np.sqrt(np.clip(y / capacity, 0, 1))
    raise ValueError(f"unknown weight_policy: {weight_policy}")


def predict_group(train_weather, test_weather, labels, scada, best_row, group, use_scada_wd_teacher=False):
    if use_scada_wd_teacher:
        train_weather, test_weather = add_scada_direction_teacher_oof(train_weather, test_weather, scada, group)
    train_weather, test_weather = add_power_curve_feature_oof(
        train_weather,
        test_weather,
        scada,
        group,
        HUB_HEIGHT_PROXY_COL,
        GROUP_N_TURBINES[group],
    )
    x_train, y_train = build_group_dataset(train_weather, labels, group)
    min_output_ratio = float(best_row["min_output_ratio"])
    mask = y_train.to_numpy(float) >= GROUP_CAPACITY_KWH[group] * min_output_ratio
    x_train = x_train.loc[mask].reset_index(drop=True)
    y_train = y_train.loc[mask].reset_index(drop=True)

    feature_cols = [c for c in train_weather.columns if c not in TIME_KEY_COLS]
    x_train = x_train.reindex(columns=feature_cols, fill_value=0)
    x_test = test_weather.reindex(columns=feature_cols, fill_value=0)

    model = LGBMRegressor(**clean_params(best_row))
    weights = sample_weight(y_train, group, best_row["weight_policy"])
    model.fit(x_train, y_train, sample_weight=weights)
    pred = model.predict(x_test)
    return np.clip(pred, 0, GROUP_CAPACITY_KWH[group]), len(feature_cols), len(x_train)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--best-csv", default="results/power_lgbm_hyperparams_v2_l1_20_best.csv")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--use-scada-wd-teacher", action="store_true")
    args = parser.parse_args()

    best = pd.read_csv(args.best_csv, encoding="utf-8-sig")
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
        row = best[best["group"] == group]
        if row.empty:
            raise ValueError(f"missing best params for {group}")
        print(f"build features and fit tuned LGBM {group}")
        train_weather = build_all_meteo_compact_v2(ldaps_train, gfs_train, group)
        test_weather = build_all_meteo_compact_v2(ldaps_test, gfs_test, group)
        pred, n_features, n_train = predict_group(
            train_weather,
            test_weather,
            labels,
            scada_by_group[group],
            row.iloc[0],
            group,
            use_scada_wd_teacher=args.use_scada_wd_teacher,
        )
        prediction[group] = pred
        print(f"{group}: train_rows={n_train}, features={n_features}, min={pred.min():.2f}, max={pred.max():.2f}")

    prediction = prediction[["forecast_id", "forecast_kst_dtm", *GROUPS]]
    if prediction[GROUPS].isna().any().any():
        raise ValueError("submission has missing predictions")
    prediction.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"saved {args.output}: {prediction.shape}")
    if args.use_scada_wd_teacher:
        print(f"added wd teacher features: {', '.join(SCADA_DIRECTION_TARGETS)}")
    print(prediction[GROUPS].agg(["min", "max", "mean"]).to_string())
    return prediction


if __name__ == "__main__":
    main()

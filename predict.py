import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

from utils.metrics import GROUP_CAPACITY_KWH
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_group_dataset, build_weather_features

# all 3 models use default hyperparameters. A prior RandomizedSearchCV attempt (tuned
# against a shuffled K-fold) improved lgbm/xgb's CV score a lot but regressed on the
# real leaderboard (0.6068 -> 0.6029): random K-fold within 2022-2024 never tests
# generalization to an unseen year, so it rewarded configs that overfit that period.
# A 2024 time-holdout check (evaluate_time_holdout.py) confirms the plain default
# 3-model ensemble beats every tuned single model in 2 of 3 groups and is a near-tie
# in the third, so tuning is parked until it can be redone against that time-based score.
MODELS = {
    "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "lgbm": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
    "xgb": XGBRegressor(random_state=42, n_jobs=-1),
}
CV_SPLITTER = KFold(n_splits=3, shuffle=True, random_state=42)

SUBMISSION_PATH = "results/submission.csv"


def ensemble_fit_predict(X_train, y_train, X_pred):
    preds = []
    for model in MODELS.values():
        model.fit(X_train, y_train)
        preds.append(model.predict(X_pred))
    return sum(preds) / len(preds)


def main():
    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    weather_train = build_weather_features(ldaps_train, gfs_train)
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")

    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada_by_group = {"kpx_group_1": scada_vestas, "kpx_group_2": scada_vestas, "kpx_group_3": scada_unison}

    ldaps_test = pd.read_csv("data/test/ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv("data/test/gfs_test.csv", encoding="utf-8-sig")
    weather_test = build_weather_features(ldaps_test, gfs_test)

    # pass 1: per group, build X/y (train) + X_test, and get OOF predictions on the
    # full training period (2022-2024) to fit the bias-correction calibration curve
    per_group = {}
    pooled_oof_pred_pct, pooled_oof_actual_pct = [], []

    for group, capacity in GROUP_CAPACITY_KWH.items():
        curve_fn = fit_group_power_curve(scada_by_group[group], group)
        n_turbines = GROUP_N_TURBINES[group]

        group_weather_train = add_power_curve_feature(weather_train, HUB_HEIGHT_PROXY_COL, curve_fn, n_turbines)
        group_weather_test = add_power_curve_feature(weather_test, HUB_HEIGHT_PROXY_COL, curve_fn, n_turbines)

        X, y = build_group_dataset(group_weather_train, labels, group)
        feature_cols = [c for c in group_weather_test.columns if c not in TIME_KEY_COLS]
        X_test = group_weather_test[feature_cols]

        oof_pred = np.zeros(len(y))
        for model in MODELS.values():
            oof_pred += cross_val_predict(clone(model), X, y, cv=CV_SPLITTER, n_jobs=1)
        oof_pred /= len(MODELS)

        per_group[group] = {"X": X, "y": y, "X_test": X_test, "weather_test": group_weather_test}
        pooled_oof_pred_pct.append(oof_pred / capacity)
        pooled_oof_actual_pct.append(y.to_numpy() / capacity)

    # one shared calibration curve fit on all 3 groups' OOF data, normalized to % of
    # capacity -- corrects the ensemble's systematic bias (over-predicts low output,
    # under-predicts high-output/peak hours), which is what the FICR settlement metric
    # penalizes hardest. Pooling across groups was validated to beat a per-group-only
    # calibration fit, especially for kpx_group_3 whose own data is comparatively small.
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.concatenate(pooled_oof_pred_pct), np.concatenate(pooled_oof_actual_pct))

    submission = pd.read_csv("data/sample_submission.csv", encoding="utf-8-sig")

    for group, capacity in GROUP_CAPACITY_KWH.items():
        data = per_group[group]
        raw_pred = ensemble_fit_predict(data["X"], data["y"], data["X_test"])
        calibrated_pct = calibrator.predict(raw_pred / capacity)
        calibrated_pred = np.clip(calibrated_pct * capacity, 0, capacity)

        pred_df = pd.DataFrame({"forecast_kst_dtm": data["weather_test"]["forecast_kst_dtm"], group: calibrated_pred})
        submission = submission.drop(columns=[group]).merge(pred_df, on="forecast_kst_dtm", how="left")

    submission = submission[["forecast_id", "forecast_kst_dtm", "kpx_group_1", "kpx_group_2", "kpx_group_3"]]
    submission.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")
    print(f"saved {SUBMISSION_PATH}: {submission.shape}")
    print(submission.head())


if __name__ == "__main__":
    main()

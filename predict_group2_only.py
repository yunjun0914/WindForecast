import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from utils.metrics import GROUP_CAPACITY_KWH
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_group_dataset, build_weather_features

# trained on kpx_group_2 data ONLY -- group_1 and group_3's own labels/SCADA are never
# used anywhere in this pipeline. The single resulting prediction (as % of capacity)
# is copied into all 3 submission columns, scaled by each group's own capacity.
MODELS = [RandomForestRegressor, LGBMRegressor, XGBRegressor]
TRAIN_GROUP = "kpx_group_2"
SUBMISSION_PATH = "results/submission.csv"


def main():
    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    weather_train = build_weather_features(ldaps_train, gfs_train)
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")

    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    curve_fn = fit_group_power_curve(scada_vestas, TRAIN_GROUP)
    n_turbines = GROUP_N_TURBINES[TRAIN_GROUP]

    group2_weather_train = add_power_curve_feature(weather_train, HUB_HEIGHT_PROXY_COL, curve_fn, n_turbines)
    X, y = build_group_dataset(group2_weather_train, labels, TRAIN_GROUP)

    ldaps_test = pd.read_csv("data/test/ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv("data/test/gfs_test.csv", encoding="utf-8-sig")
    weather_test = build_weather_features(ldaps_test, gfs_test)
    group2_weather_test = add_power_curve_feature(weather_test, HUB_HEIGHT_PROXY_COL, curve_fn, n_turbines)
    feature_cols = [c for c in group2_weather_test.columns if c not in TIME_KEY_COLS]
    X_test = group2_weather_test[feature_cols]

    preds = []
    for model_cls in MODELS:
        model = model_cls(random_state=42, n_jobs=-1)
        model.fit(X, y)
        preds.append(model.predict(X_test))

    pred_pct = (sum(preds) / len(preds)) / GROUP_CAPACITY_KWH[TRAIN_GROUP]

    submission = pd.read_csv("data/sample_submission.csv", encoding="utf-8-sig")
    for group, capacity in GROUP_CAPACITY_KWH.items():
        submission[group] = (pred_pct * capacity).clip(0, capacity)

    submission.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")
    print(f"saved {SUBMISSION_PATH}: {submission.shape}")
    print(submission.head())


if __name__ == "__main__":
    main()

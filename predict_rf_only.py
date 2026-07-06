import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from utils.metrics import GROUP_CAPACITY_KWH
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_group_dataset, build_weather_features

SUBMISSION_PATH = "results/submission.csv"


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

    submission = pd.read_csv("data/sample_submission.csv", encoding="utf-8-sig")

    for group, capacity in GROUP_CAPACITY_KWH.items():
        curve_fn = fit_group_power_curve(scada_by_group[group], group)
        n_turbines = GROUP_N_TURBINES[group]

        group_weather_train = add_power_curve_feature(weather_train, HUB_HEIGHT_PROXY_COL, curve_fn, n_turbines)
        group_weather_test = add_power_curve_feature(weather_test, HUB_HEIGHT_PROXY_COL, curve_fn, n_turbines)

        X, y = build_group_dataset(group_weather_train, labels, group)
        feature_cols = [c for c in group_weather_test.columns if c not in TIME_KEY_COLS]
        X_test = group_weather_test[feature_cols]

        model = RandomForestRegressor(random_state=42, n_jobs=-1)
        model.fit(X, y)
        pred = model.predict(X_test).clip(0, capacity)

        pred_df = pd.DataFrame({"forecast_kst_dtm": group_weather_test["forecast_kst_dtm"], group: pred})
        submission = submission.drop(columns=[group]).merge(pred_df, on="forecast_kst_dtm", how="left")

    submission = submission[["forecast_id", "forecast_kst_dtm", "kpx_group_1", "kpx_group_2", "kpx_group_3"]]
    submission.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")
    print(f"saved {SUBMISSION_PATH}: {submission.shape}")
    print(submission.head())


if __name__ == "__main__":
    main()

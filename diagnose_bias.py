import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from utils.metrics import GROUP_CAPACITY_KWH
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_weather_features

MODELS = {
    "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "lgbm": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
    "xgb": XGBRegressor(random_state=42, n_jobs=-1),
}
VAL_START = "2024-01-01 01:00:00"
N_BINS = 10


def time_split(weather_df, labels_df, group):
    merged = weather_df.merge(
        labels_df[["kst_dtm", group]], left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner"
    )
    merged = merged.dropna(subset=[group]).reset_index(drop=True)
    is_val = merged["forecast_kst_dtm"] >= VAL_START
    feature_cols = [c for c in weather_df.columns if c not in TIME_KEY_COLS]
    X_train, X_val = merged.loc[~is_val, feature_cols], merged.loc[is_val, feature_cols]
    y_train, y_val = merged.loc[~is_val, group], merged.loc[is_val, group]
    return X_train, X_val, y_train, y_val


def main():
    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    weather = build_weather_features(ldaps_train, gfs_train)
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")

    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada_by_group = {"kpx_group_1": scada_vestas, "kpx_group_2": scada_vestas, "kpx_group_3": scada_unison}

    for group, capacity in GROUP_CAPACITY_KWH.items():
        curve_fn = fit_group_power_curve(scada_by_group[group], group)
        group_weather = add_power_curve_feature(weather, HUB_HEIGHT_PROXY_COL, curve_fn, GROUP_N_TURBINES[group])
        X_train, X_val, y_train, y_val = time_split(group_weather, labels, group)

        ensemble_pred = np.zeros(len(y_val))
        for model in MODELS.values():
            model.fit(X_train, y_train)
            ensemble_pred += model.predict(X_val)
        ensemble_pred /= len(MODELS)

        df = pd.DataFrame({"actual": y_val.to_numpy(), "pred": ensemble_pred})
        df["actual_pct_capacity"] = df["actual"] / capacity * 100
        df["bias"] = df["pred"] - df["actual"]
        df["bias_pct_capacity"] = df["bias"] / capacity * 100

        df["decile"] = pd.qcut(df["actual"], N_BINS, labels=False, duplicates="drop")
        summary = df.groupby("decile").agg(
            actual_mean=("actual", "mean"),
            actual_pct_capacity_mean=("actual_pct_capacity", "mean"),
            bias_mean=("bias", "mean"),
            bias_pct_capacity_mean=("bias_pct_capacity", "mean"),
            n=("actual", "size"),
        )
        print(f"\n=== {group} (capacity={capacity}) ===")
        print(summary.round(2))


if __name__ == "__main__":
    main()

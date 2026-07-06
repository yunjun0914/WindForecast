import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_weather_features

MODELS = {
    "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "lgbm": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
    "xgb": XGBRegressor(random_state=42, n_jobs=-1),
}
VAL_START = "2024-01-01 01:00:00"


def time_split_with_dtm(weather_df, labels_df, group):
    """Same as the usual time split, but also returns forecast_kst_dtm for the val
    rows so predictions can be joined back to OTHER groups' labels by timestamp
    (each group can have different scattered missing hours, even within 2024)."""
    merged = weather_df.merge(
        labels_df[["kst_dtm", group]], left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner"
    )
    merged = merged.dropna(subset=[group]).reset_index(drop=True)
    is_val = merged["forecast_kst_dtm"] >= VAL_START
    feature_cols = [c for c in weather_df.columns if c not in TIME_KEY_COLS]
    X_train, X_val = merged.loc[~is_val, feature_cols], merged.loc[is_val, feature_cols]
    y_train, y_val = merged.loc[~is_val, group], merged.loc[is_val, group]
    dtm_val = merged.loc[is_val, "forecast_kst_dtm"]
    return X_train, X_val, y_train, y_val, dtm_val


def main():
    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    weather = build_weather_features(ldaps_train, gfs_train)
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")

    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    curve_fn = fit_group_power_curve(scada_vestas, "kpx_group_2")
    group2_weather = add_power_curve_feature(
        weather, HUB_HEIGHT_PROXY_COL, curve_fn, GROUP_N_TURBINES["kpx_group_2"]
    )

    # ONE pipeline, built entirely from group_2's own data (features + label).
    # No group-specific power curves or features are built for group_1/group_3 at all.
    X_train, X_val, y_train, y_val_g2, dtm_val = time_split_with_dtm(group2_weather, labels, "kpx_group_2")

    preds = []
    for model in MODELS.values():
        model.fit(X_train, y_train)
        preds.append(model.predict(X_val))
    group2_pred = sum(preds) / len(preds)
    pred_df = pd.DataFrame({"forecast_kst_dtm": dtm_val.to_numpy(), "pred_pct": group2_pred / GROUP_CAPACITY_KWH["kpx_group_2"]})

    rows = []
    for group, capacity in GROUP_CAPACITY_KWH.items():
        group_labels = labels[["kst_dtm", group]].dropna(subset=[group])
        group_labels = group_labels[group_labels["kst_dtm"] >= VAL_START]

        aligned = pred_df.merge(group_labels, left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner")
        pred = np.clip(aligned["pred_pct"].to_numpy() * capacity, 0, capacity)
        actual = aligned[group].to_numpy()

        nmae, ficr = group_nmae_ficr(actual, pred, capacity)
        score = 0.5 * (1 - nmae) + 0.5 * ficr
        rows.append({"group": group, "score": score, "nmae": nmae, "ficr": ficr, "n": len(aligned)})
        print(f"{group}: score={score:.4f} (nmae={nmae:.4f}, ficr={ficr:.4f}, n={len(aligned)})")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()

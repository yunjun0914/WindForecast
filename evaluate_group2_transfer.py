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


def ensemble_fit_predict(X_train, y_train, X_pred):
    preds = []
    for model in MODELS.values():
        model.fit(X_train, y_train)
        preds.append(model.predict(X_pred))
    return sum(preds) / len(preds)


def main():
    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    weather = build_weather_features(ldaps_train, gfs_train)
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")

    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada_by_group = {"kpx_group_1": scada_vestas, "kpx_group_2": scada_vestas, "kpx_group_3": scada_unison}

    # build each group's own (X_train, X_val, y_train, y_val), using that group's own
    # power_curve_est feature (fit from that group's own SCADA)
    splits = {}
    for group in GROUP_CAPACITY_KWH:
        curve_fn = fit_group_power_curve(scada_by_group[group], group)
        group_weather = add_power_curve_feature(weather, HUB_HEIGHT_PROXY_COL, curve_fn, GROUP_N_TURBINES[group])
        splits[group] = time_split(group_weather, labels, group)

    # train ONE ensemble on group_2's own data only
    x2_train, _, y2_train, _ = splits["kpx_group_2"]
    group2_capacity = GROUP_CAPACITY_KWH["kpx_group_2"]

    rows = []
    for group, capacity in GROUP_CAPACITY_KWH.items():
        X_train, X_val, y_train, y_val = splits[group]

        own_pred = ensemble_fit_predict(X_train, y_train, X_val)
        own_nmae, own_ficr = group_nmae_ficr(y_val, own_pred, capacity)
        own_score = 0.5 * (1 - own_nmae) + 0.5 * own_ficr
        rows.append({"group": group, "variant": "own_model", "score": own_score, "nmae": own_nmae, "ficr": own_ficr})

        if group == "kpx_group_2":
            transfer_pred = own_pred  # group_2 predicting itself, for reference
        else:
            group2_pred_pct = ensemble_fit_predict(x2_train, y2_train, X_val) / group2_capacity
            transfer_pred = np.clip(group2_pred_pct * capacity, 0, capacity)

        t_nmae, t_ficr = group_nmae_ficr(y_val, transfer_pred, capacity)
        t_score = 0.5 * (1 - t_nmae) + 0.5 * t_ficr
        rows.append(
            {"group": group, "variant": "group2_transfer", "score": t_score, "nmae": t_nmae, "ficr": t_ficr}
        )
        print(f"{group}/own_model: score={own_score:.4f} (nmae={own_nmae:.4f}, ficr={own_ficr:.4f})")
        print(f"{group}/group2_transfer: score={t_score:.4f} (nmae={t_nmae:.4f}, ficr={t_ficr:.4f})")

    results_df = pd.DataFrame(rows)
    print()
    print(results_df.pivot(index="group", columns="variant", values="score").round(4))
    return results_df


if __name__ == "__main__":
    main()

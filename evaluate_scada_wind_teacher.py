import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

from utils.pinn_data import build_pinn_weather
from utils.power_curve import GROUP_TURBINE_PREFIXES

VAL_START = "2024-01-01 01:00:00"


def scada_group_wind_teacher(scada_df, group):
    scada = scada_df.copy()
    scada["kst_dtm"] = pd.to_datetime(scada["kst_dtm"])
    scada["forecast_kst_dtm"] = scada["kst_dtm"].dt.floor("h")
    ws_cols = [f"{prefix}_ws" for prefix in GROUP_TURBINE_PREFIXES[group]]
    rows = scada.groupby("forecast_kst_dtm")[ws_cols].agg(["mean"])
    # Flatten back to the original turbine columns after hourly mean over 10-min rows.
    hourly = scada.groupby("forecast_kst_dtm")[ws_cols].mean()
    values = hourly.to_numpy(dtype=float)
    out = pd.DataFrame({"forecast_kst_dtm": hourly.index})
    out["scada_ws_mean"] = np.nanmean(values, axis=1)
    out["scada_ws_std"] = np.nanstd(values, axis=1)
    out["scada_ws_p10"] = np.nanpercentile(values, 10, axis=1)
    out["scada_ws_p50"] = np.nanpercentile(values, 50, axis=1)
    out["scada_ws_p90"] = np.nanpercentile(values, 90, axis=1)
    return out.reset_index(drop=True)


def main():
    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    weather = build_pinn_weather(ldaps, gfs)

    scada_by_group = {
        "kpx_group_1": pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig"),
        "kpx_group_2": pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig"),
        "kpx_group_3": pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig"),
    }

    feature_cols = [
        c
        for c in weather.columns
        if c not in ["forecast_kst_dtm", "doy", "moy", "hod"]
    ]
    target_cols = ["scada_ws_mean", "scada_ws_std", "scada_ws_p10", "scada_ws_p50", "scada_ws_p90"]

    rows = []
    for group, scada in scada_by_group.items():
        teacher = scada_group_wind_teacher(scada, group)
        df = weather.merge(teacher, on="forecast_kst_dtm", how="inner").dropna()
        is_val = df["forecast_kst_dtm"] >= VAL_START
        train_df, val_df = df.loc[~is_val], df.loc[is_val]
        if len(train_df) == 0 or len(val_df) == 0:
            print(group, "skipped: insufficient train/val rows", len(train_df), len(val_df))
            continue

        x_train, y_train = train_df[feature_cols], train_df[target_cols]
        x_val, y_val = val_df[feature_cols], val_df[target_cols]
        model = MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=300,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )
        )
        model.fit(x_train, y_train)
        pred = model.predict(x_val)

        print(f"\n{group}: train={len(train_df)} val={len(val_df)}")
        for i, target in enumerate(target_cols):
            mae = mean_absolute_error(y_val.iloc[:, i], pred[:, i])
            r2 = r2_score(y_val.iloc[:, i], pred[:, i])
            rows.append({"group": group, "target": target, "mae": mae, "r2": r2})
            print(f"{target}: mae={mae:.3f} r2={r2:.3f}")

    results = pd.DataFrame(rows)
    results.to_csv("results/scada_wind_teacher_scores.csv", index=False, encoding="utf-8-sig")
    print("\nSaved results/scada_wind_teacher_scores.csv")


if __name__ == "__main__":
    main()

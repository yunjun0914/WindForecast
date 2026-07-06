import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

from evaluate_pinn_effective_wind_teacher import (
    EXT_TARGETS,
    build_extended_pinn_weather,
    build_extended_scada_targets,
    extended_feature_cols,
)
from utils.pinn_data import WIND_CORRECTION_FEATURES, build_pinn_weather, build_scada_wind_teacher

RESULTS_PATH = "results/scada_teacher_effective_target_scores.csv"
GROUP_SCADA = {
    "kpx_group_1": "vestas",
    "kpx_group_2": "vestas",
    "kpx_group_3": "unison",
}


def split_fit_eval(weather, targets, feature_cols, target_cols, val_start, val_end, model_kind):
    df = weather.merge(targets, on="forecast_kst_dtm", how="inner").dropna()
    train = df[df["forecast_kst_dtm"] < pd.Timestamp(val_start)]
    val = df[(df["forecast_kst_dtm"] >= pd.Timestamp(val_start)) & (df["forecast_kst_dtm"] < pd.Timestamp(val_end))]
    if len(train) < 500 or len(val) < 200:
        return []

    if model_kind != "rf":
        raise ValueError(model_kind)
    base = RandomForestRegressor(n_estimators=60, min_samples_leaf=12, random_state=42, n_jobs=-1)

    model = MultiOutputRegressor(base)
    model.fit(train[feature_cols], train[target_cols])
    pred = pd.DataFrame(model.predict(val[feature_cols]), columns=target_cols)

    rows = []
    for col in target_cols:
        rows.append(
            {
                "target": col,
                "r2": r2_score(val[col], pred[col]),
                "mae": mean_absolute_error(val[col], pred[col]),
                "n_train": len(train),
                "n_val": len(val),
            }
        )
    return rows


def main():
    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    scada_by_name = {
        "vestas": pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig"),
        "unison": pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig"),
    }

    weather_base = build_pinn_weather(ldaps, gfs)
    weather_ext = build_extended_pinn_weather(ldaps, gfs)
    base_feature_cols = [col for col in WIND_CORRECTION_FEATURES if col in weather_base.columns]
    ext_feature_cols = extended_feature_cols(weather_ext)

    folds = [
        ("year_2023", "2023-01-01 01:00:00", "2024-01-01 01:00:00"),
        ("year_2024", "2024-01-01 01:00:00", "2025-01-01 01:00:00"),
    ]

    rows = []
    for group, scada_name in GROUP_SCADA.items():
        scada = scada_by_name[scada_name]
        base_targets = build_scada_wind_teacher(scada, group)
        ext_targets = build_extended_scada_targets(scada, group)
        for fold, val_start, val_end in folds:
            for model_kind in ["rf"]:
                for row in split_fit_eval(
                    weather_base,
                    base_targets,
                    base_feature_cols,
                    ["scada_ws_mean", "scada_ws_p90"],
                    val_start,
                    val_end,
                    model_kind,
                ):
                    rows.append({"group": group, "fold": fold, "feature_set": "base", "model": model_kind, **row})
                for row in split_fit_eval(
                    weather_ext,
                    ext_targets,
                    ext_feature_cols,
                    ["scada_ws_mean", "scada_ws_p75", "scada_ws_p90", "scada_ws_cubic"],
                    val_start,
                    val_end,
                    model_kind,
                ):
                    rows.append({"group": group, "fold": fold, "feature_set": "extended", "model": model_kind, **row})

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    focus = results[results["target"].isin(["scada_ws_mean", "scada_ws_p75", "scada_ws_p90", "scada_ws_cubic"])]
    print(focus.sort_values(["fold", "group", "target", "r2"], ascending=[True, True, True, False]).to_string(index=False))
    print("\n=== Mean R2 by fold/feature/model/target ===")
    print(
        focus.groupby(["fold", "feature_set", "model", "target"])["r2"]
        .mean()
        .reset_index()
        .sort_values(["fold", "target", "r2"], ascending=[True, True, False])
        .to_string(index=False)
    )
    return results


if __name__ == "__main__":
    main()

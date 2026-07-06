import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

from evaluate_pinn_effective_wind_teacher import (
    build_extended_pinn_weather,
    build_extended_scada_targets,
    extended_feature_cols,
)
from evaluate_scada_teacher_effective_targets import GROUP_SCADA
from evaluate_tree_meteo_feature_blocks_fast import add_meteo_block, build_meteo_features

RESULTS_PATH = "results/scada_teacher_model_variant_scores.csv"
TARGET_COLS = ["scada_ws_mean", "scada_ws_p75", "scada_ws_p90", "scada_ws_cubic"]
FOLDS = [
    ("year_2023", "2023-01-01 01:00:00", "2024-01-01 01:00:00"),
    ("year_2024", "2024-01-01 01:00:00", "2025-01-01 01:00:00"),
]


def make_model(model_kind):
    if model_kind == "rf":
        return MultiOutputRegressor(
            RandomForestRegressor(n_estimators=120, min_samples_leaf=10, random_state=42, n_jobs=-1)
        )
    if model_kind == "extra_trees":
        return MultiOutputRegressor(
            ExtraTreesRegressor(n_estimators=300, min_samples_leaf=10, random_state=42, n_jobs=-1)
        )
    if model_kind == "hist_gbr":
        return MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, l2_regularization=0.01, random_state=42)
        )
    raise ValueError(model_kind)


def fit_predict_one(df, feature_cols, target_cols, val_start, val_end, model_kind):
    train = df[df["forecast_kst_dtm"] < pd.Timestamp(val_start)]
    val = df[(df["forecast_kst_dtm"] >= pd.Timestamp(val_start)) & (df["forecast_kst_dtm"] < pd.Timestamp(val_end))]
    if len(train) < 500 or len(val) < 200:
        return None, val
    model = make_model(model_kind)
    model.fit(train[feature_cols], train[target_cols])
    pred = pd.DataFrame(model.predict(val[feature_cols]), columns=target_cols, index=val.index)
    return pred, val


def score_predictions(pred, val, model_kind, group, fold):
    rows = []
    for target in TARGET_COLS:
        rows.append(
            {
                "group": group,
                "fold": fold,
                "model": model_kind,
                "target": target,
                "r2": r2_score(val[target], pred[target]),
                "mae": mean_absolute_error(val[target], pred[target]),
                "n_train": np.nan,
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

    weather = build_extended_pinn_weather(ldaps, gfs)
    meteo = build_meteo_features(ldaps, gfs)
    weather = add_meteo_block(weather, meteo, "all_meteo")
    feature_cols = extended_feature_cols(weather)

    rows = []
    for group, scada_name in GROUP_SCADA.items():
        targets = build_extended_scada_targets(scada_by_name[scada_name], group)
        df = weather.merge(targets, on="forecast_kst_dtm", how="inner").dropna()
        for fold, val_start, val_end in FOLDS:
            preds = {}
            val_ref = None
            for model_kind in ["rf", "extra_trees", "hist_gbr"]:
                pred, val = fit_predict_one(df, feature_cols, TARGET_COLS, val_start, val_end, model_kind)
                if pred is None:
                    continue
                val_ref = val
                preds[model_kind] = pred
                rows.extend(score_predictions(pred, val, model_kind, group, fold))
            if "rf" in preds and "extra_trees" in preds:
                avg_pred = 0.5 * preds["rf"] + 0.5 * preds["extra_trees"]
                rows.extend(score_predictions(avg_pred, val_ref, "rf_extra_trees_avg", group, fold))
            if "rf" in preds and "hist_gbr" in preds:
                avg_pred = 0.5 * preds["rf"] + 0.5 * preds["hist_gbr"]
                rows.extend(score_predictions(avg_pred, val_ref, "rf_hist_gbr_avg", group, fold))

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    summary = (
        results.groupby(["fold", "model", "target"], as_index=False)
        .agg(r2=("r2", "mean"), mae=("mae", "mean"), n_val=("n_val", "sum"))
        .sort_values(["fold", "target", "r2"], ascending=[True, True, False])
    )
    print(summary.to_string(index=False))
    print("\n=== Mean across targets ===")
    print(
        summary.groupby(["fold", "model"], as_index=False)
        .agg(r2=("r2", "mean"), mae=("mae", "mean"))
        .sort_values(["fold", "r2"], ascending=[True, False])
        .to_string(index=False)
    )
    return results


if __name__ == "__main__":
    main()

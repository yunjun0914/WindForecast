import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

from evaluate_pinn_effective_wind_teacher import build_extended_pinn_weather
from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_weather_features

MODELS = {
    "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "lgbm": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
    "xgb": XGBRegressor(random_state=42, n_jobs=-1),
}
CV_SPLITTER = KFold(n_splits=3, shuffle=True, random_state=42)
RESULTS_PATH = "results/tree_feature_block_scores.csv"
SUMMARY_PATH = "results/tree_feature_block_summary.csv"

FOLDS = [
    {"fold": "year_2023", "fold_type": "yearly", "val_start": "2023-01-01 01:00:00", "val_end": "2024-01-01 01:00:00"},
    {"fold": "year_2024", "fold_type": "yearly", "val_start": "2024-01-01 01:00:00", "val_end": "2025-01-01 01:00:00"},
    {"fold": "q2024_1", "fold_type": "quarter", "val_start": "2024-01-01 01:00:00", "val_end": "2024-04-01 01:00:00"},
    {"fold": "q2024_2", "fold_type": "quarter", "val_start": "2024-04-01 01:00:00", "val_end": "2024-07-01 01:00:00"},
    {"fold": "q2024_3", "fold_type": "quarter", "val_start": "2024-07-01 01:00:00", "val_end": "2024-10-01 01:00:00"},
    {"fold": "q2024_4", "fold_type": "quarter", "val_start": "2024-10-01 01:00:00", "val_end": "2025-01-01 01:00:00"},
]
MIN_TRAIN_ROWS = 1000
MIN_VAL_ROWS = 200


def score_one(y, pred, capacity):
    nmae, ficr = group_nmae_ficr(y, np.clip(pred, 0, capacity), capacity)
    return 0.5 * (1 - nmae) + 0.5 * ficr, nmae, ficr


def split_fold(weather, labels, group, val_start, val_end):
    weather = weather.copy()
    weather["forecast_kst_dtm"] = pd.to_datetime(weather["forecast_kst_dtm"])
    labels = labels.copy()
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    merged = weather.merge(labels[["kst_dtm", group]], left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner")
    merged = merged.dropna(subset=[group]).reset_index(drop=True)
    val_start = pd.Timestamp(val_start)
    val_end = pd.Timestamp(val_end)
    train_mask = merged["forecast_kst_dtm"] < val_start
    val_mask = (merged["forecast_kst_dtm"] >= val_start) & (merged["forecast_kst_dtm"] < val_end)
    feature_cols = [c for c in weather.columns if c not in TIME_KEY_COLS]
    return (
        merged.loc[train_mask, feature_cols].reset_index(drop=True),
        merged.loc[val_mask, feature_cols].reset_index(drop=True),
        merged.loc[train_mask, group].reset_index(drop=True),
        merged.loc[val_mask, group].reset_index(drop=True),
    )


def add_feature_block(weather, weather_ext, block):
    out = weather.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    ext = weather_ext.copy()
    ext["forecast_kst_dtm"] = pd.to_datetime(ext["forecast_kst_dtm"])

    if block == "baseline":
        return out

    cols = ["forecast_kst_dtm"]
    if "lead" in block:
        cols += ["lead_hour"]
    if "ldaps" in block:
        cols += [c for c in ext.columns if c.startswith("ldaps_") and ("grid_" in c or c.endswith("_p75"))]
    if "gfs" in block:
        cols += [c for c in ext.columns if c.startswith("gfs_") and ("grid_" in c or c.endswith("_p75"))]

    cols = [c for c in dict.fromkeys(cols) if c in ext.columns]
    return out.merge(ext[cols], on="forecast_kst_dtm", how="left").ffill().fillna(0)


def fit_group_power_curve_before(scada_df, group, val_start):
    scada = scada_df.copy()
    scada["kst_dtm"] = pd.to_datetime(scada["kst_dtm"])
    scada = scada[scada["kst_dtm"] < pd.Timestamp(val_start)].reset_index(drop=True)
    if len(scada) == 0:
        return None
    return fit_group_power_curve(scada, group)


def evaluate_fold_block(fold, block, weather_base, weather_ext, labels, scada_by_group):
    rows = []
    pooled_pred_pct, pooled_actual_pct, per_group = [], [], {}
    for group, capacity in GROUP_CAPACITY_KWH.items():
        curve = fit_group_power_curve_before(scada_by_group[group], group, fold["val_start"])
        if curve is None:
            continue
        weather = add_power_curve_feature(weather_base, HUB_HEIGHT_PROXY_COL, curve, GROUP_N_TURBINES[group])
        weather = add_feature_block(weather, weather_ext, block)
        x_train, x_val, y_train, y_val = split_fold(weather, labels, group, fold["val_start"], fold["val_end"])
        if len(x_train) < MIN_TRAIN_ROWS or len(x_val) < MIN_VAL_ROWS:
            continue

        preds = {}
        oof = np.zeros(len(y_train))
        for model_name, model in MODELS.items():
            fitted = clone(model)
            fitted.fit(x_train, y_train)
            preds[model_name] = fitted.predict(x_val)
            oof += cross_val_predict(clone(model), x_train, y_train, cv=CV_SPLITTER, n_jobs=1)
        oof /= len(MODELS)

        per_group[group] = {"y_val": y_val.to_numpy(), "preds": preds, "capacity": capacity}
        pooled_pred_pct.append(oof / capacity)
        pooled_actual_pct.append(y_train.to_numpy() / capacity)

    calibrator = None
    if pooled_pred_pct:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(np.concatenate(pooled_pred_pct), np.concatenate(pooled_actual_pct))

    for group, data in per_group.items():
        y_val = data["y_val"]
        capacity = data["capacity"]
        preds = data["preds"]
        for model_name, pred in preds.items():
            s, nmae, ficr = score_one(y_val, pred, capacity)
            rows.append({**fold, "feature_block": block, "group": group, "model": model_name, "score": s, "nmae": nmae, "ficr": ficr, "n": len(y_val)})
        ens = sum(preds.values()) / len(preds)
        s, nmae, ficr = score_one(y_val, ens, capacity)
        rows.append({**fold, "feature_block": block, "group": group, "model": "ensemble_raw", "score": s, "nmae": nmae, "ficr": ficr, "n": len(y_val)})
        if calibrator is not None:
            cal = np.clip(calibrator.predict(ens / capacity) * capacity, 0, capacity)
            s, nmae, ficr = score_one(y_val, cal, capacity)
            rows.append({**fold, "feature_block": block, "group": group, "model": "ensemble_calibrated", "score": s, "nmae": nmae, "ficr": ficr, "n": len(y_val)})
    return rows


def add_mean_rows(results):
    means = (
        results.groupby(["fold", "fold_type", "val_start", "val_end", "feature_block", "model"], as_index=False)
        .agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"), n=("n", "sum"))
    )
    means["group"] = "mean"
    return pd.concat([results, means[results.columns]], ignore_index=True)


def main():
    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }
    weather_base = build_weather_features(ldaps, gfs)
    weather_ext = build_extended_pinn_weather(ldaps, gfs)

    blocks = ["baseline", "lead", "ldaps", "gfs", "lead_ldaps_gfs"]
    all_rows = []
    for fold in FOLDS:
        print(f"\n=== {fold['fold']} ===")
        for block in blocks:
            rows = evaluate_fold_block(fold, block, weather_base, weather_ext, labels, scada_by_group)
            all_rows.extend(rows)
            if rows:
                tmp = pd.DataFrame(rows)
                mean = tmp.groupby("model")["score"].mean().sort_values(ascending=False).head(3)
                print(f"{block}: {mean.round(4).to_dict()}")

    results = add_mean_rows(pd.DataFrame(all_rows))
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    summary = (
        results[results["group"] == "mean"]
        .groupby(["fold_type", "feature_block", "model"], as_index=False)
        .agg(mean_score=("score", "mean"), worst_fold=("score", "min"), std_score=("score", "std"), n_folds=("score", "count"))
        .sort_values(["fold_type", "mean_score"], ascending=[True, False])
    )
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    return results, summary


if __name__ == "__main__":
    main()

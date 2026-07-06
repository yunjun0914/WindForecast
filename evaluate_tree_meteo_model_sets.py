import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

from evaluate_tree_meteo_feature_blocks_fast import add_meteo_block, build_meteo_features
from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_weather_features

VAL_START = "2024-01-01 01:00:00"
RESULTS_PATH = "results/tree_meteo_model_set_scores.csv"
CV_SPLITTER = KFold(n_splits=3, shuffle=True, random_state=42)

MODEL_SETS = {
    "rf_lgbm_xgb": {
        "rf": RandomForestRegressor(random_state=42, n_jobs=-1),
        "lgbm": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        "xgb": XGBRegressor(random_state=42, n_jobs=-1),
    },
    "extra_only": {
        "extra": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
    },
    "rf_lgbm_xgb_extra": {
        "rf": RandomForestRegressor(random_state=42, n_jobs=-1),
        "lgbm": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        "xgb": XGBRegressor(random_state=42, n_jobs=-1),
        "extra": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
    },
    "lgbm_xgb_extra": {
        "lgbm": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
        "xgb": XGBRegressor(random_state=42, n_jobs=-1),
        "extra": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
    },
}


def split_time(weather_df, labels_df, group):
    weather = weather_df.copy()
    weather["forecast_kst_dtm"] = pd.to_datetime(weather["forecast_kst_dtm"])
    labels = labels_df[["kst_dtm", group]].copy()
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    merged = weather.merge(labels, left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner")
    merged = merged.dropna(subset=[group]).reset_index(drop=True)
    is_val = merged["forecast_kst_dtm"] >= pd.Timestamp(VAL_START)
    feature_cols = [c for c in weather.columns if c not in TIME_KEY_COLS]
    return (
        merged.loc[~is_val, feature_cols].reset_index(drop=True),
        merged.loc[is_val, feature_cols].reset_index(drop=True),
        merged.loc[~is_val, group].reset_index(drop=True),
        merged.loc[is_val, group].reset_index(drop=True),
    )


def score_one(y, pred, capacity):
    pred = np.clip(pred, 0, capacity)
    nmae, ficr = group_nmae_ficr(y, pred, capacity)
    return 0.5 * (1 - nmae) + 0.5 * ficr, nmae, ficr


def fit_predict(models, x_train, y_train, x_pred):
    preds = []
    for model in models.values():
        fitted = clone(model)
        fitted.fit(x_train, y_train)
        preds.append(fitted.predict(x_pred))
    return sum(preds) / len(preds)


def evaluate_model_set(model_set_name, models, weather, labels, scada_by_group):
    per_group = {}
    pooled_oof_pred_pct, pooled_oof_actual_pct = [], []

    for group, capacity in GROUP_CAPACITY_KWH.items():
        scada = scada_by_group[group].copy()
        scada["kst_dtm"] = pd.to_datetime(scada["kst_dtm"])
        scada = scada[scada["kst_dtm"] < pd.Timestamp(VAL_START)].reset_index(drop=True)
        curve_fn = fit_group_power_curve(scada, group)
        group_weather = add_power_curve_feature(weather, HUB_HEIGHT_PROXY_COL, curve_fn, GROUP_N_TURBINES[group])
        x_train, x_val, y_train, y_val = split_time(group_weather, labels, group)

        oof = np.zeros(len(y_train))
        for model in models.values():
            oof += cross_val_predict(clone(model), x_train, y_train, cv=CV_SPLITTER, n_jobs=1)
        oof /= len(models)
        raw = fit_predict(models, x_train, y_train, x_val)
        per_group[group] = {"y": y_val.to_numpy(), "raw": raw, "capacity": capacity}
        pooled_oof_pred_pct.append(oof / capacity)
        pooled_oof_actual_pct.append(y_train.to_numpy() / capacity)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.concatenate(pooled_oof_pred_pct), np.concatenate(pooled_oof_actual_pct))

    rows = []
    for group, data in per_group.items():
        capacity = data["capacity"]
        for variant, pred in {
            "raw": data["raw"],
            "pooled_isotonic": calibrator.predict(data["raw"] / capacity) * capacity,
        }.items():
            score, nmae, ficr = score_one(data["y"], pred, capacity)
            rows.append(
                {
                    "model_set": model_set_name,
                    "variant": variant,
                    "group": group,
                    "score": score,
                    "nmae": nmae,
                    "ficr": ficr,
                }
            )
    return rows


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
    weather = add_meteo_block(build_weather_features(ldaps, gfs), build_meteo_features(ldaps, gfs), "all_meteo")

    rows = []
    for model_set_name, models in MODEL_SETS.items():
        print(f"\n=== {model_set_name} ===")
        current = evaluate_model_set(model_set_name, models, weather, labels, scada_by_group)
        rows.extend(current)
        tmp = pd.DataFrame(current)
        means = tmp.groupby("variant")["score"].mean().sort_values(ascending=False)
        print(means.round(4).to_string())

    results = pd.DataFrame(rows)
    means = (
        results.groupby(["model_set", "variant"], as_index=False)
        .agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"))
        .sort_values("score", ascending=False)
    )
    results = pd.concat([results, means.assign(group="mean")[results.columns]], ignore_index=True)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    print("\n=== summary ===")
    print(means.to_string(index=False))
    return results


if __name__ == "__main__":
    main()

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
from utils.spatial_features import add_group_spatial_features


VAL_START = "2024-01-01 01:00:00"
RESULTS_PATH = "results/tree_spatial_recipe_model_set_scores.csv"
CV_SPLITTER = KFold(n_splits=3, shuffle=True, random_state=42)

MODELS = {
    "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "lgbm": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
    "xgb": XGBRegressor(random_state=42, n_jobs=-1),
    "extra_trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
}

RECIPES = {
    "none": {"kpx_group_1": "none", "kpx_group_2": "none", "kpx_group_3": "none"},
    "g2_all": {"kpx_group_1": "none", "kpx_group_2": "all", "kpx_group_3": "none"},
    "g2_all_g3_weighted": {"kpx_group_1": "none", "kpx_group_2": "all", "kpx_group_3": "weighted"},
    "g1_gfs_g2_all_g3_none": {"kpx_group_1": "gfs", "kpx_group_2": "all", "kpx_group_3": "none"},
    "g1_gfs_g2_all_g3_weighted": {"kpx_group_1": "gfs", "kpx_group_2": "all", "kpx_group_3": "weighted"},
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


def fit_predict(x_train, y_train, x_pred):
    preds = []
    for model in MODELS.values():
        fitted = clone(model)
        fitted.fit(x_train, y_train)
        preds.append(fitted.predict(x_pred))
    return sum(preds) / len(preds)


def evaluate_recipe(recipe_name, recipe, weather_cache, labels, scada_by_group):
    per_group = {}
    pooled_oof_pred_pct, pooled_oof_actual_pct = [], []
    for group, capacity in GROUP_CAPACITY_KWH.items():
        scada = scada_by_group[group].copy()
        scada["kst_dtm"] = pd.to_datetime(scada["kst_dtm"])
        scada = scada[scada["kst_dtm"] < pd.Timestamp(VAL_START)].reset_index(drop=True)
        curve_fn = fit_group_power_curve(scada, group)
        weather = add_power_curve_feature(weather_cache[group][recipe[group]], HUB_HEIGHT_PROXY_COL, curve_fn, GROUP_N_TURBINES[group])
        x_train, x_val, y_train, y_val = split_time(weather, labels, group)

        oof = np.zeros(len(y_train))
        for model in MODELS.values():
            oof += cross_val_predict(clone(model), x_train, y_train, cv=CV_SPLITTER, n_jobs=1)
        oof /= len(MODELS)
        raw = fit_predict(x_train, y_train, x_val)
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
            rows.append({"recipe": recipe_name, "variant": variant, "group": group, "score": score, "nmae": nmae, "ficr": ficr})
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
    weather_base = add_meteo_block(build_weather_features(ldaps, gfs), build_meteo_features(ldaps, gfs), "all_meteo")

    modes = sorted({mode for recipe in RECIPES.values() for mode in recipe.values()})
    weather_cache = {group: {"none": weather_base} for group in GROUP_CAPACITY_KWH}
    for group in GROUP_CAPACITY_KWH:
        for mode in modes:
            if mode == "none":
                continue
            print(f"build spatial {group} {mode}")
            weather_cache[group][mode] = add_group_spatial_features(weather_base, ldaps, gfs, group, mode)

    rows = []
    for recipe_name, recipe in RECIPES.items():
        print(f"\n=== {recipe_name} ===")
        current = evaluate_recipe(recipe_name, recipe, weather_cache, labels, scada_by_group)
        rows.extend(current)
        tmp = pd.DataFrame(current)
        print(tmp.groupby("variant")["score"].mean().round(4).to_string())

    results = pd.DataFrame(rows)
    means = (
        results.groupby(["recipe", "variant"], as_index=False)
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

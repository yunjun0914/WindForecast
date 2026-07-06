import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, cross_val_predict

from evaluate_pinn_effective_wind_teacher import EXT_TARGETS, apply_extended_teacher, build_extended_pinn_weather, fit_extended_teacher
from evaluate_pinn_meteo_teacher_model_blend import apply_teacher as apply_blend_teacher
from evaluate_pinn_meteo_teacher_model_blend import fit_teacher as fit_blend_teacher
from evaluate_tree_meteo_feature_blocks_fast import add_meteo_block, build_meteo_features
from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_weather_features


VAL_START = "2024-01-01 01:00:00"
RESULTS_PATH = "results/tree_hist_scada_teacher_scores.csv"
CV_SPLITTER = KFold(n_splits=3, shuffle=True, random_state=42)
HIST = HistGradientBoostingRegressor(max_iter=350, learning_rate=0.04, l2_regularization=0.02, random_state=42)
VARIANTS = ["none", "rf_cubic", "rf_p90", "rf_hist_cubic", "rf_hist_p90"]


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


def add_teacher_features(weather_tree, weather_teacher, scada_df, group, variant):
    if variant == "none":
        return weather_tree
    if variant.startswith("rf_hist"):
        mode = "cubic" if variant.endswith("cubic") else "p90"
        teacher = fit_blend_teacher(weather_teacher, scada_df, group, "rf_hist_gbr_avg", fit_before=VAL_START)
        pred_weather = apply_blend_teacher(weather_teacher, teacher, mode)
    else:
        mode = "cubic" if variant.endswith("cubic") else "p90"
        teacher = fit_extended_teacher(weather_teacher, scada_df, group, fit_before=VAL_START)
        pred_weather = apply_extended_teacher(weather_teacher, teacher, mode)

    add = pred_weather[["forecast_kst_dtm", "v", "v_std"] + EXT_TARGETS].copy()
    add = add.rename(columns={col: f"pred_{variant}_{col}" for col in ["v", "v_std"] + EXT_TARGETS})
    add[f"pred_{variant}_iqr"] = add[f"pred_{variant}_scada_ws_p90"] - add[f"pred_{variant}_scada_ws_p10"]
    add[f"pred_{variant}_p90_minus_mean"] = add[f"pred_{variant}_scada_ws_p90"] - add[f"pred_{variant}_scada_ws_mean"]
    out = weather_tree.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    return out.merge(add, on="forecast_kst_dtm", how="left").sort_values("forecast_kst_dtm").ffill().fillna(0)


def score_one(y, pred, capacity):
    pred = np.clip(pred, 0, capacity)
    nmae, ficr = group_nmae_ficr(y, pred, capacity)
    return 0.5 * (1 - nmae) + 0.5 * ficr, nmae, ficr


def evaluate_variant(variant, weather_tree, weather_teacher, labels, scada_by_group):
    per_group = {}
    pooled_oof_pred_pct, pooled_oof_actual_pct = [], []
    for group, capacity in GROUP_CAPACITY_KWH.items():
        scada = scada_by_group[group].copy()
        scada["kst_dtm"] = pd.to_datetime(scada["kst_dtm"])
        scada_fit = scada[scada["kst_dtm"] < pd.Timestamp(VAL_START)].reset_index(drop=True)
        curve_fn = fit_group_power_curve(scada_fit, group)
        group_weather = add_teacher_features(weather_tree, weather_teacher, scada_by_group[group], group, variant)
        group_weather = add_power_curve_feature(group_weather, HUB_HEIGHT_PROXY_COL, curve_fn, GROUP_N_TURBINES[group])
        x_train, x_val, y_train, y_val = split_time(group_weather, labels, group)

        oof = cross_val_predict(clone(HIST), x_train, y_train, cv=CV_SPLITTER, n_jobs=1)
        raw = clone(HIST).fit(x_train, y_train).predict(x_val)
        per_group[group] = {"y": y_val.to_numpy(), "raw": raw, "capacity": capacity}
        pooled_oof_pred_pct.append(oof / capacity)
        pooled_oof_actual_pct.append(y_train.to_numpy() / capacity)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.concatenate(pooled_oof_pred_pct), np.concatenate(pooled_oof_actual_pct))

    rows = []
    for group, data in per_group.items():
        capacity = data["capacity"]
        for out_variant, pred in {
            "raw": data["raw"],
            "pooled_isotonic": calibrator.predict(data["raw"] / capacity) * capacity,
        }.items():
            score, nmae, ficr = score_one(data["y"], pred, capacity)
            rows.append({"teacher_variant": variant, "variant": out_variant, "group": group, "score": score, "nmae": nmae, "ficr": ficr})
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
    meteo = build_meteo_features(ldaps, gfs)
    weather_tree = add_meteo_block(build_weather_features(ldaps, gfs), meteo, "all_meteo")
    weather_teacher = add_meteo_block(build_extended_pinn_weather(ldaps, gfs), meteo, "all_meteo")

    rows = []
    for variant in VARIANTS:
        print(f"\n=== {variant} ===")
        current = evaluate_variant(variant, weather_tree, weather_teacher, labels, scada_by_group)
        rows.extend(current)
        tmp = pd.DataFrame(current)
        print(tmp.groupby("variant")["score"].mean().round(4).to_string())

    results = pd.DataFrame(rows)
    means = (
        results.groupby(["teacher_variant", "variant"], as_index=False)
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

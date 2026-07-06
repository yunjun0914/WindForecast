import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from evaluate_pinn_effective_wind_teacher import (
    EXT_TARGETS,
    apply_extended_teacher,
    build_extended_pinn_weather,
    fit_extended_teacher,
)
from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_weather_features

MODELS = {
    "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "lgbm": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
    "xgb": XGBRegressor(random_state=42, n_jobs=-1),
}
VAL_START = "2024-01-01 01:00:00"
RESULTS_PATH = "results/effective_tree_time_holdout_scores.csv"


def time_split(weather_df, labels_df, group):
    weather_df = weather_df.copy()
    weather_df["forecast_kst_dtm"] = pd.to_datetime(weather_df["forecast_kst_dtm"])
    labels_df = labels_df.copy()
    labels_df["kst_dtm"] = pd.to_datetime(labels_df["kst_dtm"])
    merged = weather_df.merge(
        labels_df[["kst_dtm", group]],
        left_on="forecast_kst_dtm",
        right_on="kst_dtm",
        how="inner",
    )
    merged = merged.dropna(subset=[group]).reset_index(drop=True)
    is_val = merged["forecast_kst_dtm"] >= pd.Timestamp(VAL_START)
    feature_cols = [c for c in weather_df.columns if c not in TIME_KEY_COLS]
    return (
        merged.loc[~is_val, feature_cols].reset_index(drop=True),
        merged.loc[is_val, feature_cols].reset_index(drop=True),
        merged.loc[~is_val, group].reset_index(drop=True),
        merged.loc[is_val, group].reset_index(drop=True),
    )


def add_effective_teacher_features(weather_tree, weather_ext, scada_df, group):
    teacher = fit_extended_teacher(weather_ext, scada_df, group, fit_before=VAL_START)
    pred_weather = apply_extended_teacher(weather_ext, teacher, "cubic")
    teacher_cols = [col for col in EXT_TARGETS if col in pred_weather.columns]
    out = weather_tree.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    add = pred_weather[["forecast_kst_dtm"] + teacher_cols + ["v", "v_std"]].copy()
    add = add.rename(columns={col: f"pred_{col}" for col in teacher_cols})
    add = add.rename(columns={"v": "pred_effective_v_cubic", "v_std": "pred_effective_v_std"})
    out = out.merge(add, on="forecast_kst_dtm", how="left")
    out["pred_scada_iqr"] = out["pred_scada_ws_p90"] - out["pred_scada_ws_p10"]
    out["pred_scada_cubic_minus_mean"] = out["pred_scada_ws_cubic"] - out["pred_scada_ws_mean"]
    out["pred_scada_p90_minus_mean"] = out["pred_scada_ws_p90"] - out["pred_scada_ws_mean"]
    return out


def add_selected_grid_stats(weather, weather_ext):
    cols = [
        "forecast_kst_dtm",
        "lead_hour",
        "ldaps_ws10_grid_std",
        "ldaps_ws10_grid_max",
        "ldaps_ws10_p75",
        "ldaps_ws50max_grid_max",
        "ldaps_ws50max_p75",
        "gfs_ws100_grid_max",
        "gfs_ws100_p75",
        "gfs_ws850_grid_max",
        "gfs_ws850_p75",
        "gfs_gust_grid_max",
    ]
    keep = [col for col in cols if col in weather_ext.columns]
    out = weather.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    return out.merge(weather_ext[keep], on="forecast_kst_dtm", how="left")


def score(y, pred, capacity):
    nmae, ficr = group_nmae_ficr(y, np.clip(pred, 0, capacity), capacity)
    return 0.5 * (1 - nmae) + 0.5 * ficr, nmae, ficr


def main():
    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    weather_base = build_weather_features(ldaps, gfs)
    weather_ext = build_extended_pinn_weather(ldaps, gfs)
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }

    rows = []
    for feature_set in ["effective_teacher", "effective_teacher_grid"]:
        print(f"\n=== feature_set={feature_set} ===")
        for group, capacity in GROUP_CAPACITY_KWH.items():
            curve_fn = fit_group_power_curve(scada_by_group[group], group)
            group_weather = add_power_curve_feature(
                weather_base, HUB_HEIGHT_PROXY_COL, curve_fn, GROUP_N_TURBINES[group]
            )
            group_weather = add_effective_teacher_features(group_weather, weather_ext, scada_by_group[group], group)
            if feature_set.endswith("_grid"):
                group_weather = add_selected_grid_stats(group_weather, weather_ext)

            x_train, x_val, y_train, y_val = time_split(group_weather, labels, group)
            print(f"{group}: train={len(x_train)} val={len(x_val)} features={x_train.shape[1]}")
            preds = {}
            for model_name, base_model in MODELS.items():
                model = clone(base_model)
                model.fit(x_train, y_train)
                pred = model.predict(x_val)
                preds[model_name] = pred
                s, nmae, ficr = score(y_val, pred, capacity)
                rows.append(
                    {
                        "feature_set": feature_set,
                        "group": group,
                        "model": model_name,
                        "score": s,
                        "nmae": nmae,
                        "ficr": ficr,
                    }
                )
                print(f"  {model_name}: {s:.4f}")
            ens = sum(preds.values()) / len(preds)
            s, nmae, ficr = score(y_val, ens, capacity)
            rows.append(
                {
                    "feature_set": feature_set,
                    "group": group,
                    "model": "ensemble",
                    "score": s,
                    "nmae": nmae,
                    "ficr": ficr,
                }
            )
            print(f"  ensemble: {s:.4f}")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    print("\n=== Mean score ===")
    print(results.groupby(["feature_set", "model"])["score"].mean().sort_values(ascending=False).round(4))
    print("\n=== Pivot ===")
    print(results.pivot_table(index=["feature_set", "group"], columns="model", values="score").round(4))
    return results


if __name__ == "__main__":
    main()

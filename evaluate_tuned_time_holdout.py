import ast

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_weather_features

VAL_START = "2024-01-01 01:00:00"
TUNING_RESULTS_PATH = "results/tuning_results.csv"

MODEL_CLASSES = {
    "random_forest": RandomForestRegressor,
    "lgbm": LGBMRegressor,
    "xgb": XGBRegressor,
}


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


def load_best_params(tuning_results, group, model_name):
    row = tuning_results[(tuning_results["group"] == group) & (tuning_results["model"] == model_name)].iloc[0]
    return ast.literal_eval(row["best_params"])


def main():
    tuning_results = pd.read_csv(TUNING_RESULTS_PATH, encoding="utf-8-sig")

    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    weather = build_weather_features(ldaps_train, gfs_train)
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")

    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada_by_group = {"kpx_group_1": scada_vestas, "kpx_group_2": scada_vestas, "kpx_group_3": scada_unison}

    rows = []
    for group, capacity in GROUP_CAPACITY_KWH.items():
        curve_fn = fit_group_power_curve(scada_by_group[group], group)
        group_weather = add_power_curve_feature(weather, HUB_HEIGHT_PROXY_COL, curve_fn, GROUP_N_TURBINES[group])
        X_train, X_val, y_train, y_val = time_split(group_weather, labels, group)

        preds = {}
        for model_name, model_cls in MODEL_CLASSES.items():
            params = load_best_params(tuning_results, group, model_name)
            model = model_cls(random_state=42, n_jobs=-1, **params)
            model.fit(X_train, y_train)
            preds[model_name] = model.predict(X_val)

            nmae, ficr = group_nmae_ficr(y_val, preds[model_name], capacity)
            score = 0.5 * (1 - nmae) + 0.5 * ficr
            rows.append({"group": group, "model": f"{model_name}(tuned)", "score": score, "nmae": nmae, "ficr": ficr})
            print(f"{group}/{model_name}(tuned): score={score:.4f} (nmae={nmae:.4f}, ficr={ficr:.4f})")

        ensemble_pred = sum(preds.values()) / len(preds)
        nmae, ficr = group_nmae_ficr(y_val, ensemble_pred, capacity)
        score = 0.5 * (1 - nmae) + 0.5 * ficr
        rows.append({"group": group, "model": "ensemble(tuned)", "score": score, "nmae": nmae, "ficr": ficr})
        print(f"{group}/ensemble(tuned): score={score:.4f} (nmae={nmae:.4f}, ficr={ficr:.4f})")

        lgbm_xgb_pred = (preds["lgbm"] + preds["xgb"]) / 2
        nmae, ficr = group_nmae_ficr(y_val, lgbm_xgb_pred, capacity)
        score = 0.5 * (1 - nmae) + 0.5 * ficr
        rows.append({"group": group, "model": "lgbm+xgb(tuned)", "score": score, "nmae": nmae, "ficr": ficr})
        print(f"{group}/lgbm+xgb(tuned): score={score:.4f} (nmae={nmae:.4f}, ficr={ficr:.4f})")

    results_df = pd.DataFrame(rows)
    print()
    print(results_df.pivot(index="group", columns="model", values="score").round(4))

    print()
    print("3-group average total_score (== official metric, by linearity):")
    print(results_df.groupby("model")["score"].mean().sort_values(ascending=False).round(4))

    return results_df


if __name__ == "__main__":
    main()

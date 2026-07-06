import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr
from utils.power_curve import GROUP_N_TURBINES, add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_weather_features
from utils.scada_teacher import add_honest_scada_teacher_features

MODELS = {
    "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "lgbm": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
    "xgb": XGBRegressor(random_state=42, n_jobs=-1),
}

VAL_START = "2024-01-01 01:00:00"
RESULTS_PATH = "results/scada_teacher_time_holdout_scores.csv"


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


def main():
    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    weather = build_weather_features(ldaps_train, gfs_train)
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")

    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }

    rows = []
    for group, capacity in GROUP_CAPACITY_KWH.items():
        curve_fn = fit_group_power_curve(scada_by_group[group], group)
        group_weather = add_power_curve_feature(weather, HUB_HEIGHT_PROXY_COL, curve_fn, GROUP_N_TURBINES[group])
        group_weather = add_honest_scada_teacher_features(group_weather, scada_by_group[group], group, VAL_START)

        X_train, X_val, y_train, y_val = time_split(group_weather, labels, group)
        print(f"{group}: train={len(X_train)} rows, val(2024)={len(X_val)} rows, features={X_train.shape[1]}")

        preds = {}
        for model_name, base_model in MODELS.items():
            model = clone(base_model)
            model.fit(X_train, y_train)
            pred = model.predict(X_val)
            preds[model_name] = pred

            nmae, ficr = group_nmae_ficr(y_val, pred, capacity)
            score = 0.5 * (1 - nmae) + 0.5 * ficr
            rows.append({"group": group, "model": model_name, "score": score, "nmae": nmae, "ficr": ficr})
            print(f"  {model_name}: score={score:.4f} (nmae={nmae:.4f}, ficr={ficr:.4f})")

        ensemble_pred = sum(preds.values()) / len(preds)
        nmae, ficr = group_nmae_ficr(y_val, ensemble_pred, capacity)
        score = 0.5 * (1 - nmae) + 0.5 * ficr
        rows.append({"group": group, "model": "ensemble", "score": score, "nmae": nmae, "ficr": ficr})
        print(f"  ensemble: score={score:.4f} (nmae={nmae:.4f}, ficr={ficr:.4f})")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    print()
    print(results.pivot(index="group", columns="model", values="score").round(4))
    print()
    print("Mean score:")
    print(results.groupby("model")["score"].mean().sort_values(ascending=False).round(4))
    return results


if __name__ == "__main__":
    main()

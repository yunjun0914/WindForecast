import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, cross_val_predict
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
CV_SPLITTER = KFold(n_splits=3, shuffle=True, random_state=42)
RESULTS_PATH = "results/scada_teacher_calibration_comparison.csv"


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


def model_set_predict(models, X_train, y_train, X_pred):
    preds = []
    for model in models.values():
        fitted = clone(model)
        fitted.fit(X_train, y_train)
        preds.append(fitted.predict(X_pred))
    return sum(preds) / len(preds)


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

    model_sets = {
        "rf_only": {"random_forest": MODELS["random_forest"]},
        "all3": MODELS,
    }

    per_group = {}
    pooled = {name: {"pred": [], "actual": []} for name in model_sets}

    for group, capacity in GROUP_CAPACITY_KWH.items():
        curve_fn = fit_group_power_curve(scada_by_group[group], group)
        group_weather = add_power_curve_feature(weather, HUB_HEIGHT_PROXY_COL, curve_fn, GROUP_N_TURBINES[group])
        group_weather = add_honest_scada_teacher_features(group_weather, scada_by_group[group], group, VAL_START)
        X_train, X_val, y_train, y_val = time_split(group_weather, labels, group)

        per_group[group] = {"capacity": capacity, "y_val": y_val, "sets": {}}
        for set_name, models in model_sets.items():
            oof = np.zeros(len(y_train))
            for model in models.values():
                oof += cross_val_predict(clone(model), X_train, y_train, cv=CV_SPLITTER, n_jobs=1)
            oof /= len(models)
            raw_val = model_set_predict(models, X_train, y_train, X_val)

            per_group[group]["sets"][set_name] = {
                "oof": oof,
                "y_train": y_train.to_numpy(),
                "raw_val": raw_val,
            }
            pooled[set_name]["pred"].append(oof / capacity)
            pooled[set_name]["actual"].append(y_train.to_numpy() / capacity)

    calibrators = {}
    for set_name in model_sets:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(np.concatenate(pooled[set_name]["pred"]), np.concatenate(pooled[set_name]["actual"]))
        calibrators[set_name] = calibrator

    rows = []
    for group, group_data in per_group.items():
        capacity = group_data["capacity"]
        y_val = group_data["y_val"]
        for set_name, data in group_data["sets"].items():
            raw = np.clip(data["raw_val"], 0, capacity)
            cal_pct = calibrators[set_name].predict(data["raw_val"] / capacity)
            cal = np.clip(cal_pct * capacity, 0, capacity)

            for variant, pred in [("raw", raw), ("pooled_isotonic", cal)]:
                nmae, ficr = group_nmae_ficr(y_val, pred, capacity)
                score = 0.5 * (1 - nmae) + 0.5 * ficr
                rows.append(
                    {
                        "group": group,
                        "model_set": set_name,
                        "variant": variant,
                        "score": score,
                        "nmae": nmae,
                        "ficr": ficr,
                    }
                )
                print(f"{group}/{set_name}/{variant}: score={score:.4f} (nmae={nmae:.4f}, ficr={ficr:.4f})")

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    print()
    print(results.pivot_table(index="group", columns=["model_set", "variant"], values="score").round(4))
    print()
    print("Mean score:")
    print(results.groupby(["model_set", "variant"])["score"].mean().sort_values(ascending=False).round(4))
    return results


if __name__ == "__main__":
    main()

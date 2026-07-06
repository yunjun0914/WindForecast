import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

from models.pinn import PowerCurvePINN, TurbineGroupBias
from train_pinn import DEVICE, group_prediction, load_training_data, time_split
from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr
from utils.pinn_data import C_MAX_BY_MANUFACTURER, GROUP_MANUFACTURER, GROUP_N_TURBINES, build_group_pinn_dataset
from utils.pinn_physics import MANUFACTURER_AREA
from utils.power_curve import GROUP_N_TURBINES as TREE_GROUP_N_TURBINES
from utils.power_curve import add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_group_dataset, build_weather_features

MODELS = {
    "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "lgbm": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
    "xgb": XGBRegressor(random_state=42, n_jobs=-1),
}
CV_SPLITTER = KFold(n_splits=3, shuffle=True, random_state=42)
VAL_START = "2024-01-01 01:00:00"
RESULTS_PATH = "results/pinn_tree_blend_scores.csv"


def tree_time_split(weather_df, labels_df, group):
    merged = weather_df.merge(
        labels_df[["kst_dtm", group]], left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner"
    )
    merged = merged.dropna(subset=[group]).reset_index(drop=True)
    is_val = merged["forecast_kst_dtm"] >= VAL_START
    feature_cols = [c for c in weather_df.columns if c not in TIME_KEY_COLS]
    X_train, X_val = merged.loc[~is_val, feature_cols], merged.loc[is_val, feature_cols]
    y_train, y_val = merged.loc[~is_val, group], merged.loc[is_val, group]
    val_time = merged.loc[is_val, "forecast_kst_dtm"].reset_index(drop=True)
    return X_train, X_val, y_train, y_val, val_time


def ensemble_predict(models, X_train, y_train, X_pred):
    preds = []
    for model in models.values():
        fitted = clone(model)
        fitted.fit(X_train, y_train)
        preds.append(fitted.predict(X_pred))
    return sum(preds) / len(preds)


def build_tree_predictions():
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

    out = {}
    pooled_oof_pred_pct, pooled_oof_actual_pct = [], []
    for group, capacity in GROUP_CAPACITY_KWH.items():
        curve_fn = fit_group_power_curve(scada_by_group[group], group)
        group_weather = add_power_curve_feature(weather, HUB_HEIGHT_PROXY_COL, curve_fn, TREE_GROUP_N_TURBINES[group])
        X_train, X_val, y_train, y_val, val_time = tree_time_split(group_weather, labels, group)

        oof = np.zeros(len(y_train))
        for model in MODELS.values():
            oof += cross_val_predict(clone(model), X_train, y_train, cv=CV_SPLITTER, n_jobs=1)
        oof /= len(MODELS)
        raw_val = ensemble_predict(MODELS, X_train, y_train, X_val)

        out[group] = {
            "time": pd.to_datetime(val_time),
            "y": y_val.to_numpy(),
            "raw": raw_val,
            "capacity": capacity,
        }
        pooled_oof_pred_pct.append(oof / capacity)
        pooled_oof_actual_pct.append(y_train.to_numpy() / capacity)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.concatenate(pooled_oof_pred_pct), np.concatenate(pooled_oof_actual_pct))
    for group, data in out.items():
        capacity = data["capacity"]
        data["calibrated"] = np.clip(calibrator.predict(data["raw"] / capacity) * capacity, 0, capacity)
        data["raw"] = np.clip(data["raw"], 0, capacity)
    return out


def build_pinn_predictions():
    weather_by_manufacturer, labels = load_training_data()
    out = {}
    for manufacturer, weather_by_group in weather_by_manufacturer.items():
        model = PowerCurvePINN(C_MAX_BY_MANUFACTURER[manufacturer], MANUFACTURER_AREA[manufacturer]).to(DEVICE)
        model.load_state_dict(torch.load(f"results/pinn_{manufacturer}_stage2.pt", map_location=DEVICE))
        model.eval()
        for group, group_manufacturer in GROUP_MANUFACTURER.items():
            if group_manufacturer != manufacturer:
                continue
            ds = build_group_pinn_dataset(weather_by_group[group], labels, group)
            train_df, val_df = time_split(ds)
            train_years = train_df["forecast_kst_dtm"].dt.year.unique()
            bias = TurbineGroupBias(
                GROUP_CAPACITY_KWH[group],
                n_train_rows=None,
                n_train_years=None,
            ).to(DEVICE)
            bias.load_state_dict(torch.load(f"results/pinn_{group}_bias.pt", map_location=DEVICE))
            bias.eval()
            with torch.no_grad():
                pred = group_prediction(
                    model,
                    val_df,
                    GROUP_N_TURBINES[group],
                    bias=bias,
                    use_wind_distribution=True,
                )
                pred = torch.clamp(pred, min=0.0, max=GROUP_CAPACITY_KWH[group]).cpu().numpy()
            out[group] = {
                "time": pd.to_datetime(val_df["forecast_kst_dtm"]).reset_index(drop=True),
                "pred": pred,
            }
    return out


def score_one(y, pred, capacity):
    nmae, ficr = group_nmae_ficr(y, pred, capacity)
    return 0.5 * (1 - nmae) + 0.5 * ficr, nmae, ficr


def main():
    tree = build_tree_predictions()
    pinn = build_pinn_predictions()

    rows = []
    weights = np.linspace(0, 1, 21)  # weight on PINN
    for tree_variant in ["raw", "calibrated"]:
        for w in weights:
            group_scores = []
            for group, tree_data in tree.items():
                if not tree_data["time"].equals(pinn[group]["time"]):
                    raise ValueError(f"time mismatch for {group}")
                capacity = tree_data["capacity"]
                pred = w * pinn[group]["pred"] + (1 - w) * tree_data[tree_variant]
                pred = np.clip(pred, 0, capacity)
                score, nmae, ficr = score_one(tree_data["y"], pred, capacity)
                rows.append(
                    {
                        "tree_variant": tree_variant,
                        "pinn_weight": w,
                        "group": group,
                        "score": score,
                        "nmae": nmae,
                        "ficr": ficr,
                    }
                )
                group_scores.append(score)
            rows.append(
                {
                    "tree_variant": tree_variant,
                    "pinn_weight": w,
                    "group": "mean",
                    "score": float(np.mean(group_scores)),
                    "nmae": np.nan,
                    "ficr": np.nan,
                }
            )

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    mean_rows = results[results["group"] == "mean"].sort_values("score", ascending=False)
    print(mean_rows.head(10).to_string(index=False))

    best = mean_rows.iloc[0]
    print("\nBest group breakdown:")
    print(
        results[
            (results["tree_variant"] == best["tree_variant"])
            & (results["pinn_weight"] == best["pinn_weight"])
            & (results["group"] != "mean")
        ].to_string(index=False)
    )
    return results


if __name__ == "__main__":
    main()

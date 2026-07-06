import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from evaluate_pinn_tree_blend import build_pinn_predictions, build_tree_predictions, score_one, tree_time_split
from models.pinn import PowerCurvePINN, TurbineGroupBias
from train_pinn import DEVICE, group_prediction, load_training_data, time_split
from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr
from utils.pinn_data import C_MAX_BY_MANUFACTURER, GROUP_N_TURBINES, build_group_pinn_dataset
from utils.pinn_physics import MANUFACTURER_AREA
from utils.power_curve import GROUP_N_TURBINES as TREE_GROUP_N_TURBINES
from utils.power_curve import add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_weather_features

MODELS = {
    "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "lgbm": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
    "xgb": XGBRegressor(random_state=42, n_jobs=-1),
}
VAL_START = "2024-01-01 01:00:00"
GROUP3 = "kpx_group_3"
GROUP2 = "kpx_group_2"
RESULTS_PATH = "results/group3_transfer_blend_scores.csv"


def ensemble_fit_predict(x_train, y_train, x_pred):
    preds = []
    for model in MODELS.values():
        fitted = clone(model)
        fitted.fit(x_train, y_train)
        preds.append(fitted.predict(x_pred))
    return sum(preds) / len(preds)


def group2_transfer_tree_candidates():
    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    weather = build_weather_features(ldaps_train, gfs_train)
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")

    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")

    group2_curve = fit_group_power_curve(scada_vestas, GROUP2)
    group2_weather = add_power_curve_feature(
        weather, HUB_HEIGHT_PROXY_COL, group2_curve, TREE_GROUP_N_TURBINES[GROUP2]
    )
    g2_train_x, g2_val_x, g2_train_y, _, g2_val_time = tree_time_split(group2_weather, labels, GROUP2)
    g2_val_pred = ensemble_fit_predict(g2_train_x, g2_train_y, g2_val_x)
    g2_proxy = pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(g2_val_time),
            "pred": np.clip(
                g2_val_pred / GROUP_CAPACITY_KWH[GROUP2] * GROUP_CAPACITY_KWH[GROUP3],
                0,
                GROUP_CAPACITY_KWH[GROUP3],
            ),
        }
    )

    group3_curve = fit_group_power_curve(scada_unison, GROUP3)
    group3_weather = add_power_curve_feature(
        weather, HUB_HEIGHT_PROXY_COL, group3_curve, TREE_GROUP_N_TURBINES[GROUP3]
    )
    _, g3_val_x, _, g3_val_y, g3_val_time = tree_time_split(group3_weather, labels, GROUP3)
    cross_feature_pred = ensemble_fit_predict(g2_train_x, g2_train_y, g3_val_x)
    cross_feature_pred = np.clip(
        cross_feature_pred / GROUP_CAPACITY_KWH[GROUP2] * GROUP_CAPACITY_KWH[GROUP3],
        0,
        GROUP_CAPACITY_KWH[GROUP3],
    )

    base = pd.DataFrame({"forecast_kst_dtm": pd.to_datetime(g3_val_time), "y": g3_val_y.to_numpy()})
    proxy_aligned = base.merge(g2_proxy, on="forecast_kst_dtm", how="inner")
    if len(proxy_aligned) != len(base):
        raise ValueError(f"group2 proxy alignment mismatch: {len(proxy_aligned)} vs {len(base)}")

    return {
        "tree_group2_proxy_same_time": {
            "time": base["forecast_kst_dtm"],
            "y": base["y"].to_numpy(),
            "pred": proxy_aligned["pred"].to_numpy(),
        },
        "tree_group2_transfer_group3_features": {
            "time": base["forecast_kst_dtm"],
            "y": base["y"].to_numpy(),
            "pred": cross_feature_pred,
        },
    }


def vestas_pinn_transfer_candidates(base_time):
    weather_by_manufacturer, labels = load_training_data()
    capacity3 = GROUP_CAPACITY_KWH[GROUP3]
    out = {}

    vestas_model = PowerCurvePINN(C_MAX_BY_MANUFACTURER["vestas"], MANUFACTURER_AREA["vestas"]).to(DEVICE)
    vestas_model.load_state_dict(torch.load("results/pinn_vestas_stage2.pt", map_location=DEVICE))
    vestas_model.eval()

    group2_bias = TurbineGroupBias(GROUP_CAPACITY_KWH[GROUP2]).to(DEVICE)
    group2_bias.load_state_dict(torch.load("results/pinn_kpx_group_2_bias.pt", map_location=DEVICE))
    group2_bias.eval()

    group3_bias = TurbineGroupBias(capacity3).to(DEVICE)
    group3_bias.load_state_dict(torch.load("results/pinn_kpx_group_3_bias.pt", map_location=DEVICE))
    group3_bias.eval()

    # Same-time proxy: use the VESTAS PINN's group2 prediction percentage and scale to group3 capacity.
    group2_ds = build_group_pinn_dataset(weather_by_manufacturer["vestas"][GROUP2], labels, GROUP2)
    _, group2_val = time_split(group2_ds)
    with torch.no_grad():
        group2_pred = group_prediction(
            vestas_model,
            group2_val,
            GROUP_N_TURBINES[GROUP2],
            bias=group2_bias,
            use_wind_distribution=True,
        )
        group2_pred = torch.clamp(group2_pred, min=0.0, max=GROUP_CAPACITY_KWH[GROUP2]).cpu().numpy()
    proxy_df = pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(group2_val["forecast_kst_dtm"]),
            "pred": np.clip(group2_pred / GROUP_CAPACITY_KWH[GROUP2] * capacity3, 0, capacity3),
        }
    )
    proxy_aligned = pd.DataFrame({"forecast_kst_dtm": base_time}).merge(proxy_df, on="forecast_kst_dtm", how="inner")
    if len(proxy_aligned) == len(base_time):
        out["pinn_vestas_group2_proxy_same_time"] = proxy_aligned["pred"].to_numpy()

    # Cross-physics candidate: run the VESTAS backbone directly on group3's SCADA-teacher weather.
    group3_ds = build_group_pinn_dataset(weather_by_manufacturer["unison"][GROUP3], labels, GROUP3)
    _, group3_val = time_split(group3_ds)
    if not pd.to_datetime(group3_val["forecast_kst_dtm"]).reset_index(drop=True).equals(base_time.reset_index(drop=True)):
        raise ValueError("group3 PINN time mismatch")
    with torch.no_grad():
        direct_pred = group_prediction(
            vestas_model,
            group3_val,
            GROUP_N_TURBINES[GROUP3],
            bias=group3_bias,
            use_wind_distribution=True,
        )
        direct_pred = torch.clamp(direct_pred, min=0.0, max=capacity3).cpu().numpy()
    out["pinn_vestas_backbone_on_group3_weather"] = direct_pred
    return out


def score_candidate(name, y, pred):
    score, nmae, ficr = score_one(y, np.clip(pred, 0, GROUP_CAPACITY_KWH[GROUP3]), GROUP_CAPACITY_KWH[GROUP3])
    return {"candidate": name, "pinn_weight": np.nan, "score": score, "nmae": nmae, "ficr": ficr}


def main():
    tree = build_tree_predictions()
    pinn = build_pinn_predictions()
    transfer_tree = group2_transfer_tree_candidates()

    base = tree[GROUP3]
    base_time = base["time"].reset_index(drop=True)
    y = base["y"]
    capacity = GROUP_CAPACITY_KWH[GROUP3]

    if not base_time.equals(pinn[GROUP3]["time"].reset_index(drop=True)):
        raise ValueError("base/PINN time mismatch")

    candidates = {
        "pinn_unison": pinn[GROUP3]["pred"],
        "tree_own_raw": base["raw"],
        "tree_own_calibrated": base["calibrated"],
    }
    for name, data in transfer_tree.items():
        if not base_time.equals(data["time"].reset_index(drop=True)):
            raise ValueError(f"{name} time mismatch")
        candidates[name] = data["pred"]
    candidates.update(vestas_pinn_transfer_candidates(base_time))

    rows = [score_candidate(name, y, pred) for name, pred in candidates.items()]
    weights = np.linspace(0, 1, 21)
    anchor = candidates["pinn_unison"]
    for name, pred2 in candidates.items():
        if name == "pinn_unison":
            continue
        for w in weights:
            pred = np.clip(w * anchor + (1 - w) * pred2, 0, capacity)
            score, nmae, ficr = score_one(y, pred, capacity)
            rows.append(
                {
                    "candidate": f"blend:pinn_unison+{name}",
                    "pinn_weight": w,
                    "score": score,
                    "nmae": nmae,
                    "ficr": ficr,
                }
            )

    results = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    print(results.head(25).to_string(index=False))

    fixed_g1_g2 = 0.635736, 0.651669
    best = results.iloc[0]
    implied_mean = (fixed_g1_g2[0] + fixed_g1_g2[1] + best["score"]) / 3
    print(f"\nBest group3={best['score']:.6f}; implied mean with previous best g1/g2={implied_mean:.6f}")
    return results


if __name__ == "__main__":
    main()

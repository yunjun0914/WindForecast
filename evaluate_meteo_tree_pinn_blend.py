import numpy as np
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import cross_val_predict

from evaluate_group3_effective_teacher_mix import blend_weather
from evaluate_pinn_effective_wind_teacher import (
    apply_extended_teacher,
    build_extended_pinn_weather,
    fit_extended_teacher,
)
from evaluate_tree_meteo_feature_blocks_fast import add_meteo_block, build_meteo_features
from predict_tree_feature_block import CV_SPLITTER, MODELS, ensemble_fit_predict
from train_pinn import DEVICE, group_prediction, train_manufacturer
from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr
from utils.pinn_data import GROUP_N_TURBINES as PINN_GROUP_N_TURBINES
from utils.power_curve import GROUP_N_TURBINES as TREE_GROUP_N_TURBINES
from utils.power_curve import add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_weather_features

VAL_START = "2024-01-01 01:00:00"
RESULTS_PATH = "results/meteo_tree_pinn_blend_scores.csv"
GROUPS = ["kpx_group_1", "kpx_group_2", "kpx_group_3"]


def score_one(y, pred, capacity):
    pred = np.clip(pred, 0, capacity)
    nmae, ficr = group_nmae_ficr(y, pred, capacity)
    return 0.5 * (1 - nmae) + 0.5 * ficr, nmae, ficr


def tree_time_split(weather_df, labels_df, group):
    weather_df = weather_df.copy()
    weather_df["forecast_kst_dtm"] = pd.to_datetime(weather_df["forecast_kst_dtm"])
    labels = labels_df[["kst_dtm", group]].copy()
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    merged = weather_df.merge(labels, left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner")
    merged = merged.dropna(subset=[group]).reset_index(drop=True)
    is_val = merged["forecast_kst_dtm"] >= pd.Timestamp(VAL_START)
    feature_cols = [c for c in weather_df.columns if c not in TIME_KEY_COLS]
    return (
        merged.loc[~is_val, feature_cols].reset_index(drop=True),
        merged.loc[is_val, feature_cols].reset_index(drop=True),
        merged.loc[~is_val, group].reset_index(drop=True),
        merged.loc[is_val, group].reset_index(drop=True),
        merged.loc[is_val, "forecast_kst_dtm"].reset_index(drop=True),
    )


def build_tree_meteo_predictions(ldaps, gfs, labels, scada_vestas, scada_unison):
    weather_base = build_weather_features(ldaps, gfs)
    meteo = build_meteo_features(ldaps, gfs)
    weather = add_meteo_block(weather_base, meteo, "all_meteo")
    scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }

    per_group = {}
    pooled_oof_pred_pct, pooled_oof_actual_pct = [], []
    for group, capacity in GROUP_CAPACITY_KWH.items():
        scada = scada_by_group[group].copy()
        scada["kst_dtm"] = pd.to_datetime(scada["kst_dtm"])
        scada = scada[scada["kst_dtm"] < pd.Timestamp(VAL_START)].reset_index(drop=True)
        curve_fn = fit_group_power_curve(scada, group)
        group_weather = add_power_curve_feature(weather, HUB_HEIGHT_PROXY_COL, curve_fn, TREE_GROUP_N_TURBINES[group])
        x_train, x_val, y_train, y_val, val_time = tree_time_split(group_weather, labels, group)

        oof = np.zeros(len(y_train))
        for model in MODELS.values():
            oof += cross_val_predict(clone(model), x_train, y_train, cv=CV_SPLITTER, n_jobs=1)
        oof /= len(MODELS)
        raw = ensemble_fit_predict(x_train, y_train, x_val)
        per_group[group] = {
            "time": pd.to_datetime(val_time).reset_index(drop=True),
            "y": y_val.to_numpy(),
            "raw": np.clip(raw, 0, capacity),
            "capacity": capacity,
        }
        pooled_oof_pred_pct.append(oof / capacity)
        pooled_oof_actual_pct.append(y_train.to_numpy() / capacity)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.concatenate(pooled_oof_pred_pct), np.concatenate(pooled_oof_actual_pct))
    for group, data in per_group.items():
        capacity = data["capacity"]
        data["calibrated"] = np.clip(calibrator.predict(data["raw"] / capacity) * capacity, 0, capacity)
    return per_group


def teacher_weather(weather, scada_df, group, mode):
    teacher = fit_extended_teacher(weather, scada_df, group, fit_before=VAL_START)
    return apply_extended_teacher(weather, teacher, mode)


def build_pinn_meteo_weather(ldaps, gfs, scada_vestas, scada_unison):
    weather_ext = build_extended_pinn_weather(ldaps, gfs)
    meteo = build_meteo_features(ldaps, gfs)
    weather = add_meteo_block(weather_ext, meteo, "all_meteo")

    g1 = teacher_weather(weather, scada_vestas, "kpx_group_1", "cubic")
    g2 = teacher_weather(weather, scada_vestas, "kpx_group_2", "p90")
    g3_unison = teacher_weather(weather, scada_unison, "kpx_group_3", "p90")
    g3_vestas = teacher_weather(weather, scada_vestas, "kpx_group_2", "p90")
    g3 = blend_weather("group3_meteo_p90_mix", g3_unison, g3_vestas, 0.30)
    return {"kpx_group_1": g1, "kpx_group_2": g2, "kpx_group_3": g3}


def predict_pinn_validation(weather_by_group, labels):
    out = {}
    vestas_model, vestas_data, _, _ = train_manufacturer(
        "vestas",
        {"kpx_group_1": weather_by_group["kpx_group_1"], "kpx_group_2": weather_by_group["kpx_group_2"]},
        labels,
        verbose=False,
        save=False,
    )
    unison_model, unison_data, _, _ = train_manufacturer(
        "unison",
        {"kpx_group_3": weather_by_group["kpx_group_3"]},
        labels,
        verbose=False,
        save=False,
    )
    for model, group_data in [(vestas_model, vestas_data), (unison_model, unison_data)]:
        model.eval()
        for group, data in group_data.items():
            with torch.no_grad():
                pred = group_prediction(
                    model,
                    data["val"],
                    PINN_GROUP_N_TURBINES[group],
                    bias=data["bias"],
                    use_wind_distribution=True,
                )
                pred = torch.clamp(pred, min=0.0, max=data["capacity"]).cpu().numpy()
            out[group] = {
                "time": pd.to_datetime(data["val"]["forecast_kst_dtm"]).reset_index(drop=True),
                "pred": pred,
            }
    return out


def append_candidate(rows, name, tree, pinn, weights):
    group_scores = []
    for group in GROUPS:
        capacity = tree[group]["capacity"]
        pred = np.clip((1 - weights[group]) * tree[group]["calibrated"] + weights[group] * pinn[group]["pred"], 0, capacity)
        score, nmae, ficr = score_one(tree[group]["y"], pred, capacity)
        rows.append({"candidate": name, "group": group, "pinn_weight": weights[group], "score": score, "nmae": nmae, "ficr": ficr})
        group_scores.append(score)
    rows.append(
        {
            "candidate": name,
            "group": "mean",
            "pinn_weight": np.mean([weights[g] for g in GROUPS]),
            "score": float(np.mean(group_scores)),
            "nmae": np.nan,
            "ficr": np.nan,
        }
    )


def main():
    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")

    tree = build_tree_meteo_predictions(ldaps, gfs, labels, scada_vestas, scada_unison)
    pinn_weather = build_pinn_meteo_weather(ldaps, gfs, scada_vestas, scada_unison)
    pinn = predict_pinn_validation(pinn_weather, labels)
    for group in GROUPS:
        if not tree[group]["time"].equals(pinn[group]["time"]):
            raise ValueError(f"time mismatch for {group}")

    rows = []
    append_candidate(rows, "tree_meteo", tree, pinn, {g: 0.0 for g in GROUPS})
    append_candidate(rows, "pinn_meteo", tree, pinn, {g: 1.0 for g in GROUPS})
    for w in np.linspace(0.05, 0.50, 10):
        append_candidate(rows, f"global_pinn_{w:.2f}", tree, pinn, {g: float(w) for g in GROUPS})

    per_group_best = {}
    for group in GROUPS:
        best = None
        for w in np.linspace(0, 0.80, 17):
            capacity = tree[group]["capacity"]
            pred = np.clip((1 - w) * tree[group]["calibrated"] + w * pinn[group]["pred"], 0, capacity)
            score, nmae, ficr = score_one(tree[group]["y"], pred, capacity)
            row = {"weight": float(w), "score": score, "nmae": nmae, "ficr": ficr}
            if best is None or row["score"] > best["score"]:
                best = row
        per_group_best[group] = best
    append_candidate(rows, "per_group_best", tree, pinn, {g: per_group_best[g]["weight"] for g in GROUPS})

    results = pd.DataFrame(rows)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    print("\n=== mean candidates ===")
    print(results[results["group"] == "mean"].sort_values("score", ascending=False).to_string(index=False))
    print("\n=== per-group best weights ===")
    print(pd.DataFrame([{ "group": g, **v } for g, v in per_group_best.items()]).to_string(index=False))
    print("\n=== best candidate breakdown ===")
    best_name = results[results["group"] == "mean"].sort_values("score", ascending=False).iloc[0]["candidate"]
    print(results[(results["candidate"] == best_name) & (results["group"] != "mean")].to_string(index=False))
    return results


if __name__ == "__main__":
    main()

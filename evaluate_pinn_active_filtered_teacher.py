import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from evaluate_group3_effective_teacher_mix import blend_weather
from evaluate_pinn_effective_wind_teacher import EXT_TARGETS, build_extended_pinn_weather, extended_feature_cols
from evaluate_tree_meteo_feature_blocks_fast import add_meteo_block, build_meteo_features
from train_pinn import train_manufacturer
from utils.pinn_data import GROUP_MANUFACTURER
from utils.power_curve import GROUP_TURBINE_PREFIXES


RESULTS_PATH = "results/pinn_active_filtered_teacher_scores.csv"
VAL_START = "2024-01-01 01:00:00"
GROUP1 = "kpx_group_1"
GROUP2 = "kpx_group_2"
GROUP3 = "kpx_group_3"
BASE_GROUP_SCADA = {GROUP1: "vestas", GROUP2: "vestas", GROUP3: "unison"}
TARGET_COLS = EXT_TARGETS + ["scada_power_active_ratio", "scada_filter_keep_ratio"]


def build_active_filtered_targets(scada_df, group, power_threshold=10.0, low_wind_keep=3.5):
    scada = scada_df.copy()
    scada["kst_dtm"] = pd.to_datetime(scada["kst_dtm"])
    scada["hour"] = scada["kst_dtm"].dt.floor("h")
    rows = []
    for hour, hourly in scada.groupby("hour"):
        ws_parts, power_parts = [], []
        for prefix in GROUP_TURBINE_PREFIXES[group]:
            ws_parts.append(hourly[f"{prefix}_ws"].to_numpy(dtype=float))
            power_parts.append(hourly[f"{prefix}_power_kw10m"].to_numpy(dtype=float))
        ws = np.concatenate(ws_parts)
        power = np.clip(np.concatenate(power_parts), 0, None)
        valid = np.isfinite(ws) & np.isfinite(power)
        if not valid.any():
            continue
        ws = ws[valid]
        power = power[valid]
        active = power > power_threshold
        keep = active | (ws < low_wind_keep)
        if keep.sum() < max(3, int(0.2 * len(ws))):
            keep = np.ones_like(active, dtype=bool)
        kept_ws = ws[keep]
        rows.append(
            {
                "forecast_kst_dtm": hour,
                "scada_ws_mean": np.nanmean(kept_ws),
                "scada_ws_std": np.nanstd(kept_ws),
                "scada_ws_p10": np.nanpercentile(kept_ws, 10),
                "scada_ws_p50": np.nanpercentile(kept_ws, 50),
                "scada_ws_p75": np.nanpercentile(kept_ws, 75),
                "scada_ws_p90": np.nanpercentile(kept_ws, 90),
                "scada_ws_max": np.nanmax(kept_ws),
                "scada_ws_cubic": np.cbrt(np.nanmean(np.clip(kept_ws, 0, None) ** 3)),
                "scada_power_active_ratio": float(active.mean()),
                "scada_filter_keep_ratio": float(keep.mean()),
            }
        )
    out = pd.DataFrame(rows)
    out["scada_ws_ramp"] = out["scada_ws_mean"].diff().fillna(0)
    return out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def make_models():
    rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=120, min_samples_leaf=10, random_state=42, n_jobs=-1))
    hist = MultiOutputRegressor(
        HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, l2_regularization=0.01, random_state=42)
    )
    return [rf, hist]


def fit_teacher(weather, scada_df, group, fit_before=VAL_START):
    targets = build_active_filtered_targets(scada_df, group)
    df = weather.merge(targets, on="forecast_kst_dtm", how="inner").dropna()
    if fit_before is not None:
        df = df[df["forecast_kst_dtm"] < pd.Timestamp(fit_before)]
    feature_cols = extended_feature_cols(weather)
    models = make_models()
    for model in models:
        model.fit(df[feature_cols], df[TARGET_COLS])
    return feature_cols, models


def apply_teacher(weather, teacher, v_mode):
    feature_cols, models = teacher
    pred = sum(model.predict(weather[feature_cols]) for model in models) / len(models)
    pred = pd.DataFrame(pred, columns=TARGET_COLS, index=weather.index)
    for col in TARGET_COLS:
        pred[col] = np.clip(pred[col], 0, None)

    out = weather.copy()
    for col in TARGET_COLS:
        out[col] = pred[col]
    if v_mode == "filtered_cubic":
        v = pred["scada_ws_cubic"]
    elif v_mode == "filtered_p90":
        v = pred["scada_ws_p90"]
    elif v_mode == "filtered_mix_cubic_p90":
        v = 0.5 * pred["scada_ws_cubic"] + 0.5 * pred["scada_ws_p90"]
    else:
        raise ValueError(v_mode)
    spread_sigma = (pred["scada_ws_p90"] - pred["scada_ws_p10"]) / 2.563
    out["v"] = np.clip(v, 0, None)
    out["v_std"] = np.clip(0.5 * pred["scada_ws_std"] + 0.5 * spread_sigma, 0.05, None)
    return out


def build_variant_weather(variant, weather, scada_by_name):
    by_group = {}
    for group, scada_name in BASE_GROUP_SCADA.items():
        teacher = fit_teacher(weather, scada_by_name[scada_name], group, fit_before=VAL_START)
        by_group[group] = apply_teacher(weather, teacher, variant)

    g3_vestas_teacher = fit_teacher(weather, scada_by_name["vestas"], GROUP2, fit_before=VAL_START)
    g3_vestas = apply_teacher(weather, g3_vestas_teacher, variant)
    by_group[GROUP3] = blend_weather(f"group3_{variant}_active_filtered_mix", by_group[GROUP3], g3_vestas, 0.30)
    return {
        "vestas": {group: by_group[group] for group in by_group if GROUP_MANUFACTURER[group] == "vestas"},
        "unison": {group: by_group[group] for group in by_group if GROUP_MANUFACTURER[group] == "unison"},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="filtered_cubic,filtered_p90,filtered_mix_cubic_p90")
    args = parser.parse_args()

    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    scada_by_name = {
        "vestas": pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig"),
        "unison": pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig"),
    }
    weather = build_extended_pinn_weather(ldaps, gfs)
    weather = add_meteo_block(weather, build_meteo_features(ldaps, gfs), "all_meteo")

    rows = []
    for variant in [v.strip() for v in args.variants.split(",") if v.strip()]:
        print(f"\n=== variant: {variant} ===")
        weather_by_manufacturer = build_variant_weather(variant, weather, scada_by_name)
        for manufacturer, weather_by_group in weather_by_manufacturer.items():
            _, _, stage1, stage2 = train_manufacturer(
                manufacturer,
                weather_by_group,
                labels,
                verbose=False,
                save=False,
            )
            stage1["stage"] = "physics_only"
            stage2["stage"] = "with_bias"
            for frame in [stage1, stage2]:
                frame["variant"] = variant
                frame["manufacturer"] = manufacturer
                rows.append(frame)
        current = pd.concat(rows, ignore_index=True)
        print(current[(current["variant"] == variant) & (current["stage"] == "with_bias")].to_string(index=False))

    results = pd.concat(rows, ignore_index=True)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    summary = (
        results[results["stage"] == "with_bias"]
        .groupby("variant", as_index=False)
        .agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"))
        .sort_values("score", ascending=False)
    )
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    return results


if __name__ == "__main__":
    main()

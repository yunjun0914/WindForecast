import numpy as np
import pandas as pd
import argparse

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from train_pinn import train_manufacturer
from utils.metrics import GROUP_CAPACITY_KWH
from utils.pinn_data import (
    GROUP_MANUFACTURER,
    SCADA_TEACHER_TARGETS,
    apply_scada_wind_teacher,
    build_pinn_weather,
    fit_scada_wind_teacher,
)
from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.preprocessing import TIME_KEY_COLS

VAL_START = "2024-01-01 01:00:00"
RESULTS_PATH = "results/pinn_effective_wind_teacher_scores.csv"

BASE_GROUP_SCADA = {
    "kpx_group_1": "vestas",
    "kpx_group_2": "vestas",
    "kpx_group_3": "unison",
}

EXT_TARGETS = [
    "scada_ws_mean",
    "scada_ws_std",
    "scada_ws_p10",
    "scada_ws_p50",
    "scada_ws_p75",
    "scada_ws_p90",
    "scada_ws_max",
    "scada_ws_cubic",
    "scada_ws_ramp",
]


def speed(u, v):
    return np.sqrt(u**2 + v**2)


def add_lead_feature(out, raw_df):
    keys = raw_df[["forecast_kst_dtm", "data_available_kst_dtm"]].drop_duplicates().copy()
    keys["forecast_kst_dtm"] = pd.to_datetime(keys["forecast_kst_dtm"])
    keys["data_available_kst_dtm"] = pd.to_datetime(keys["data_available_kst_dtm"])
    lead = keys.groupby("forecast_kst_dtm")["data_available_kst_dtm"].min().reset_index()
    lead["lead_hour"] = (lead["forecast_kst_dtm"] - lead["data_available_kst_dtm"]).dt.total_seconds() / 3600.0
    return out.merge(lead[["forecast_kst_dtm", "lead_hour"]], on="forecast_kst_dtm", how="left")


def aggregate_speed_distribution(df, specs, prefix, group_cols):
    work = df[group_cols].copy()
    for u_col, v_col, name in specs:
        work[f"_{name}"] = speed(df[u_col], df[v_col])

    rows = []
    for name in [spec[2] for spec in specs]:
        grouped = work.groupby("forecast_kst_dtm")[f"_{name}"]
        stat = grouped.agg(["mean", "std", "min", "max"]).reset_index()
        stat[f"{prefix}_{name}_p75"] = grouped.quantile(0.75).to_numpy()
        stat = stat.rename(
            columns={
                "mean": f"{prefix}_{name}_grid_mean",
                "std": f"{prefix}_{name}_grid_std",
                "min": f"{prefix}_{name}_grid_min",
                "max": f"{prefix}_{name}_grid_max",
            }
        )
        rows.append(stat)

    out = rows[0]
    for row in rows[1:]:
        out = out.merge(row, on="forecast_kst_dtm", how="left")
    return out


def build_extended_pinn_weather(ldaps_df, gfs_df):
    base = build_pinn_weather(ldaps_df, gfs_df)

    ldaps = ldaps_df.copy()
    ldaps["forecast_kst_dtm"] = pd.to_datetime(ldaps["forecast_kst_dtm"])
    ldaps_specs = [
        ("heightAboveGround_10_10u", "heightAboveGround_10_10v", "ws10"),
        ("heightAboveGround_50_50MUmax", "heightAboveGround_50_50MVmax", "ws50max"),
        ("heightAboveGround_50_50MUmin", "heightAboveGround_50_50MVmin", "ws50min"),
        ("heightAboveGround_5_XBLWS", "heightAboveGround_5_YBLWS", "ws5bl"),
    ]
    ldaps_dist = aggregate_speed_distribution(ldaps, ldaps_specs, "ldaps", ["forecast_kst_dtm"])

    gfs = gfs_df.copy()
    gfs["forecast_kst_dtm"] = pd.to_datetime(gfs["forecast_kst_dtm"])
    gfs_specs = [
        ("heightAboveGround_80_u", "heightAboveGround_80_v", "ws80"),
        ("heightAboveGround_100_100u", "heightAboveGround_100_100v", "ws100"),
        ("isobaricInhPa_850_u", "isobaricInhPa_850_v", "ws850"),
        ("surface_0_gust", "surface_0_gust", "gust_proxy"),
    ]
    # gust is scalar; using speed(gust, gust) is not physical, overwrite after aggregation.
    gfs_dist = aggregate_speed_distribution(gfs, gfs_specs[:3], "gfs", ["forecast_kst_dtm"])
    gust = gfs.groupby("forecast_kst_dtm")["surface_0_gust"].agg(["mean", "std", "min", "max"]).reset_index()
    gust = gust.rename(
        columns={
            "mean": "gfs_gust_grid_mean",
            "std": "gfs_gust_grid_std",
            "min": "gfs_gust_grid_min",
            "max": "gfs_gust_grid_max",
        }
    )
    gfs_dist = gfs_dist.merge(gust, on="forecast_kst_dtm", how="left")

    out = base.merge(ldaps_dist, on="forecast_kst_dtm", how="left")
    out = out.merge(gfs_dist, on="forecast_kst_dtm", how="left")
    out = add_lead_feature(out, ldaps_df)
    out = out.sort_values("forecast_kst_dtm").ffill().fillna(0).reset_index(drop=True)
    return out


def build_extended_scada_targets(scada_df, group):
    scada = scada_df.copy()
    scada["kst_dtm"] = pd.to_datetime(scada["kst_dtm"])
    scada["hour"] = scada["kst_dtm"].dt.floor("h")
    ws_cols = [f"{prefix}_ws" for prefix in GROUP_TURBINE_PREFIXES[group]]
    hourly = scada.groupby("hour")[ws_cols].mean()
    values = hourly.to_numpy(dtype=float)

    out = pd.DataFrame({"forecast_kst_dtm": hourly.index})
    out["scada_ws_mean"] = np.nanmean(values, axis=1)
    out["scada_ws_std"] = np.nanstd(values, axis=1)
    out["scada_ws_p10"] = np.nanpercentile(values, 10, axis=1)
    out["scada_ws_p50"] = np.nanpercentile(values, 50, axis=1)
    out["scada_ws_p75"] = np.nanpercentile(values, 75, axis=1)
    out["scada_ws_p90"] = np.nanpercentile(values, 90, axis=1)
    out["scada_ws_max"] = np.nanmax(values, axis=1)
    out["scada_ws_cubic"] = np.cbrt(np.nanmean(np.clip(values, 0, None) ** 3, axis=1))
    out["scada_ws_ramp"] = out["scada_ws_mean"].diff().fillna(0)
    return out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def extended_feature_cols(weather):
    blocked = set(TIME_KEY_COLS + ["rho", "doy", "moy", "hod"])
    return [col for col in weather.columns if col not in blocked]


def fit_extended_teacher(weather, scada_df, group, fit_before=VAL_START):
    targets = build_extended_scada_targets(scada_df, group)
    df = weather.merge(targets, on="forecast_kst_dtm", how="inner").dropna()
    if fit_before is not None:
        df = df[df["forecast_kst_dtm"] < pd.Timestamp(fit_before)]
    feature_cols = extended_feature_cols(weather)
    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, min_samples_leaf=10, random_state=42, n_jobs=-1)
    )
    model.fit(df[feature_cols], df[EXT_TARGETS])
    return feature_cols, model


def apply_extended_teacher(weather, teacher, v_mode):
    feature_cols, model = teacher
    out = weather.copy()
    pred = pd.DataFrame(model.predict(out[feature_cols]), columns=EXT_TARGETS, index=out.index)
    for col in EXT_TARGETS:
        out[col] = np.clip(pred[col], 0, None)

    if v_mode == "mean":
        v = out["scada_ws_mean"]
    elif v_mode == "cubic":
        v = out["scada_ws_cubic"]
    elif v_mode == "p75":
        v = out["scada_ws_p75"]
    elif v_mode == "p90":
        v = out["scada_ws_p90"]
    elif v_mode == "mix_cubic_p90":
        v = 0.5 * out["scada_ws_cubic"] + 0.5 * out["scada_ws_p90"]
    else:
        raise ValueError(f"unknown v_mode: {v_mode}")

    spread_sigma = (out["scada_ws_p90"] - out["scada_ws_p10"]) / 2.563
    out["v"] = np.clip(v, 0, None)
    out["v_std"] = np.clip(0.5 * out["scada_ws_std"] + 0.5 * spread_sigma, 0.05, None)
    return out


def build_variant_weather(variant, weather_base, weather_ext, scada_by_name):
    by_group = {}
    for group, scada_name in BASE_GROUP_SCADA.items():
        scada = scada_by_name[scada_name]
        if variant == "standard":
            teacher = fit_scada_wind_teacher(weather_base, scada, group, fit_before=VAL_START)
            by_group[group] = apply_scada_wind_teacher(weather_base, teacher)
        else:
            teacher = fit_extended_teacher(weather_ext, scada, group, fit_before=VAL_START)
            by_group[group] = apply_extended_teacher(weather_ext, teacher, variant)

    return {
        "vestas": {group: by_group[group] for group in [GROUP for GROUP in by_group if GROUP_MANUFACTURER[GROUP] == "vestas"]},
        "unison": {group: by_group[group] for group in [GROUP for GROUP in by_group if GROUP_MANUFACTURER[GROUP] == "unison"]},
    }


def run_variant(variant, weather_by_manufacturer, labels, stage1_epochs, stage2_epochs):
    rows = []
    for manufacturer, weather_by_group in weather_by_manufacturer.items():
        _, _, stage1, stage2 = train_manufacturer(
            manufacturer,
            weather_by_group,
            labels,
            verbose=False,
            save=False,
            stage1_epochs=stage1_epochs,
            stage2_epochs=stage2_epochs,
        )
        stage1["stage"] = "physics_only"
        stage2["stage"] = "with_bias"
        rows.append(stage1)
        rows.append(stage2)
    out = pd.concat(rows, ignore_index=True)
    out["variant"] = variant
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", default="p90,cubic,mix_cubic_p90")
    parser.add_argument("--stage1-epochs", type=int, default=500)
    parser.add_argument("--stage2-epochs", type=int, default=2000)
    args = parser.parse_args()

    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    scada_by_name = {
        "vestas": pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig"),
        "unison": pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig"),
    }

    weather_base = build_pinn_weather(ldaps_train, gfs_train)
    weather_ext = build_extended_pinn_weather(ldaps_train, gfs_train)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    results = []
    for variant in variants:
        print(f"\n=== variant: {variant} ===")
        weather_by_manufacturer = build_variant_weather(variant, weather_base, weather_ext, scada_by_name)
        result = run_variant(variant, weather_by_manufacturer, labels, args.stage1_epochs, args.stage2_epochs)
        results.append(result)
        print(result[result["stage"] == "with_bias"].to_string(index=False))

    final = pd.concat(results, ignore_index=True)
    final.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")

    with_bias = final[final["stage"] == "with_bias"]
    summary = with_bias.groupby("variant").agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"))
    print("\n=== mean with_bias summary ===")
    print(summary.sort_values("score", ascending=False).to_string())
    return final


if __name__ == "__main__":
    main()

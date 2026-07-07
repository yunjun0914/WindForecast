import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from utils.pinn_physics import air_density
from utils.power_curve import GROUP_N_TURBINES, GROUP_TURBINE_PREFIXES
from utils.preprocessing import FARM_CENTROID, haversine_km

GROUP_MANUFACTURER = {
    "kpx_group_1": "vestas",
    "kpx_group_2": "vestas",
    "kpx_group_3": "unison",
}

MANUFACTURER_SCADA_WS_COLS = {
    "vestas": [f"vestas_wtg{i:02d}_ws" for i in range(1, 13)],
    "unison": [f"unison_wtg{i:02d}_ws" for i in range(1, 6)],
}

# GROUP_N_TURBINES (imported above) matters here too: models/pinn.py's physics
# equation is per-turbine, so group-level predictions need P_phys * GROUP_N_TURBINES[group]

# 95th percentile of empirically implied C_eff from SCADA, v>=9 m/s (estimate_cp_max.py)
C_MAX_BY_MANUFACTURER = {
    "vestas": 0.4235,
    "unison": 0.4882,
}

CUT_IN_SPEED = 3.5
RATED_SPEED = 12.0
CUT_OUT_SPEED = 25.0

HUB_HEIGHT_M = 117.0
WIND_SHEAR_ALPHA = 0.14  # standard open-terrain exponent for the power-law extrapolation
WIND_CORRECTION_FEATURES = [
    "v",
    "ldaps_ws50_max",
    "ldaps_ws50_min",
    "ldaps_ws5_bl",
    "ldaps_blh",
    "gfs_ws80",
    "gfs_ws100",
    "gfs_ws850",
    "gfs_850_u",
    "gfs_850_v",
    "gfs_gust",
]
RIDGE_ALPHA = 1e-3
SCADA_TEACHER_TARGETS = ["scada_ws_mean", "scada_ws_std", "scada_ws_p10", "scada_ws_p50", "scada_ws_p90"]


def _nearest_gfs_features(gfs_df):
    grids = gfs_df[["grid_id", "latitude", "longitude"]].drop_duplicates()
    dist = haversine_km(FARM_CENTROID[0], FARM_CENTROID[1], grids["latitude"], grids["longitude"])
    grid_id = grids.loc[dist.idxmin(), "grid_id"]
    g = gfs_df.loc[gfs_df["grid_id"] == grid_id].copy()
    g["gfs_ws850"] = np.sqrt(g["isobaricInhPa_850_u"] ** 2 + g["isobaricInhPa_850_v"] ** 2)
    g["gfs_ws80"] = np.sqrt(g["heightAboveGround_80_u"] ** 2 + g["heightAboveGround_80_v"] ** 2)
    g["gfs_ws100"] = np.sqrt(g["heightAboveGround_100_100u"] ** 2 + g["heightAboveGround_100_100v"] ** 2)
    return g[
        [
            "forecast_kst_dtm",
            "gfs_ws80",
            "gfs_ws100",
            "gfs_ws850",
            "isobaricInhPa_850_u",
            "isobaricInhPa_850_v",
            "surface_0_gust",
        ]
    ].rename(
        columns={
            "isobaricInhPa_850_u": "gfs_850_u",
            "isobaricInhPa_850_v": "gfs_850_v",
            "surface_0_gust": "gfs_gust",
        }
    )


def build_pinn_weather(ldaps_df, gfs_df=None):
    """Hub-height wind speed v, air density rho, and calendar fields, from LDAPS
    (16-grid mean). GFS's 100m wind was tried first but reads ~2x low against SCADA
    (0.25deg/~28km grid can't resolve this ridge-top site's terrain speed-up); LDAPS's
    1.5km grid tracks SCADA far better (corr 0.80 vs 0.70) even extrapolated up from
    only 10m via a standard power-law profile: v_hub = v_10m*(hub_height/10)^alpha.
    See docs/pinn_plan.md for the full comparison."""
    wind_speed_10m = np.sqrt(
        ldaps_df["heightAboveGround_10_10u"] ** 2 + ldaps_df["heightAboveGround_10_10v"] ** 2
    )
    tmp = ldaps_df.assign(_v_10m=wind_speed_10m)
    cols = [
        "heightAboveGround_10_10u",
        "heightAboveGround_10_10v",
        "heightAboveGround_50_50MUmax",
        "heightAboveGround_50_50MVmax",
        "heightAboveGround_50_50MUmin",
        "heightAboveGround_50_50MVmin",
        "heightAboveGround_5_XBLWS",
        "heightAboveGround_5_YBLWS",
        "heightAboveGround_2_t",
        "surface_0_sp",
        "etc_0_blh",
    ]
    agg = tmp.groupby("forecast_kst_dtm", as_index=False)[cols].mean()
    spread = tmp.groupby("forecast_kst_dtm")["_v_10m"].std(ddof=0).rename("_v_10m_std").reset_index()
    agg = agg.merge(spread, on="forecast_kst_dtm", how="left")
    # forward-fill only (never backfill) any rare source gaps, same policy as
    # utils/preprocessing.py's weather pipeline
    agg = agg.sort_values("forecast_kst_dtm").ffill().reset_index(drop=True)

    out = pd.DataFrame({"forecast_kst_dtm": pd.to_datetime(agg["forecast_kst_dtm"])})
    v_10m = np.sqrt(agg["heightAboveGround_10_10u"] ** 2 + agg["heightAboveGround_10_10v"] ** 2)
    out["v"] = v_10m * (HUB_HEIGHT_M / 10.0) ** WIND_SHEAR_ALPHA
    out["v_std"] = agg["_v_10m_std"].fillna(0).to_numpy() * (HUB_HEIGHT_M / 10.0) ** WIND_SHEAR_ALPHA
    out["ldaps_ws50_max"] = np.sqrt(
        agg["heightAboveGround_50_50MUmax"] ** 2 + agg["heightAboveGround_50_50MVmax"] ** 2
    )
    out["ldaps_ws50_min"] = np.sqrt(
        agg["heightAboveGround_50_50MUmin"] ** 2 + agg["heightAboveGround_50_50MVmin"] ** 2
    )
    out["ldaps_ws5_bl"] = np.sqrt(agg["heightAboveGround_5_XBLWS"] ** 2 + agg["heightAboveGround_5_YBLWS"] ** 2)
    out["ldaps_blh"] = agg["etc_0_blh"].to_numpy()
    out["rho"] = air_density(agg["heightAboveGround_2_t"].to_numpy(), agg["surface_0_sp"].to_numpy())
    out["doy"] = out["forecast_kst_dtm"].dt.dayofyear
    out["moy"] = out["forecast_kst_dtm"].dt.month
    out["hod"] = out["forecast_kst_dtm"].dt.hour
    if gfs_df is not None:
        gfs_features = _nearest_gfs_features(gfs_df)
        gfs_features["forecast_kst_dtm"] = pd.to_datetime(gfs_features["forecast_kst_dtm"])
        out = out.merge(gfs_features, on="forecast_kst_dtm", how="left").ffill()
    return out


def fit_wind_speed_correction(weather_pinn, scada_df, manufacturer, fit_before=None):
    """Even after the hub-height extrapolation, LDAPS v is only corr~0.8 with real
    SCADA wind speed. Fit a simple linear bias correction (v_true ~ a + b*v_ldaps) on
    the training period's own SCADA, to be applied at both train and test time -- the
    correction itself never touches test-period SCADA (which doesn't exist), only the
    LDAPS forecast, so this is leakage-free."""
    scada = scada_df.copy()
    scada["kst_dtm"] = pd.to_datetime(scada["kst_dtm"])
    scada["hour"] = scada["kst_dtm"].dt.floor("h")
    ws_cols = MANUFACTURER_SCADA_WS_COLS[manufacturer]
    scada_v = scada.groupby("hour")[ws_cols].mean().mean(axis=1).rename("scada_v")

    w = weather_pinn.copy()
    w["hour"] = w["forecast_kst_dtm"].dt.floor("h")
    correction_cols = [col for col in WIND_CORRECTION_FEATURES if col in w.columns]
    merged = w.set_index("hour")[correction_cols].join(scada_v).dropna()
    if fit_before is not None:
        merged = merged[merged.index < pd.Timestamp(fit_before)]

    x = merged[correction_cols].to_numpy()
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_std = np.where(x_std < 1e-6, 1.0, x_std)
    x_scaled = (x - x_mean) / x_std
    x_design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    penalty = np.eye(x_design.shape[1]) * RIDGE_ALPHA
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(x_design.T @ x_design + penalty, x_design.T @ merged["scada_v"].to_numpy())
    return correction_cols, x_mean, x_std, coef


def apply_wind_speed_correction(weather_pinn, correction):
    correction_cols, x_mean, x_std, coef = correction
    out = weather_pinn.copy()
    x = out[correction_cols].to_numpy()
    x_scaled = (x - x_mean) / x_std
    x_design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    out["v"] = np.clip(x_design @ coef, 0, None)
    if "v_std" in out.columns:
        v_idx = correction_cols.index("v")
        v_coef = coef[1 + v_idx] / x_std[v_idx]
        out["v_std"] = np.abs(v_coef) * out["v_std"]
    return out


def build_scada_wind_teacher(scada_df, group):
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
    out["scada_ws_p90"] = np.nanpercentile(values, 90, axis=1)
    return out.dropna().reset_index(drop=True)


def _teacher_feature_cols(weather_pinn):
    return [col for col in WIND_CORRECTION_FEATURES if col in weather_pinn.columns]


def fit_scada_wind_teacher(weather_pinn, scada_df, group, fit_before=None):
    teacher = build_scada_wind_teacher(scada_df, group)
    df = weather_pinn.merge(teacher, on="forecast_kst_dtm", how="inner").dropna()
    if fit_before is not None:
        df = df[df["forecast_kst_dtm"] < pd.Timestamp(fit_before)]
    feature_cols = _teacher_feature_cols(weather_pinn)
    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
    )
    model.fit(df[feature_cols], df[SCADA_TEACHER_TARGETS])
    return feature_cols, model


def _apply_scada_wind_teacher_predictions(weather_pinn, pred):
    out = weather_pinn.copy()
    pred_df = pd.DataFrame(pred, columns=SCADA_TEACHER_TARGETS, index=out.index)
    spread_sigma = (pred_df["scada_ws_p90"] - pred_df["scada_ws_p10"]) / 2.563
    out["v"] = np.clip(pred_df["scada_ws_mean"], 0, None)
    out["v_std"] = np.clip(0.5 * pred_df["scada_ws_std"] + 0.5 * spread_sigma, 0.05, None)
    for col in SCADA_TEACHER_TARGETS:
        out[col] = pred_df[col]
    return out


def apply_scada_wind_teacher(weather_pinn, teacher_model):
    feature_cols, model = teacher_model
    return _apply_scada_wind_teacher_predictions(weather_pinn, model.predict(weather_pinn[feature_cols]))


def apply_scada_wind_teacher_oob(weather_pinn, scada_df, group, fit_before=None):
    teacher = build_scada_wind_teacher(scada_df, group)
    weather_indexed = weather_pinn.reset_index().rename(columns={"index": "_weather_idx"})
    df = weather_indexed.merge(teacher, on="forecast_kst_dtm", how="inner").dropna()
    if fit_before is not None:
        df = df[df["forecast_kst_dtm"] < pd.Timestamp(fit_before)]
    feature_cols = _teacher_feature_cols(weather_pinn)
    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        oob_score=True,
        bootstrap=True,
    )
    model.fit(df[feature_cols], df[SCADA_TEACHER_TARGETS])

    pred = model.predict(weather_pinn[feature_cols])
    oob_pred = np.asarray(model.oob_prediction_)
    if oob_pred.ndim == 1:
        oob_pred = oob_pred.reshape(-1, 1)
    valid_oob = np.isfinite(oob_pred).all(axis=1)
    pred[df.loc[valid_oob, "_weather_idx"].to_numpy(dtype=int)] = oob_pred[valid_oob]
    return _apply_scada_wind_teacher_predictions(weather_pinn, pred)


def build_group_pinn_dataset(weather_pinn, labels_df, group):
    """Merge shared PINN weather inputs with one KPX group's labels (drops rows with
    no label for that group, e.g. kpx_group_3's 2022 gap)."""
    labels = labels_df[["kst_dtm", group]].copy()
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    merged = weather_pinn.merge(labels, left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner")
    merged = merged.dropna(subset=[group]).drop(columns=["kst_dtm"]).reset_index(drop=True)
    return merged.rename(columns={group: "y"})

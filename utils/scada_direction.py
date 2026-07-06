import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from evaluate_pinn_effective_wind_teacher import extended_feature_cols
from utils.power_curve import GROUP_TURBINE_PREFIXES


DIRECTION_TARGETS = [
    "scada_wd_sin",
    "scada_wd_cos",
    "scada_wd_concentration",
    "scada_wd_sin_std",
    "scada_wd_cos_std",
]


def build_scada_direction_targets(scada_df, group):
    scada = scada_df.copy()
    scada["kst_dtm"] = pd.to_datetime(scada["kst_dtm"])
    scada["hour"] = scada["kst_dtm"].dt.floor("h")
    wd_cols = [f"{prefix}_wd" for prefix in GROUP_TURBINE_PREFIXES[group]]
    radians = np.radians(scada[wd_cols].astype(float))
    sin_cols = np.sin(radians)
    cos_cols = np.cos(radians)
    sin_cols["hour"] = scada["hour"]
    cos_cols["hour"] = scada["hour"]

    hourly_sin = sin_cols.groupby("hour").mean()
    hourly_cos = cos_cols.groupby("hour").mean()
    sin_mean = hourly_sin.mean(axis=1)
    cos_mean = hourly_cos.mean(axis=1)
    out = pd.DataFrame({"forecast_kst_dtm": sin_mean.index})
    out["scada_wd_sin"] = sin_mean.to_numpy()
    out["scada_wd_cos"] = cos_mean.to_numpy()
    out["scada_wd_concentration"] = np.sqrt(out["scada_wd_sin"] ** 2 + out["scada_wd_cos"] ** 2)
    out["scada_wd_sin_std"] = hourly_sin.std(axis=1).fillna(0).to_numpy()
    out["scada_wd_cos_std"] = hourly_cos.std(axis=1).fillna(0).to_numpy()
    return out.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def _make_direction_model(kind):
    if kind == "rf":
        return MultiOutputRegressor(RandomForestRegressor(n_estimators=120, min_samples_leaf=12, random_state=42, n_jobs=-1))
    if kind == "rf_hist_gbr_avg":
        rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=120, min_samples_leaf=12, random_state=42, n_jobs=-1))
        hist = MultiOutputRegressor(
            HistGradientBoostingRegressor(max_iter=220, learning_rate=0.04, l2_regularization=0.01, random_state=42)
        )
        return rf, hist
    raise ValueError(kind)


def fit_direction_teacher(weather, scada_df, group, fit_before, kind="rf_hist_gbr_avg"):
    targets = build_scada_direction_targets(scada_df, group)
    df = weather.merge(targets, on="forecast_kst_dtm", how="inner").dropna()
    if fit_before is not None:
        df = df[df["forecast_kst_dtm"] < pd.Timestamp(fit_before)]
    feature_cols = extended_feature_cols(weather)
    model = _make_direction_model(kind)
    if isinstance(model, tuple):
        for m in model:
            m.fit(df[feature_cols], df[DIRECTION_TARGETS])
    else:
        model.fit(df[feature_cols], df[DIRECTION_TARGETS])
    return feature_cols, model


def apply_direction_teacher(weather, teacher, group):
    feature_cols, model = teacher
    out = weather.copy()
    if isinstance(model, tuple):
        pred = sum(m.predict(out[feature_cols]) for m in model) / len(model)
    else:
        pred = model.predict(out[feature_cols])
    pred = pd.DataFrame(pred, columns=DIRECTION_TARGETS, index=out.index)
    pred["scada_wd_concentration"] = np.clip(pred["scada_wd_concentration"], 0, 1)
    for col in DIRECTION_TARGETS:
        out[f"{group}_{col}_teacher"] = pred[col]
    out[f"{group}_scada_wd_sin_x_ws100"] = out[f"{group}_scada_wd_sin_teacher"] * out.get("gfs_ws100_speed", 0)
    out[f"{group}_scada_wd_cos_x_ws100"] = out[f"{group}_scada_wd_cos_teacher"] * out.get("gfs_ws100_speed", 0)
    return out

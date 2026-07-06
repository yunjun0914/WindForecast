import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from utils.pinn_data import SCADA_TEACHER_TARGETS, build_scada_wind_teacher
from utils.preprocessing import TIME_KEY_COLS

TEACHER_FEATURE_PREFIX = "pred_"


def _feature_cols(weather_df):
    return [c for c in weather_df.columns if c not in TIME_KEY_COLS]


def fit_scada_teacher(weather_df, scada_df, group, fit_before=None):
    """Fit forecast-weather -> SCADA wind-distribution teacher for one group.

    `fit_before` is used for honest time-holdout evaluation: the teacher only sees
    SCADA targets before the validation period, but its predictions can be generated
    for both train and validation weather rows.
    """
    teacher = build_scada_wind_teacher(scada_df, group)
    weather = weather_df.copy()
    weather["forecast_kst_dtm"] = pd.to_datetime(weather["forecast_kst_dtm"])
    teacher["forecast_kst_dtm"] = pd.to_datetime(teacher["forecast_kst_dtm"])
    merged = weather.merge(teacher, on="forecast_kst_dtm", how="inner").dropna()
    if fit_before is not None:
        merged = merged[merged["forecast_kst_dtm"] < pd.Timestamp(fit_before)]

    feature_cols = _feature_cols(weather_df)
    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
    )
    model.fit(merged[feature_cols], merged[SCADA_TEACHER_TARGETS])
    return feature_cols, model


def add_scada_teacher_features(weather_df, teacher_model):
    feature_cols, model = teacher_model
    out = weather_df.copy()
    pred = model.predict(out[feature_cols])
    pred_df = pd.DataFrame(
        pred,
        columns=[f"{TEACHER_FEATURE_PREFIX}{c}" for c in SCADA_TEACHER_TARGETS],
        index=out.index,
    )
    for col in pred_df.columns:
        out[col] = np.clip(pred_df[col], 0, None)

    out["pred_scada_ws_iqr"] = out["pred_scada_ws_p90"] - out["pred_scada_ws_p10"]
    out["pred_scada_ws_sigma_q"] = out["pred_scada_ws_iqr"] / 2.563
    out["pred_scada_ws_mean_minus_gfs100"] = out["pred_scada_ws_mean"] - out.get("gfs_ws100_speed", 0)
    out["pred_scada_ws_mean_minus_gfs850"] = out["pred_scada_ws_mean"] - out.get("gfs_ws850_speed", 0)
    return out


def add_honest_scada_teacher_features(weather_df, scada_df, group, val_start):
    teacher = fit_scada_teacher(weather_df, scada_df, group, fit_before=val_start)
    return add_scada_teacher_features(weather_df, teacher)


def add_full_scada_teacher_features(train_weather_df, pred_weather_df, scada_df, group):
    teacher = fit_scada_teacher(train_weather_df, scada_df, group)
    return add_scada_teacher_features(pred_weather_df, teacher)

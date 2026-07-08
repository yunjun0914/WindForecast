import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.preprocessing import TIME_KEY_COLS


SCADA_DIRECTION_TARGETS = [
    "scada_wd_sin",
    "scada_wd_cos",
    "scada_wd_concentration",
    "scada_wd_spread",
    "scada_wd_sin_std",
    "scada_wd_cos_std",
    "scada_ws_dir_sin",
    "scada_ws_dir_cos",
]


def build_scada_direction_targets(scada_df, group):
    scada = scada_df.copy()
    scada["kst_dtm"] = pd.to_datetime(scada["kst_dtm"])
    scada["hour"] = scada["kst_dtm"].dt.floor("h")

    prefixes = GROUP_TURBINE_PREFIXES[group]
    wd_cols = [f"{prefix}_wd" for prefix in prefixes]
    ws_cols = [f"{prefix}_ws" for prefix in prefixes]
    cols = ["hour", *wd_cols, *ws_cols]
    scada = scada[cols].replace([np.inf, -np.inf], np.nan)

    rows = []
    for hour, part in scada.groupby("hour", sort=True):
        wd = part[wd_cols].to_numpy(dtype=float).reshape(-1)
        ws = part[ws_cols].to_numpy(dtype=float).reshape(-1)
        valid_wd = np.isfinite(wd)
        if not valid_wd.any():
            continue

        angle = np.deg2rad(np.mod(wd[valid_wd], 360.0))
        sin_values = np.sin(angle)
        cos_values = np.cos(angle)
        sin_mean = float(np.mean(sin_values))
        cos_mean = float(np.mean(cos_values))
        concentration = float(np.sqrt(sin_mean**2 + cos_mean**2))

        valid_vec = valid_wd & np.isfinite(ws)
        if valid_vec.any():
            angle_vec = np.deg2rad(np.mod(wd[valid_vec], 360.0))
            ws_vec = np.clip(ws[valid_vec], 0.0, None)
            ws_dir_sin = float(np.mean(ws_vec * np.sin(angle_vec)))
            ws_dir_cos = float(np.mean(ws_vec * np.cos(angle_vec)))
        else:
            ws_dir_sin = 0.0
            ws_dir_cos = 0.0

        rows.append(
            {
                "forecast_kst_dtm": hour,
                "scada_wd_sin": sin_mean,
                "scada_wd_cos": cos_mean,
                "scada_wd_concentration": concentration,
                "scada_wd_spread": 1.0 - concentration,
                "scada_wd_sin_std": float(np.std(sin_values)),
                "scada_wd_cos_std": float(np.std(cos_values)),
                "scada_ws_dir_sin": ws_dir_sin,
                "scada_ws_dir_cos": ws_dir_cos,
            }
        )

    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def _feature_cols(weather):
    blocked = set(TIME_KEY_COLS + SCADA_DIRECTION_TARGETS)
    return [col for col in weather.columns if col not in blocked]


def _make_teacher(seed=42):
    return MultiOutputRegressor(
        LGBMRegressor(
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
            n_estimators=700,
            learning_rate=0.035,
            num_leaves=48,
            min_child_samples=80,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=2.0,
        )
    )


def _apply_predictions(weather, pred):
    out = weather.copy()
    pred = pd.DataFrame(pred, columns=SCADA_DIRECTION_TARGETS, index=out.index)
    for col in SCADA_DIRECTION_TARGETS:
        out[col] = pred[col].to_numpy(dtype=float)

    out["scada_wd_concentration"] = np.clip(out["scada_wd_concentration"], 0.0, 1.0)
    out["scada_wd_spread"] = np.clip(out["scada_wd_spread"], 0.0, 1.0)
    return out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)


def add_scada_direction_teacher_oof(train_weather, pred_weather, scada_df, group, n_splits=3):
    """Add weather-predicted SCADA wind-direction features.

    Train rows use time-fold OOF teacher predictions. Prediction rows use a teacher
    fitted on all train-period SCADA. The raw SCADA wind direction never enters the
    validation/test rows directly.
    """
    targets = build_scada_direction_targets(scada_df, group)
    if targets.empty:
        return train_weather.copy(), pred_weather.copy()

    feature_cols = _feature_cols(train_weather)
    weather_indexed = train_weather.reset_index().rename(columns={"index": "_weather_idx"})
    df = weather_indexed.merge(targets, on="forecast_kst_dtm", how="inner").dropna()
    df = df.sort_values("forecast_kst_dtm").reset_index(drop=True)
    if len(df) < 100:
        return train_weather.copy(), pred_weather.copy()

    full_model = _make_teacher(seed=42)
    full_model.fit(df[feature_cols], df[SCADA_DIRECTION_TARGETS])
    train_pred = full_model.predict(train_weather[feature_cols])

    n_folds = min(int(n_splits), len(df))
    if n_folds >= 2:
        for fold_id, fold_idx in enumerate(np.array_split(np.arange(len(df)), n_folds)):
            if len(fold_idx) == 0:
                continue
            fit_idx = np.setdiff1d(np.arange(len(df)), fold_idx, assume_unique=True)
            if len(fit_idx) == 0:
                continue
            model = _make_teacher(seed=43 + fold_id)
            model.fit(df.iloc[fit_idx][feature_cols], df.iloc[fit_idx][SCADA_DIRECTION_TARGETS])
            oof_pred = model.predict(df.iloc[fold_idx][feature_cols])
            train_pred[df.iloc[fold_idx]["_weather_idx"].to_numpy(dtype=int)] = oof_pred

    train_out = _apply_predictions(train_weather, train_pred)
    pred_out = _apply_predictions(pred_weather, full_model.predict(pred_weather[feature_cols]))
    return train_out, pred_out

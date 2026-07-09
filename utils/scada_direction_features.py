import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
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
    if scada.empty:
        return pd.DataFrame(columns=["forecast_kst_dtm", *SCADA_DIRECTION_TARGETS])

    n_turbines = len(prefixes)
    hour = np.repeat(scada["hour"].to_numpy(), n_turbines)
    wd = scada[wd_cols].to_numpy(dtype=float).reshape(-1)
    ws = scada[ws_cols].to_numpy(dtype=float).reshape(-1)
    valid_wd = np.isfinite(wd)
    if not valid_wd.any():
        return pd.DataFrame(columns=["forecast_kst_dtm", *SCADA_DIRECTION_TARGETS])

    angle = np.deg2rad(np.mod(wd[valid_wd], 360.0))
    long = pd.DataFrame(
        {
            "hour": hour[valid_wd],
            "sin": np.sin(angle),
            "cos": np.cos(angle),
            "ws": ws[valid_wd],
        }
    )
    long["sin_sq"] = long["sin"] ** 2
    long["cos_sq"] = long["cos"] ** 2

    grouped = long.groupby("hour", sort=True).agg(
        scada_wd_sin=("sin", "mean"),
        scada_wd_cos=("cos", "mean"),
        sin_sq_mean=("sin_sq", "mean"),
        cos_sq_mean=("cos_sq", "mean"),
    )
    grouped["scada_wd_concentration"] = np.sqrt(grouped["scada_wd_sin"] ** 2 + grouped["scada_wd_cos"] ** 2)
    grouped["scada_wd_spread"] = 1.0 - grouped["scada_wd_concentration"]
    grouped["scada_wd_sin_std"] = np.sqrt(
        np.clip(grouped["sin_sq_mean"] - grouped["scada_wd_sin"] ** 2, 0.0, None)
    )
    grouped["scada_wd_cos_std"] = np.sqrt(
        np.clip(grouped["cos_sq_mean"] - grouped["scada_wd_cos"] ** 2, 0.0, None)
    )

    vec = long[np.isfinite(long["ws"])].copy()
    if len(vec) > 0:
        vec["ws_clipped"] = np.clip(vec["ws"].to_numpy(dtype=float), 0.0, None)
        vec["ws_dir_sin"] = vec["ws_clipped"] * vec["sin"]
        vec["ws_dir_cos"] = vec["ws_clipped"] * vec["cos"]
        vec_grouped = vec.groupby("hour", sort=True).agg(
            scada_ws_dir_sin=("ws_dir_sin", "mean"),
            scada_ws_dir_cos=("ws_dir_cos", "mean"),
        )
        grouped = grouped.merge(vec_grouped, left_index=True, right_index=True, how="left")
    else:
        grouped["scada_ws_dir_sin"] = 0.0
        grouped["scada_ws_dir_cos"] = 0.0

    grouped[["scada_ws_dir_sin", "scada_ws_dir_cos"]] = grouped[
        ["scada_ws_dir_sin", "scada_ws_dir_cos"]
    ].fillna(0.0)
    grouped = grouped.drop(columns=["sin_sq_mean", "cos_sq_mean"]).reset_index()
    grouped = grouped.rename(columns={"hour": "forecast_kst_dtm"})
    return grouped.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)


def _feature_cols(weather):
    blocked = set(TIME_KEY_COLS + SCADA_DIRECTION_TARGETS)
    return [col for col in weather.columns if col not in blocked]


def _make_teacher(seed=42, backend="extra_trees", teacher_params=None):
    params = {} if teacher_params is None else dict(teacher_params)
    if backend == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=int(params.get("n_estimators", 80)),
            max_depth=None if params.get("max_depth", 18) is None else int(params.get("max_depth", 18)),
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            max_features=float(params.get("max_features", 0.70)),
            random_state=seed,
            n_jobs=-1,
        )
    if backend == "random_forest":
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 120)),
            max_depth=None if params.get("max_depth", 18) is None else int(params.get("max_depth", 18)),
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            max_features=float(params.get("max_features", 0.70)),
            random_state=seed,
            n_jobs=-1,
        )
    if backend == "lgbm":
        return MultiOutputRegressor(
            LGBMRegressor(
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
                n_estimators=int(params.get("n_estimators", 700)),
                learning_rate=float(params.get("learning_rate", 0.035)),
                num_leaves=int(params.get("num_leaves", 48)),
                min_child_samples=int(params.get("min_child_samples", 80)),
                subsample=float(params.get("subsample", 0.85)),
                colsample_bytree=float(params.get("colsample_bytree", 0.85)),
                reg_alpha=float(params.get("reg_alpha", 0.05)),
                reg_lambda=float(params.get("reg_lambda", 2.0)),
            )
        )
    raise ValueError(f"unknown SCADA direction teacher backend: {backend}")


def _apply_predictions(weather, pred):
    out = weather.copy()
    pred = pd.DataFrame(pred, columns=SCADA_DIRECTION_TARGETS, index=out.index)
    for col in SCADA_DIRECTION_TARGETS:
        out[col] = pred[col].to_numpy(dtype=float)

    out["scada_wd_concentration"] = np.clip(out["scada_wd_concentration"], 0.0, 1.0)
    out["scada_wd_spread"] = np.clip(out["scada_wd_spread"], 0.0, 1.0)
    return out.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)


def add_scada_direction_teacher_oof(
    train_weather,
    pred_weather,
    scada_df,
    group,
    n_splits=3,
    backend="extra_trees",
    teacher_params=None,
):
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

    full_model = _make_teacher(seed=42, backend=backend, teacher_params=teacher_params)
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
            model = _make_teacher(seed=43 + fold_id, backend=backend, teacher_params=teacher_params)
            model.fit(df.iloc[fit_idx][feature_cols], df.iloc[fit_idx][SCADA_DIRECTION_TARGETS])
            oof_pred = model.predict(df.iloc[fold_idx][feature_cols])
            train_pred[df.iloc[fold_idx]["_weather_idx"].to_numpy(dtype=int)] = oof_pred

    train_out = _apply_predictions(train_weather, train_pred)
    pred_out = _apply_predictions(pred_weather, full_model.predict(pred_weather[feature_cols]))
    return train_out, pred_out

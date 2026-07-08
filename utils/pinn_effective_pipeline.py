import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from models.pinn import PowerCurvePINN, TurbineGroupBias
from train_pinn import (
    BIAS_EPS,
    BIAS_LR,
    DEVICE,
    GAMMA,
    HOUR_BIAS_LR,
    LAMBDA,
    LR,
    MOY_BIAS_LR,
    SCADA_WD_AMPLITUDE,
    STAGE1_EPOCHS,
    STAGE2_EPOCHS,
    DOW_BIAS_LR,
    USE_DOW_BIAS,
    USE_MOY_BIAS,
    USE_SCADA_WD_CORRECTION,
    USE_TRAIN_ONLY_HOUR_BIAS,
    USE_TRAIN_ONLY_YEAR_BIAS,
    WD_BIAS_LR,
    YEAR_BIAS_LR,
    group_prediction,
    physics_losses,
)
from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr
from utils.meteo_features import add_lead_feature, aggregate_speed_distribution
from utils.pinn_data import C_MAX_BY_MANUFACTURER, GROUP_N_TURBINES, build_group_pinn_dataset, build_pinn_weather
from utils.pinn_losses import bias_l2, data_loss, hour_bias_abs_summary, soft_threshold_train_only_hour_bias
from utils.pinn_physics import MANUFACTURER_AREA, SINGLE_TURBINE_CAPACITY_W
from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.preprocessing import TIME_KEY_COLS


GROUP1 = "kpx_group_1"
GROUP2 = "kpx_group_2"
GROUP3 = "kpx_group_3"
GROUPS = [GROUP1, GROUP2, GROUP3]
TEACHER_BACKEND = "rf_oob"
TEACHER_TIME_FOLDS = 5
EARLY_STOPPING = True
EARLY_STOP_MIN_DELTA = 1e-5
STAGE1_EVAL_INTERVAL = 10
STAGE1_PATIENCE = 12
STAGE2_EVAL_INTERVAL = 50
STAGE2_PATIENCE = 10

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


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filter_forecast_years(df, years):
    out = df.copy()
    out["forecast_kst_dtm"] = pd.to_datetime(out["forecast_kst_dtm"])
    return out[out["forecast_kst_dtm"].dt.year.isin(years)].reset_index(drop=True)


def filter_label_years(df, years):
    out = df.copy()
    out["kst_dtm"] = pd.to_datetime(out["kst_dtm"])
    return out[out["kst_dtm"].dt.year.isin(years)].reset_index(drop=True)


def filter_scada_years(df, years):
    out = df.copy()
    out["kst_dtm"] = pd.to_datetime(out["kst_dtm"])
    return out[out["kst_dtm"].dt.year.isin(years)].reset_index(drop=True)


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
    ]
    gfs_dist = aggregate_speed_distribution(gfs, gfs_specs, "gfs", ["forecast_kst_dtm"])
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
    hourly = hourly.dropna(how="all")
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
    blocked = set(TIME_KEY_COLS + ["rho", "doy", "moy", "hod", "dow"])
    return [col for col in weather.columns if col not in blocked]


def fit_extended_teacher(weather, scada_df, group, fit_before=None):
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


def _make_lgbm_teacher(seed=42):
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


def _apply_extended_teacher_predictions(weather, pred, v_mode):
    out = weather.copy()
    pred = pd.DataFrame(pred, columns=EXT_TARGETS, index=out.index)
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


def apply_extended_teacher(weather, teacher, v_mode):
    feature_cols, model = teacher
    return _apply_extended_teacher_predictions(weather, model.predict(weather[feature_cols]), v_mode)


def _apply_extended_teacher_rf_oob(weather_train, weather_pred, scada_df, group, v_mode):
    """Fit teacher on all train years, but use RF OOB predictions for train rows."""
    targets = build_extended_scada_targets(scada_df, group)
    weather_indexed = weather_train.reset_index().rename(columns={"index": "_weather_idx"})
    df = weather_indexed.merge(targets, on="forecast_kst_dtm", how="inner").dropna()
    feature_cols = extended_feature_cols(weather_train)
    model = RandomForestRegressor(
        n_estimators=100,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
        oob_score=True,
        bootstrap=True,
    )
    model.fit(df[feature_cols], df[EXT_TARGETS])

    train_pred = model.predict(weather_train[feature_cols])
    oob_pred = np.asarray(model.oob_prediction_)
    if oob_pred.ndim == 1:
        oob_pred = oob_pred.reshape(-1, 1)
    valid_oob = np.isfinite(oob_pred).all(axis=1)
    train_pred[df.loc[valid_oob, "_weather_idx"].to_numpy(dtype=int)] = oob_pred[valid_oob]

    train_teacher = _apply_extended_teacher_predictions(weather_train, train_pred, v_mode)
    pred_teacher = _apply_extended_teacher_predictions(weather_pred, model.predict(weather_pred[feature_cols]), v_mode)
    return train_teacher, pred_teacher


def _apply_extended_teacher_lgbm_time_oof(weather_train, weather_pred, scada_df, group, v_mode):
    """Fit LGBM teacher on train years, using time-fold OOF predictions for train rows."""
    targets = build_extended_scada_targets(scada_df, group)
    weather_indexed = weather_train.reset_index().rename(columns={"index": "_weather_idx"})
    df = weather_indexed.merge(targets, on="forecast_kst_dtm", how="inner").dropna()
    df = df.sort_values("forecast_kst_dtm").reset_index(drop=True)
    feature_cols = extended_feature_cols(weather_train)

    full_model = _make_lgbm_teacher(seed=42)
    full_model.fit(df[feature_cols], df[EXT_TARGETS])
    train_pred = full_model.predict(weather_train[feature_cols])

    n_folds = min(TEACHER_TIME_FOLDS, len(df))
    if n_folds >= 2:
        for fold_id, fold_idx in enumerate(np.array_split(np.arange(len(df)), n_folds)):
            if len(fold_idx) == 0:
                continue
            fit_idx = np.setdiff1d(np.arange(len(df)), fold_idx, assume_unique=True)
            if len(fit_idx) == 0:
                continue
            model = _make_lgbm_teacher(seed=42 + fold_id + 1)
            model.fit(df.iloc[fit_idx][feature_cols], df.iloc[fit_idx][EXT_TARGETS])
            oof_pred = model.predict(df.iloc[fold_idx][feature_cols])
            train_pred[df.iloc[fold_idx]["_weather_idx"].to_numpy(dtype=int)] = oof_pred

    train_teacher = _apply_extended_teacher_predictions(weather_train, train_pred, v_mode)
    pred_teacher = _apply_extended_teacher_predictions(weather_pred, full_model.predict(weather_pred[feature_cols]), v_mode)
    return train_teacher, pred_teacher


def apply_extended_teacher_crossfit(weather_train, weather_pred, scada_df, group, v_mode, backend=None):
    backend = TEACHER_BACKEND if backend is None else backend
    if backend == "rf_oob":
        return _apply_extended_teacher_rf_oob(weather_train, weather_pred, scada_df, group, v_mode)
    if backend == "lgbm_time_oof":
        return _apply_extended_teacher_lgbm_time_oof(weather_train, weather_pred, scada_df, group, v_mode)
    raise ValueError(f"unknown teacher backend: {backend}")


def apply_extended_teacher_oob(weather_train, weather_pred, scada_df, group, v_mode):
    return _apply_extended_teacher_rf_oob(weather_train, weather_pred, scada_df, group, v_mode)


def blend_weather(name, weather_a, weather_b, weight_a):
    out = weather_a.copy()
    blend_cols = [
        "v",
        "v_std",
        "scada_ws_mean",
        "scada_ws_std",
        "scada_ws_p10",
        "scada_ws_p50",
        "scada_ws_p75",
        "scada_ws_p90",
        "scada_ws_cubic",
        "scada_ws_ramp",
        "scada_wd_sin",
        "scada_wd_cos",
        "scada_wd_concentration",
        "scada_wd_spread",
        "scada_wd_sin_std",
        "scada_wd_cos_std",
        "scada_ws_dir_sin",
        "scada_ws_dir_cos",
    ]
    for col in blend_cols:
        if col in weather_a.columns and col in weather_b.columns:
            out[col] = weight_a * weather_a[col].to_numpy() + (1 - weight_a) * weather_b[col].to_numpy()
    out["teacher_recipe"] = name
    return out


def _clone_state_dict(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _clone_bias_states(group_data):
    return {group: _clone_state_dict(data["bias"]) for group, data in group_data.items()}


def _load_bias_states(group_data, states):
    for group, state in states.items():
        group_data[group]["bias"].load_state_dict(state)


def _build_valid_group_data(valid_weather_by_group, valid_labels, groups, group_data):
    if valid_weather_by_group is None or valid_labels is None:
        return {}
    valid_data = {}
    for group in groups:
        if group not in valid_weather_by_group:
            continue
        ds = build_group_pinn_dataset(valid_weather_by_group[group], valid_labels, group)
        if len(ds) == 0:
            continue
        valid_data[group] = {
            "df": ds.copy(),
            "bias": group_data[group]["bias"],
            "capacity": GROUP_CAPACITY_KWH[group],
        }
    return valid_data


def _cache_eval_tensors(eval_data, model=None):
    with torch.no_grad():
        for group, data in eval_data.items():
            df = data["df"]
            data["y"] = torch.tensor(df["y"].to_numpy(), dtype=torch.float32, device=DEVICE)
            data["hod_idx"] = torch.tensor(df["hod"].to_numpy(), dtype=torch.long, device=DEVICE)
            data["moy_idx"] = (
                torch.tensor(df["moy"].to_numpy(), dtype=torch.long, device=DEVICE)
                if USE_MOY_BIAS
                else None
            )
            if USE_DOW_BIAS and "dow" not in df.columns:
                raise ValueError("USE_DOW_BIAS=True requires a dow column")
            data["dow_idx"] = (
                torch.tensor(df["dow"].to_numpy(), dtype=torch.long, device=DEVICE)
                if USE_DOW_BIAS
                else None
            )
            if model is not None:
                data["base_pred"] = group_prediction(
                    model,
                    df,
                    GROUP_N_TURBINES[group],
                    bias=data["bias"] if USE_SCADA_WD_CORRECTION else None,
                    use_calendar_bias=False,
                    use_wind_distribution=True,
                ).detach()


def _score_valid_groups(model, valid_data, use_bias=False, use_cached_base=False):
    if not valid_data:
        return None, []
    rows = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for group, data in valid_data.items():
            if use_cached_base and "base_pred" in data:
                pred = data["base_pred"]
                if use_bias:
                    pred = pred + data["bias"].calendar(data["hod_idx"], data["moy_idx"], data["dow_idx"])
            else:
                bias = data["bias"] if (use_bias or USE_SCADA_WD_CORRECTION) else None
                pred = group_prediction(
                    model,
                    data["df"],
                    GROUP_N_TURBINES[group],
                    bias=bias,
                    use_calendar_bias=use_bias,
                    use_wind_distribution=True,
                )
            pred = torch.clamp(pred, min=0.0, max=data["capacity"])
            y = data.get("y")
            if y is None:
                y = torch.tensor(data["df"]["y"].to_numpy(), dtype=torch.float32, device=DEVICE)
            nmae, ficr = group_nmae_ficr(y.cpu().numpy(), pred.cpu().numpy(), data["capacity"])
            rows.append(
                {
                    "group": group,
                    "score": 0.5 * (1 - nmae) + 0.5 * ficr,
                    "nmae": nmae,
                    "ficr": ficr,
                }
            )
    if was_training:
        model.train()
    score = float(np.nanmean([row["score"] for row in rows])) if rows else None
    return score, rows


def train_full_pinn(
    physical_manufacturer,
    group_weather_train,
    labels,
    model_cls=PowerCurvePINN,
    model_kwargs=None,
    valid_weather_by_group=None,
    valid_labels=None,
    early_stopping=EARLY_STOPPING,
    stage1_early_stopping=True,
    stage2_early_stopping=True,
    stage1_patience=STAGE1_PATIENCE,
    stage2_patience=STAGE2_PATIENCE,
    stage1_eval_interval=STAGE1_EVAL_INTERVAL,
    stage2_eval_interval=STAGE2_EVAL_INTERVAL,
    early_stop_min_delta=EARLY_STOP_MIN_DELTA,
):
    seed_everything(42)
    model_kwargs = {} if model_kwargs is None else model_kwargs
    groups = list(group_weather_train)
    c_max = C_MAX_BY_MANUFACTURER[physical_manufacturer]
    area = MANUFACTURER_AREA[physical_manufacturer]
    turbine_capacity_w = SINGLE_TURBINE_CAPACITY_W[physical_manufacturer]
    model = model_cls(c_max, area, **model_kwargs).to(DEVICE)
    v_pool = np.concatenate([group_weather_train[group]["v"].to_numpy() for group in groups])

    group_data = {}
    for group in groups:
        ds = build_group_pinn_dataset(group_weather_train[group], labels, group)
        train_df = ds.copy()
        train_years = sorted(train_df["forecast_kst_dtm"].dt.year.unique())
        year_to_idx = {year: idx for idx, year in enumerate(train_years)}
        train_df["year_idx"] = train_df["forecast_kst_dtm"].dt.year.map(year_to_idx)
        bias = TurbineGroupBias(
            GROUP_CAPACITY_KWH[group],
            n_train_rows=len(train_df) if USE_TRAIN_ONLY_HOUR_BIAS else None,
            n_train_years=len(train_years) if USE_TRAIN_ONLY_YEAR_BIAS else None,
            use_direction_correction=USE_SCADA_WD_CORRECTION,
            direction_amplitude=SCADA_WD_AMPLITUDE,
        ).to(DEVICE)
        group_data[group] = {
            "train": train_df,
            "bias": bias,
            "row_idx": torch.arange(len(train_df), dtype=torch.long, device=DEVICE),
            "year_idx": torch.tensor(train_df["year_idx"].to_numpy(), dtype=torch.long, device=DEVICE),
            "capacity": GROUP_CAPACITY_KWH[group],
        }

    valid_data = _build_valid_group_data(valid_weather_by_group, valid_labels, groups, group_data)
    use_stage1_early_stop = early_stopping and stage1_early_stopping and bool(valid_data)
    use_stage2_early_stop = early_stopping and stage2_early_stopping and bool(valid_data)
    if use_stage1_early_stop or use_stage2_early_stop:
        _cache_eval_tensors(valid_data)
        print(
            f"[{physical_manufacturer}] early stopping enabled: "
            f"valid_groups={list(valid_data)} "
            f"stage1={'on' if use_stage1_early_stop else 'off'} "
            f"stage2={'on' if use_stage2_early_stop else 'off'} "
            f"stage1_patience={stage1_patience} stage2_patience={stage2_patience}"
        )

    stage1_param_groups = [{"params": list(model.parameters()), "lr": LR, "weight_decay": 0.0}]
    if USE_TRAIN_ONLY_HOUR_BIAS:
        stage1_param_groups.append(
            {
                "params": [p for data in group_data.values() for p in data["bias"].hour_bias.parameters()],
                "lr": HOUR_BIAS_LR,
                "eps": BIAS_EPS,
                "weight_decay": LAMBDA["hour"],
            }
        )
    if USE_TRAIN_ONLY_YEAR_BIAS:
        stage1_param_groups.append(
            {
                "params": [p for data in group_data.values() for p in data["bias"].year_bias.parameters()],
                "lr": YEAR_BIAS_LR,
                "eps": BIAS_EPS,
                "weight_decay": LAMBDA["year"],
            }
        )
    if USE_SCADA_WD_CORRECTION:
        stage1_param_groups.append(
            {
                "params": [
                    p
                    for data in group_data.values()
                    if data["bias"].direction is not None
                    for p in data["bias"].direction.parameters()
                ],
                "lr": WD_BIAS_LR,
                "eps": BIAS_EPS,
                "weight_decay": LAMBDA.get("wd", 0.01),
            }
        )
    if len(stage1_param_groups) == 1:
        opt1 = torch.optim.Adam(model.parameters(), lr=LR)
    else:
        opt1 = torch.optim.AdamW(stage1_param_groups)
        print(
            f"[{physical_manufacturer}] stage1 auxiliary parameters enabled: "
            f"hour={'on' if USE_TRAIN_ONLY_HOUR_BIAS else 'off'} "
            f"year={'on' if USE_TRAIN_ONLY_YEAR_BIAS else 'off'} "
            f"wd={'on' if USE_SCADA_WD_CORRECTION else 'off'}; "
            "stage2 will train calendar bias only"
        )
    best_stage1_score = -np.inf
    best_stage1_epoch = -1
    best_stage1_state = None
    best_stage1_bias_states = None
    stage1_bad_checks = 0
    for epoch in range(STAGE1_EPOCHS):
        model.train()
        opt1.zero_grad()
        l_phys, _ = physics_losses(model, v_pool, c_max, turbine_capacity_w, LAMBDA)
        l_data_sum = 0.0
        for group, data in group_data.items():
            pred = group_prediction(
                model,
                data["train"],
                GROUP_N_TURBINES[group],
                bias=data["bias"] if USE_SCADA_WD_CORRECTION else None,
                use_calendar_bias=False,
                use_wind_distribution=True,
            )
            if USE_TRAIN_ONLY_HOUR_BIAS:
                pred = pred + data["bias"].train_only(data["row_idx"])
            if USE_TRAIN_ONLY_YEAR_BIAS:
                pred = pred + data["bias"].train_year(data["year_idx"])
            y = torch.tensor(data["train"]["y"].to_numpy(), dtype=torch.float32, device=DEVICE)
            l_data, _, _ = data_loss(y, pred, data["capacity"], gamma=GAMMA)
            l_data_sum = l_data_sum + l_data
        loss = l_phys + l_data_sum
        loss.backward()
        opt1.step()

        hour_l1 = LAMBDA.get("hour_l1", 0.0)
        hour_prox_start_epoch = int(LAMBDA.get("hour_prox_start_epoch", 0))
        hour_shrink = HOUR_BIAS_LR * hour_l1 if USE_TRAIN_ONLY_HOUR_BIAS and epoch >= hour_prox_start_epoch else 0.0
        if hour_shrink > 0:
            for data in group_data.values():
                soft_threshold_train_only_hour_bias(data["bias"], hour_shrink)

        should_eval = use_stage1_early_stop and (epoch % stage1_eval_interval == 0 or epoch == STAGE1_EPOCHS - 1)
        if should_eval:
            valid_score, valid_rows = _score_valid_groups(model, valid_data, use_bias=False)
            improved = valid_score is not None and valid_score > best_stage1_score + early_stop_min_delta
            if improved:
                best_stage1_score = valid_score
                best_stage1_epoch = epoch
                best_stage1_state = _clone_state_dict(model)
                best_stage1_bias_states = _clone_bias_states(group_data)
                stage1_bad_checks = 0
            else:
                stage1_bad_checks += 1
            print(
                f"[{physical_manufacturer}] stage1 epoch {epoch}: loss={loss.item():.4f} "
                f"valid_score={valid_score:.6f} best={best_stage1_score:.6f}@{best_stage1_epoch} "
                f"bad_checks={stage1_bad_checks}/{stage1_patience}"
            )
            if stage1_bad_checks >= stage1_patience:
                print(f"[{physical_manufacturer}] stage1 early stop at epoch {epoch}")
                break
        elif epoch % 250 == 0 or epoch == STAGE1_EPOCHS - 1:
            hour_stats = {}
            if USE_TRAIN_ONLY_HOUR_BIAS:
                stats = [hour_bias_abs_summary(data["bias"]) for data in group_data.values()]
                if stats and stats[0]:
                    hour_stats = {
                        "mean": sum(item["mean"] for item in stats) / len(stats),
                        "p99": max(item["p99"] for item in stats),
                        "max": max(item["max"] for item in stats),
                        "gt_001": sum(item["gt_001"] for item in stats),
                        "gt_005": sum(item["gt_005"] for item in stats),
                    }
            hour_msg = (
                f" hour_prox_l1={hour_l1:g} shrink={hour_shrink:.6f}"
                + (
                    f" hour_abs_mean={hour_stats['mean']:.6f}"
                    f" hour_abs_p99={hour_stats['p99']:.6f}"
                    f" hour_abs_max={hour_stats['max']:.6f}"
                    f" gt001={hour_stats['gt_001']} gt005={hour_stats['gt_005']}"
                    if hour_stats
                    else ""
                )
            )
            print(f"[{physical_manufacturer}] stage1 epoch {epoch}: loss={loss.item():.4f}{hour_msg}")

    if use_stage1_early_stop and best_stage1_state is not None:
        model.load_state_dict(best_stage1_state)
        if best_stage1_bias_states is not None:
            _load_bias_states(group_data, best_stage1_bias_states)
        print(
            f"[{physical_manufacturer}] restored stage1 best checkpoint: "
            f"epoch={best_stage1_epoch} valid_score={best_stage1_score:.6f}"
        )

    for param in model.parameters():
        param.requires_grad_(False)

    with torch.no_grad():
        for group, data in group_data.items():
            data["base_pred"] = group_prediction(
                model,
                data["train"],
                GROUP_N_TURBINES[group],
                bias=data["bias"] if USE_SCADA_WD_CORRECTION else None,
                use_calendar_bias=False,
                use_wind_distribution=True,
            ).detach()
            data["y"] = torch.tensor(data["train"]["y"].to_numpy(), dtype=torch.float32, device=DEVICE)
            data["hod_idx"] = torch.tensor(data["train"]["hod"].to_numpy(), dtype=torch.long, device=DEVICE)
            data["moy_idx"] = (
                torch.tensor(data["train"]["moy"].to_numpy(), dtype=torch.long, device=DEVICE)
                if USE_MOY_BIAS
                else None
            )
            if USE_DOW_BIAS and "dow" not in data["train"].columns:
                raise ValueError("USE_DOW_BIAS=True requires a dow column")
            data["dow_idx"] = (
                torch.tensor(data["train"]["dow"].to_numpy(), dtype=torch.long, device=DEVICE)
                if USE_DOW_BIAS
                else None
            )
        if valid_data:
            _cache_eval_tensors(valid_data, model=model)

    param_groups = [
        {
            "params": [p for data in group_data.values() for p in data["bias"].hod_bias.parameters()],
            "lr": BIAS_LR,
            "eps": BIAS_EPS,
            "weight_decay": LAMBDA["hod"],
        }
    ]
    if USE_MOY_BIAS:
        param_groups.append(
            {
                "params": [p for data in group_data.values() for p in data["bias"].moy_bias.parameters()],
                "lr": MOY_BIAS_LR,
                "eps": BIAS_EPS,
                "weight_decay": LAMBDA["moy"],
            }
        )
    if USE_DOW_BIAS:
        param_groups.append(
            {
                "params": [p for data in group_data.values() for p in data["bias"].dow_bias.parameters()],
                "lr": DOW_BIAS_LR,
                "eps": BIAS_EPS,
                "weight_decay": LAMBDA.get("dow", LAMBDA["hod"]),
            }
        )
    opt2 = torch.optim.AdamW(param_groups)
    best_stage2_score = -np.inf
    best_stage2_epoch = -1
    best_stage2_states = None
    stage2_bad_checks = 0
    if use_stage2_early_stop:
        best_stage2_score, _ = _score_valid_groups(model, valid_data, use_bias=True, use_cached_base=True)
        best_stage2_epoch = -1
        best_stage2_states = _clone_bias_states(group_data)
        print(
            f"[{physical_manufacturer}] stage2 initial valid_score={best_stage2_score:.6f} "
            f"best={best_stage2_score:.6f}@-1"
        )
    for epoch in range(STAGE2_EPOCHS):
        opt2.zero_grad()
        l_data_sum = 0.0
        for group, data in group_data.items():
            pred = data["base_pred"] + data["bias"].calendar(data["hod_idx"], data["moy_idx"], data["dow_idx"])
            l_data, _, _ = data_loss(data["y"], pred, data["capacity"], gamma=GAMMA)
            l_data_sum = l_data_sum + l_data
        l_data_sum.backward()
        opt2.step()
        if epoch % 500 == 0 or epoch == STAGE2_EPOCHS - 1:
            bias_diag = sum(bias_l2(data["bias"])["hod"].item() for data in group_data.values())
            print(
                f"[{physical_manufacturer}] stage2 epoch {epoch}: "
                f"data={l_data_sum.item():.4f} hod_l2={bias_diag:.6f}"
            )
        should_eval = use_stage2_early_stop and (epoch % stage2_eval_interval == 0 or epoch == STAGE2_EPOCHS - 1)
        if should_eval:
            valid_score, _ = _score_valid_groups(model, valid_data, use_bias=True, use_cached_base=True)
            improved = valid_score is not None and valid_score > best_stage2_score + early_stop_min_delta
            if improved:
                best_stage2_score = valid_score
                best_stage2_epoch = epoch
                best_stage2_states = _clone_bias_states(group_data)
                stage2_bad_checks = 0
            else:
                stage2_bad_checks += 1
            print(
                f"[{physical_manufacturer}] stage2 epoch {epoch}: "
                f"valid_score={valid_score:.6f} best={best_stage2_score:.6f}@{best_stage2_epoch} "
                f"bad_checks={stage2_bad_checks}/{stage2_patience}"
            )
            if stage2_bad_checks >= stage2_patience:
                print(f"[{physical_manufacturer}] stage2 early stop at epoch {epoch}")
                break

    if use_stage2_early_stop and best_stage2_states is not None:
        _load_bias_states(group_data, best_stage2_states)
        print(
            f"[{physical_manufacturer}] restored stage2 best bias: "
            f"epoch={best_stage2_epoch} valid_score={best_stage2_score:.6f}"
        )

    return model, {group: data["bias"] for group in group_data}


def predict_pinn(model, bias, group, weather_test):
    with torch.no_grad():
        pred = group_prediction(model, weather_test, GROUP_N_TURBINES[group], bias=bias, use_wind_distribution=True)
        return torch.clamp(pred, min=0.0, max=GROUP_CAPACITY_KWH[group]).cpu().numpy()

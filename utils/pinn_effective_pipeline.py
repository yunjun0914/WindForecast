import numpy as np
import pandas as pd
import torch
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
    STAGE1_EPOCHS,
    STAGE2_EPOCHS,
    USE_MOY_BIAS,
    USE_TRAIN_ONLY_HOUR_BIAS,
    USE_TRAIN_ONLY_YEAR_BIAS,
    YEAR_BIAS_LR,
    group_prediction,
    physics_losses,
)
from utils.metrics import GROUP_CAPACITY_KWH
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
    ]
    for col in blend_cols:
        if col in weather_a.columns and col in weather_b.columns:
            out[col] = weight_a * weather_a[col].to_numpy() + (1 - weight_a) * weather_b[col].to_numpy()
    out["teacher_recipe"] = name
    return out


def train_full_pinn(physical_manufacturer, group_weather_train, labels):
    seed_everything(42)
    groups = list(group_weather_train)
    c_max = C_MAX_BY_MANUFACTURER[physical_manufacturer]
    area = MANUFACTURER_AREA[physical_manufacturer]
    turbine_capacity_w = SINGLE_TURBINE_CAPACITY_W[physical_manufacturer]
    model = PowerCurvePINN(c_max, area).to(DEVICE)
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
        ).to(DEVICE)
        group_data[group] = {
            "train": train_df,
            "bias": bias,
            "row_idx": torch.arange(len(train_df), dtype=torch.long, device=DEVICE),
            "year_idx": torch.tensor(train_df["year_idx"].to_numpy(), dtype=torch.long, device=DEVICE),
            "capacity": GROUP_CAPACITY_KWH[group],
        }

    opt1 = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(STAGE1_EPOCHS):
        opt1.zero_grad()
        l_phys, _ = physics_losses(model, v_pool, c_max, turbine_capacity_w, LAMBDA)
        l_data_sum = 0.0
        for group, data in group_data.items():
            pred = group_prediction(model, data["train"], GROUP_N_TURBINES[group], use_wind_distribution=True)
            y = torch.tensor(data["train"]["y"].to_numpy(), dtype=torch.float32, device=DEVICE)
            l_data, _, _ = data_loss(y, pred, data["capacity"], gamma=GAMMA)
            l_data_sum = l_data_sum + l_data
        loss = l_phys + l_data_sum
        loss.backward()
        opt1.step()
        if epoch % 250 == 0 or epoch == STAGE1_EPOCHS - 1:
            print(f"[{physical_manufacturer}] stage1 epoch {epoch}: loss={loss.item():.4f}")

    for param in model.parameters():
        param.requires_grad_(False)

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
    if USE_TRAIN_ONLY_HOUR_BIAS:
        param_groups.append(
            {
                "params": [p for data in group_data.values() for p in data["bias"].hour_bias.parameters()],
                "lr": HOUR_BIAS_LR,
                "eps": BIAS_EPS,
                "weight_decay": LAMBDA["hour"],
            }
        )
    if USE_TRAIN_ONLY_YEAR_BIAS:
        param_groups.append(
            {
                "params": [p for data in group_data.values() for p in data["bias"].year_bias.parameters()],
                "lr": YEAR_BIAS_LR,
                "eps": BIAS_EPS,
                "weight_decay": LAMBDA["year"],
            }
        )

    opt2 = torch.optim.AdamW(param_groups)
    for epoch in range(STAGE2_EPOCHS):
        opt2.zero_grad()
        l_data_sum = 0.0
        for group, data in group_data.items():
            pred = group_prediction(
                model,
                data["train"],
                GROUP_N_TURBINES[group],
                bias=data["bias"],
                row_idx=data["row_idx"],
                year_idx=data["year_idx"],
                use_wind_distribution=True,
            )
            y = torch.tensor(data["train"]["y"].to_numpy(), dtype=torch.float32, device=DEVICE)
            l_data, _, _ = data_loss(y, pred, data["capacity"], gamma=GAMMA)
            l_data_sum = l_data_sum + l_data
        l_data_sum.backward()
        opt2.step()

        hour_l1 = LAMBDA.get("hour_l1", 0.0)
        hour_prox_start_epoch = int(LAMBDA.get("hour_prox_start_epoch", 0))
        hour_shrink = HOUR_BIAS_LR * hour_l1 if USE_TRAIN_ONLY_HOUR_BIAS and epoch >= hour_prox_start_epoch else 0.0
        if hour_shrink > 0:
            for data in group_data.values():
                soft_threshold_train_only_hour_bias(data["bias"], hour_shrink)
        if epoch % 500 == 0 or epoch == STAGE2_EPOCHS - 1:
            bias_diag = sum(bias_l2(data["bias"])["hod"].item() for data in group_data.values())
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
            print(
                f"[{physical_manufacturer}] stage2 epoch {epoch}: "
                f"data={l_data_sum.item():.4f}{hour_msg} hod_l2={bias_diag:.6f}"
            )

    return model, {group: data["bias"] for group in group_data}


def predict_pinn(model, bias, group, weather_test):
    with torch.no_grad():
        pred = group_prediction(model, weather_test, GROUP_N_TURBINES[group], bias=bias, use_wind_distribution=True)
        return torch.clamp(pred, min=0.0, max=GROUP_CAPACITY_KWH[group]).cpu().numpy()


import numpy as np
import pandas as pd
import torch

from models.pinn import PowerCurvePINN, TurbineGroupBias
from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr
from utils.pinn_data import (
    C_MAX_BY_MANUFACTURER,
    CUT_OUT_SPEED,
    GROUP_MANUFACTURER,
    GROUP_N_TURBINES,
    apply_scada_wind_teacher_oob,
    RATED_SPEED,
    apply_wind_speed_correction,
    build_group_pinn_dataset,
    build_pinn_weather,
    fit_wind_speed_correction,
)
from utils.pinn_losses import (
    bias_l2,
    boundary_condition_loss,
    data_loss,
    betz_loss,
    flatness_loss,
    hour_bias_abs_summary,
    smoothness_loss,
    soft_threshold_train_only_hour_bias,
)
from utils.pinn_physics import MANUFACTURER_AREA, SINGLE_TURBINE_CAPACITY_W

VAL_START = "2024-01-01 01:00:00"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

STAGE1_EPOCHS = 500
STAGE2_EPOCHS = 2000  # bias needs this many epochs (at BIAS_LR=1e-2) to reach a size that matters
LR = 1e-3
N_COLLOCATION = 2000
USE_MOY_BIAS = False
USE_DOW_BIAS = False
USE_TRAIN_ONLY_HOUR_BIAS = False
USE_TRAIN_ONLY_YEAR_BIAS = False
USE_WIND_DISTRIBUTION = {
    "vestas": True,
    "unison": True,
}
USE_SCADA_WIND_TEACHER = True
HONEST_SCADA_TEACHER_HOLDOUT = True
USE_SCADA_WD_CORRECTION = False
SCADA_WD_AMPLITUDE = 0.02

# best-known lambdas found via sweep_pinn.py (results/pinn_sweep_results.csv, trial 11)
# -- hour/day/month dropped (see docs/pinn_plan.md 11.4: measured to actively hurt
# out-of-sample generalization), hod added (the one bias term that generalized)
LAMBDA = {
    "betz": 2.026253819683706,
    "bc": 0.004598334508698543,
    "flat": 0.2930203005161716,
    "smooth": 0.004331969897949324,
    "hod": 0.001,
    "moy": 0.001,
    "hour": 0.01,
    "hour_l1": 0.0,
    "hour_prox_start_epoch": 0,
    "year": 0.01,
    "wd": 0.01,
}
GAMMA = 0.039709016191988696

# hod_bias is parameterized in capacity-normalized units inside TurbineGroupBias, so
# a parameter value of 0.05 means a 5%-of-capacity correction (about 1,000 kWh). This
# keeps the optimizer step size on the same scale as the official normalized error.
BIAS_LR = 1e-3
MOY_BIAS_LR = 1e-3
DOW_BIAS_LR = 1e-3
HOUR_BIAS_LR = 1e-3
YEAR_BIAS_LR = 1e-3
WD_BIAS_LR = 1e-3
BIAS_EPS = 1e-12


def time_split(df):
    is_val = df["forecast_kst_dtm"] >= VAL_START
    return df[~is_val].reset_index(drop=True), df[is_val].reset_index(drop=True)


def to_device(df, cols):
    return {c: torch.tensor(df[c].to_numpy(), dtype=torch.float32, device=DEVICE) for c in cols}


def sample_collocation(v_pool, n, device):
    v_c = torch.tensor(np.random.choice(v_pool, n), dtype=torch.float32, device=device)
    doy_c = torch.randint(1, 366, (n,), dtype=torch.float32, device=device)
    moy_c = torch.randint(1, 13, (n,), dtype=torch.float32, device=device)
    rho_c = torch.full((n,), 1.22, device=device)
    return v_c, doy_c, moy_c, rho_c


def physics_losses(model, v_pool, c_max, turbine_capacity_w, lam):
    v_c, doy_c, moy_c, rho_c = sample_collocation(v_pool, N_COLLOCATION, DEVICE)
    l_betz = betz_loss(model, v_c, doy_c, moy_c, c_max)
    l_bc = boundary_condition_loss(model, doy_c, moy_c)
    v_flat = torch.rand(N_COLLOCATION, device=DEVICE) * (CUT_OUT_SPEED - RATED_SPEED) + RATED_SPEED
    l_flat = flatness_loss(model, v_flat, rho_c, doy_c, moy_c, turbine_capacity_w)
    l_smooth = smoothness_loss(model, v_c)
    total = lam["betz"] * l_betz + lam["bc"] * l_bc + lam["flat"] * l_flat + lam["smooth"] * l_smooth
    breakdown = {"betz": l_betz.detach(), "bc": l_bc.detach(), "flat": l_flat.detach(), "smooth": l_smooth.detach()}
    return total, breakdown


def group_prediction(
    model,
    df,
    n_turbines,
    bias=None,
    row_idx=None,
    year_idx=None,
    use_wind_distribution=True,
    use_calendar_bias=True,
    use_direction_correction=True,
):
    cols = ["v", "rho", "doy", "moy"] + (["v_std"] if use_wind_distribution and "v_std" in df.columns else [])
    t = to_device(df, cols)
    c_eff_add = None
    if (
        USE_SCADA_WD_CORRECTION
        and use_direction_correction
        and bias is not None
        and getattr(bias, "direction", None) is not None
    ):
        if "scada_wd_sin" not in df.columns or "scada_wd_cos" not in df.columns:
            raise ValueError("USE_SCADA_WD_CORRECTION=True requires scada_wd_sin/scada_wd_cos columns")
        wd_sin = torch.tensor(df["scada_wd_sin"].to_numpy(), dtype=torch.float32, device=DEVICE)
        wd_cos = torch.tensor(df["scada_wd_cos"].to_numpy(), dtype=torch.float32, device=DEVICE)
        c_eff_add = bias.direction_c_eff(wd_sin, wd_cos)
    p_smoothed = model(
        t["v"], t["rho"], t["doy"], t["moy"], v_std=t.get("v_std"), c_eff_add=c_eff_add
    ) * n_turbines / 1000.0  # W -> kW(h)
    if bias is not None:
        if use_calendar_bias:
            hod_idx = torch.tensor(df["hod"].to_numpy(), dtype=torch.long, device=DEVICE)
            moy_idx = torch.tensor(df["moy"].to_numpy(), dtype=torch.long, device=DEVICE) if USE_MOY_BIAS else None
            if USE_DOW_BIAS and "dow" not in df.columns:
                raise ValueError("USE_DOW_BIAS=True requires a dow column")
            dow_idx = torch.tensor(df["dow"].to_numpy(), dtype=torch.long, device=DEVICE) if USE_DOW_BIAS else None
            p_smoothed = p_smoothed + bias.calendar(hod_idx, moy_idx, dow_idx)
        if row_idx is not None and getattr(bias, "hour_bias", None) is not None:
            p_smoothed = p_smoothed + bias.train_only(row_idx)
        if year_idx is not None and getattr(bias, "year_bias", None) is not None:
            p_smoothed = p_smoothed + bias.train_year(year_idx)
    return p_smoothed


def evaluate(model, group_data, use_bias):
    rows = []
    for group, gd in group_data.items():
        with torch.no_grad():
            bias = gd["bias"] if (use_bias or USE_SCADA_WD_CORRECTION) else None
            pred = group_prediction(
                model,
                gd["val"],
                gd["n_turbines"],
                bias=bias,
                use_calendar_bias=use_bias,
                use_wind_distribution=gd["use_wind_distribution"],
            )
            pred = torch.clamp(pred, min=0.0, max=gd["capacity"])
        y = torch.tensor(gd["val"]["y"].to_numpy(), dtype=torch.float32, device=DEVICE)
        nmae, ficr = group_nmae_ficr(y.cpu().numpy(), pred.cpu().numpy(), gd["capacity"])
        score = 0.5 * (1 - nmae) + 0.5 * ficr
        rows.append({"group": group, "nmae": nmae, "ficr": ficr, "score": score})
    return pd.DataFrame(rows)


def train_manufacturer(
    manufacturer,
    weather_by_group,
    labels,
    lam=None,
    gamma=GAMMA,
    stage1_epochs=STAGE1_EPOCHS,
    stage2_epochs=STAGE2_EPOCHS,
    verbose=True,
    save=True,
    model_cls=PowerCurvePINN,
    model_kwargs=None,
):
    lam = LAMBDA if lam is None else lam
    model_kwargs = {} if model_kwargs is None else model_kwargs
    groups = [g for g, m in GROUP_MANUFACTURER.items() if m == manufacturer]
    area = MANUFACTURER_AREA[manufacturer]
    c_max = C_MAX_BY_MANUFACTURER[manufacturer]
    turbine_capacity_w = SINGLE_TURBINE_CAPACITY_W[manufacturer]

    model = model_cls(c_max, area, **model_kwargs).to(DEVICE)
    v_pool = np.concatenate([weather_by_group[group]["v"].to_numpy() for group in groups])

    group_data = {}
    for group in groups:
        weather_train = weather_by_group[group]
        ds = build_group_pinn_dataset(weather_train, labels, group)
        train_df, val_df = time_split(ds)
        train_df = train_df.copy()
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
            "val": val_df,
            "bias": bias,
            "train_row_idx": torch.arange(len(train_df), dtype=torch.long, device=DEVICE),
            "train_year_idx": torch.tensor(train_df["year_idx"].to_numpy(), dtype=torch.long, device=DEVICE),
            "use_wind_distribution": USE_WIND_DISTRIBUTION[manufacturer],
            "n_turbines": GROUP_N_TURBINES[group],
            "capacity": GROUP_CAPACITY_KWH[group],
        }

    # ---- stage 1: physics backbone + optional train-only anomaly bias ----
    stage1_param_groups = [{"params": list(model.parameters()), "lr": LR, "weight_decay": 0.0}]
    if USE_TRAIN_ONLY_HOUR_BIAS:
        stage1_param_groups.append(
            {
                "params": [p for gd in group_data.values() for p in gd["bias"].hour_bias.parameters()],
                "lr": HOUR_BIAS_LR,
                "eps": BIAS_EPS,
                "weight_decay": lam["hour"],
            }
        )
    if USE_TRAIN_ONLY_YEAR_BIAS:
        stage1_param_groups.append(
            {
                "params": [p for gd in group_data.values() for p in gd["bias"].year_bias.parameters()],
                "lr": YEAR_BIAS_LR,
                "eps": BIAS_EPS,
                "weight_decay": lam["year"],
            }
        )
    if USE_SCADA_WD_CORRECTION:
        stage1_param_groups.append(
            {
                "params": [
                    p
                    for gd in group_data.values()
                    if gd["bias"].direction is not None
                    for p in gd["bias"].direction.parameters()
                ],
                "lr": WD_BIAS_LR,
                "eps": BIAS_EPS,
                "weight_decay": lam.get("wd", 0.01),
            }
        )
    opt1 = torch.optim.Adam(model.parameters(), lr=LR) if len(stage1_param_groups) == 1 else torch.optim.AdamW(stage1_param_groups)
    for epoch in range(stage1_epochs):
        opt1.zero_grad()
        l_phys, phys_breakdown = physics_losses(model, v_pool, c_max, turbine_capacity_w, lam)
        l_data_sum = 0.0
        for group, gd in group_data.items():
            pred = group_prediction(
                model,
                gd["train"],
                gd["n_turbines"],
                bias=gd["bias"] if USE_SCADA_WD_CORRECTION else None,
                use_calendar_bias=False,
                use_wind_distribution=gd["use_wind_distribution"],
            )
            if USE_TRAIN_ONLY_HOUR_BIAS:
                pred = pred + gd["bias"].train_only(gd["train_row_idx"])
            if USE_TRAIN_ONLY_YEAR_BIAS:
                pred = pred + gd["bias"].train_year(gd["train_year_idx"])
            y = torch.tensor(gd["train"]["y"].to_numpy(), dtype=torch.float32, device=DEVICE)
            l_data, _, _ = data_loss(y, pred, gd["capacity"], gamma=gamma)
            l_data_sum = l_data_sum + l_data
        loss = l_phys + l_data_sum
        loss.backward()
        opt1.step()
        hour_l1 = lam.get("hour_l1", 0.0)
        hour_prox_start_epoch = int(lam.get("hour_prox_start_epoch", 0))
        hour_shrink = HOUR_BIAS_LR * hour_l1 if USE_TRAIN_ONLY_HOUR_BIAS and epoch >= hour_prox_start_epoch else 0.0
        if hour_shrink > 0:
            for gd in group_data.values():
                soft_threshold_train_only_hour_bias(gd["bias"], hour_shrink)
        if verbose and (epoch % 100 == 0 or epoch == stage1_epochs - 1):
            hour_stats = {}
            if USE_TRAIN_ONLY_HOUR_BIAS:
                stats = [hour_bias_abs_summary(gd["bias"]) for gd in group_data.values()]
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
                f"[{manufacturer}] stage1 epoch {epoch}: loss={loss.item():.4f} "
                f"(data={l_data_sum.item():.4f}, physics={l_phys.item():.4f} "
                f"[betz={phys_breakdown['betz'].item():.4f} bc={phys_breakdown['bc'].item():.4f} "
                f"flat={phys_breakdown['flat'].item():.4f} smooth={phys_breakdown['smooth'].item():.4f}]) "
                f"tau={model.response.tau.item():.3f}{hour_msg}"
            )

    stage1_scores = evaluate(model, group_data, use_bias=False)
    if verbose:
        print(f"[{manufacturer}] stage1 (physics only) validation:\n{stage1_scores}")
    if save:
        torch.save(model.state_dict(), f"results/pinn_{manufacturer}_stage1.pt")

    # ---- stage 2: freeze the physics backbone, train ONLY the bias embeddings ----
    # letting model.parameters() keep moving here would let the (shared, test-time-used)
    # physics backbone soak up train-period-only noise that bias is supposed to isolate
    # instead -- corrupting the physics fit AND starving bias of the residual signal it
    # needs. Freezing model.parameters() makes stage2's effect on validation
    # attributable to bias alone, with no confound.
    for p in model.parameters():
        p.requires_grad_(False)

    param_groups = [
        {
            "params": [p for gd in group_data.values() for p in gd["bias"].hod_bias.parameters()],
            "lr": BIAS_LR,
            "eps": BIAS_EPS,
            "weight_decay": lam["hod"],
        }
    ]
    if USE_MOY_BIAS:
        param_groups.append(
            {
                "params": [p for gd in group_data.values() for p in gd["bias"].moy_bias.parameters()],
                "lr": MOY_BIAS_LR,
                "eps": BIAS_EPS,
                "weight_decay": lam["moy"],
            }
        )
    if USE_DOW_BIAS:
        param_groups.append(
            {
                "params": [p for gd in group_data.values() for p in gd["bias"].dow_bias.parameters()],
                "lr": DOW_BIAS_LR,
                "eps": BIAS_EPS,
                "weight_decay": lam.get("dow", lam["hod"]),
            }
        )
    opt2 = torch.optim.AdamW(param_groups)
    for epoch in range(stage2_epochs):
        opt2.zero_grad()
        l_data_sum = 0.0
        bias_breakdown = {"hod": 0.0}
        if USE_DOW_BIAS:
            bias_breakdown["dow"] = 0.0
        if USE_MOY_BIAS:
            bias_breakdown["moy"] = 0.0
        for group, gd in group_data.items():
            pred = group_prediction(
                model,
                gd["train"],
                gd["n_turbines"],
                bias=gd["bias"],
                use_wind_distribution=gd["use_wind_distribution"],
            )
            y = torch.tensor(gd["train"]["y"].to_numpy(), dtype=torch.float32, device=DEVICE)
            l_data, _, _ = data_loss(y, pred, gd["capacity"], gamma=gamma)
            l_data_sum = l_data_sum + l_data
            with torch.no_grad():
                l_bias = bias_l2(gd["bias"])
                for k in bias_breakdown:
                    bias_breakdown[k] += l_bias[k].item()
        l_data_sum.backward()
        opt2.step()
        if verbose and (epoch % 100 == 0 or epoch == stage2_epochs - 1):
            print(
                f"[{manufacturer}] stage2 epoch {epoch}: data={l_data_sum.item():.4f}"
                f"bias_l2(diag) [{', '.join(f'{k}={v:.6f}' for k, v in bias_breakdown.items())}]"
            )

    stage2_scores = evaluate(model, group_data, use_bias=True)
    if verbose:
        print(f"[{manufacturer}] stage2 (+bias) validation:\n{stage2_scores}")
    if save:
        torch.save(model.state_dict(), f"results/pinn_{manufacturer}_stage2.pt")
        for group, gd in group_data.items():
            torch.save(gd["bias"].state_dict(), f"results/pinn_{group}_bias.pt")

    return model, group_data, stage1_scores, stage2_scores


def load_training_data():
    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    weather_train_raw = build_pinn_weather(ldaps_train, gfs_train)

    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada_by_manufacturer = {"vestas": scada_vestas, "unison": scada_unison}
    scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }

    group_weather = {}
    for group, scada in scada_by_group.items():
        if USE_SCADA_WIND_TEACHER:
            fit_before = VAL_START if HONEST_SCADA_TEACHER_HOLDOUT else None
            group_weather[group] = apply_scada_wind_teacher_oob(
                weather_train_raw, scada, group, fit_before=fit_before
            )
        else:
            manufacturer = GROUP_MANUFACTURER[group]
            fit_before = VAL_START if HONEST_SCADA_TEACHER_HOLDOUT else None
            correction = fit_wind_speed_correction(
                weather_train_raw, scada_by_manufacturer[manufacturer], manufacturer, fit_before=fit_before
            )
            group_weather[group] = apply_wind_speed_correction(weather_train_raw, correction)

    weather_by_manufacturer = {}
    for manufacturer in scada_by_manufacturer:
        weather_by_manufacturer[manufacturer] = {
            group: weather for group, weather in group_weather.items() if GROUP_MANUFACTURER[group] == manufacturer
        }
    return weather_by_manufacturer, labels


def run_all_manufacturers(corrected_weather, labels, lam=None, gamma=GAMMA, stage1_epochs=STAGE1_EPOCHS,
                           stage2_epochs=STAGE2_EPOCHS, verbose=True, save=True,
                           model_cls=PowerCurvePINN, model_kwargs=None):
    all_scores = []
    for manufacturer, weather_by_group in corrected_weather.items():
        _, _, stage1, stage2 = train_manufacturer(
            manufacturer, weather_by_group, labels, lam=lam, gamma=gamma,
            stage1_epochs=stage1_epochs, stage2_epochs=stage2_epochs, verbose=verbose, save=save,
            model_cls=model_cls, model_kwargs=model_kwargs,
        )
        stage1["stage"] = "physics_only"
        stage2["stage"] = "with_bias"
        all_scores.append(stage1)
        all_scores.append(stage2)
    return pd.concat(all_scores, ignore_index=True)


def main():
    corrected_weather, labels = load_training_data()
    results = run_all_manufacturers(corrected_weather, labels)
    print("\n=== Summary ===")
    print(results)
    results.to_csv("results/pinn_time_holdout_scores.csv", index=False, encoding="utf-8-sig")
    return results


if __name__ == "__main__":
    main()

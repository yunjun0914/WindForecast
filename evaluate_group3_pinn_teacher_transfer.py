import numpy as np
import pandas as pd
import torch

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
    time_split,
)
from utils.metrics import GROUP_CAPACITY_KWH, group_nmae_ficr
from utils.pinn_data import (
    C_MAX_BY_MANUFACTURER,
    GROUP_N_TURBINES,
    apply_scada_wind_teacher,
    build_group_pinn_dataset,
    build_pinn_weather,
    fit_scada_wind_teacher,
)
from utils.pinn_losses import bias_l2, data_loss
from utils.pinn_physics import MANUFACTURER_AREA, SINGLE_TURBINE_CAPACITY_W

GROUP3 = "kpx_group_3"
GROUP2 = "kpx_group_2"
GROUP1 = "kpx_group_1"
VAL_START = "2024-01-01 01:00:00"
RESULTS_PATH = "results/group3_pinn_teacher_transfer_scores.csv"
BLEND_RESULTS_PATH = "results/group3_pinn_teacher_proxy_blend_scores.csv"


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def blend_weather(name, weather_a, weather_b, weight_a):
    out = weather_a.copy()
    blend_cols = ["v", "v_std", "scada_ws_mean", "scada_ws_std", "scada_ws_p10", "scada_ws_p50", "scada_ws_p90"]
    for col in blend_cols:
        if col in weather_a.columns and col in weather_b.columns:
            out[col] = weight_a * weather_a[col].to_numpy() + (1 - weight_a) * weather_b[col].to_numpy()
    out["teacher_recipe"] = name
    return out


def evaluate_predictions(y, pred, capacity):
    pred = np.clip(pred, 0, capacity)
    nmae, ficr = group_nmae_ficr(y, pred, capacity)
    return 0.5 * (1 - nmae) + 0.5 * ficr, nmae, ficr


def predict_model(model, bias, val_df):
    capacity = GROUP_CAPACITY_KWH[GROUP3]
    with torch.no_grad():
        pred = group_prediction(
            model,
            val_df,
            GROUP_N_TURBINES[GROUP3],
            bias=bias,
            use_wind_distribution=True,
        )
        pred = torch.clamp(pred, min=0.0, max=capacity).cpu().numpy()
    return pred


def evaluate_model(model, bias, val_df):
    capacity = GROUP_CAPACITY_KWH[GROUP3]
    pred = predict_model(model, bias, val_df)
    y = val_df["y"].to_numpy()
    score, nmae, ficr = evaluate_predictions(y, pred, capacity)
    return score, nmae, ficr


def train_single_group(physical_manufacturer, weather, labels, seed=42, verbose=False):
    seed_everything(seed)
    lam = LAMBDA
    capacity = GROUP_CAPACITY_KWH[GROUP3]
    area = MANUFACTURER_AREA[physical_manufacturer]
    c_max = C_MAX_BY_MANUFACTURER[physical_manufacturer]
    turbine_capacity_w = SINGLE_TURBINE_CAPACITY_W[physical_manufacturer]

    model = PowerCurvePINN(c_max, area).to(DEVICE)
    v_pool = weather["v"].to_numpy()

    ds = build_group_pinn_dataset(weather, labels, GROUP3)
    train_df, val_df = time_split(ds)
    train_df = train_df.copy()
    train_years = sorted(train_df["forecast_kst_dtm"].dt.year.unique())
    year_to_idx = {year: idx for idx, year in enumerate(train_years)}
    train_df["year_idx"] = train_df["forecast_kst_dtm"].dt.year.map(year_to_idx)

    bias = TurbineGroupBias(
        capacity,
        n_train_rows=len(train_df) if USE_TRAIN_ONLY_HOUR_BIAS else None,
        n_train_years=len(train_years) if USE_TRAIN_ONLY_YEAR_BIAS else None,
    ).to(DEVICE)
    train_row_idx = torch.arange(len(train_df), dtype=torch.long, device=DEVICE)
    train_year_idx = torch.tensor(train_df["year_idx"].to_numpy(), dtype=torch.long, device=DEVICE)

    opt1 = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(STAGE1_EPOCHS):
        opt1.zero_grad()
        l_phys, _ = physics_losses(model, v_pool, c_max, turbine_capacity_w, lam)
        pred = group_prediction(model, train_df, GROUP_N_TURBINES[GROUP3], use_wind_distribution=True)
        y = torch.tensor(train_df["y"].to_numpy(), dtype=torch.float32, device=DEVICE)
        l_data, _, _ = data_loss(y, pred, capacity, gamma=GAMMA)
        loss = l_phys + l_data
        loss.backward()
        opt1.step()
        if verbose and (epoch % 250 == 0 or epoch == STAGE1_EPOCHS - 1):
            print(f"[{physical_manufacturer}] stage1 epoch {epoch}: loss={loss.item():.4f}")

    stage1_score, stage1_nmae, stage1_ficr = evaluate_model(model, None, val_df)

    for p in model.parameters():
        p.requires_grad_(False)

    param_groups = [
        {
            "params": list(bias.hod_bias.parameters()),
            "lr": BIAS_LR,
            "eps": BIAS_EPS,
            "weight_decay": lam["hod"],
        }
    ]
    if USE_MOY_BIAS:
        param_groups.append(
            {
                "params": list(bias.moy_bias.parameters()),
                "lr": MOY_BIAS_LR,
                "eps": BIAS_EPS,
                "weight_decay": lam["moy"],
            }
        )
    if USE_TRAIN_ONLY_HOUR_BIAS:
        param_groups.append(
            {
                "params": list(bias.hour_bias.parameters()),
                "lr": HOUR_BIAS_LR,
                "eps": BIAS_EPS,
                "weight_decay": lam["hour"],
            }
        )
    if USE_TRAIN_ONLY_YEAR_BIAS:
        param_groups.append(
            {
                "params": list(bias.year_bias.parameters()),
                "lr": YEAR_BIAS_LR,
                "eps": BIAS_EPS,
                "weight_decay": lam["year"],
            }
        )
    opt2 = torch.optim.AdamW(param_groups)
    for epoch in range(STAGE2_EPOCHS):
        opt2.zero_grad()
        pred = group_prediction(
            model,
            train_df,
            GROUP_N_TURBINES[GROUP3],
            bias=bias,
            row_idx=train_row_idx,
            year_idx=train_year_idx,
            use_wind_distribution=True,
        )
        y = torch.tensor(train_df["y"].to_numpy(), dtype=torch.float32, device=DEVICE)
        l_data, _, _ = data_loss(y, pred, capacity, gamma=GAMMA)
        l_data.backward()
        opt2.step()
        if verbose and (epoch % 500 == 0 or epoch == STAGE2_EPOCHS - 1):
            l_bias = bias_l2(bias)
            print(
                f"[{physical_manufacturer}] stage2 epoch {epoch}: "
                f"data={l_data.item():.4f} hod_l2={l_bias['hod'].item():.6f}"
            )

    stage2_pred = predict_model(model, bias, val_df)
    stage2_score, stage2_nmae, stage2_ficr = evaluate_predictions(
        val_df["y"].to_numpy(), stage2_pred, GROUP_CAPACITY_KWH[GROUP3]
    )
    return {
        "stage1_score": stage1_score,
        "stage1_nmae": stage1_nmae,
        "stage1_ficr": stage1_ficr,
        "stage2_score": stage2_score,
        "stage2_nmae": stage2_nmae,
        "stage2_ficr": stage2_ficr,
        "val_time": pd.to_datetime(val_df["forecast_kst_dtm"]).reset_index(drop=True),
        "y": val_df["y"].to_numpy(),
        "pred": stage2_pred,
    }


def build_teacher_weather_variants():
    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    weather_raw = build_pinn_weather(ldaps_train, gfs_train)

    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")

    unison_g3 = apply_scada_wind_teacher(
        weather_raw, fit_scada_wind_teacher(weather_raw, scada_unison, GROUP3, fit_before=VAL_START)
    )
    vestas_g2 = apply_scada_wind_teacher(
        weather_raw, fit_scada_wind_teacher(weather_raw, scada_vestas, GROUP2, fit_before=VAL_START)
    )
    vestas_g1 = apply_scada_wind_teacher(
        weather_raw, fit_scada_wind_teacher(weather_raw, scada_vestas, GROUP1, fit_before=VAL_START)
    )
    vestas_g12 = blend_weather("vestas_g12_avg_teacher", vestas_g1, vestas_g2, 0.5)

    variants = {
        "unison_g3_teacher": unison_g3,
        "vestas_g2_teacher": vestas_g2,
        "vestas_g1_teacher": vestas_g1,
        "vestas_g12_avg_teacher": vestas_g12,
        "mix_50_unison_g3_vestas_g2": blend_weather("mix_50_unison_g3_vestas_g2", unison_g3, vestas_g2, 0.5),
        "mix_70_unison_g3_vestas_g2": blend_weather("mix_70_unison_g3_vestas_g2", unison_g3, vestas_g2, 0.7),
        "mix_30_unison_g3_vestas_g2": blend_weather("mix_30_unison_g3_vestas_g2", unison_g3, vestas_g2, 0.3),
    }
    return variants, labels


def main():
    from evaluate_group3_transfer_blend import group2_transfer_tree_candidates

    variants, labels = build_teacher_weather_variants()
    rows = []
    pred_by_recipe = {}
    recipes = []
    for teacher_name, weather in variants.items():
        recipes.append(("unison", teacher_name, weather))
    for teacher_name in ["vestas_g2_teacher", "vestas_g12_avg_teacher"]:
        recipes.append(("vestas", teacher_name, variants[teacher_name]))

    for idx, (physical_manufacturer, teacher_name, weather) in enumerate(recipes, start=1):
        print(f"\n=== {idx}/{len(recipes)} physical={physical_manufacturer}, teacher={teacher_name} ===")
        result = train_single_group(physical_manufacturer, weather, labels, seed=42, verbose=False)
        pred_key = f"{physical_manufacturer}__{teacher_name}"
        pred_by_recipe[pred_key] = {
            "time": result.pop("val_time"),
            "y": result.pop("y"),
            "pred": result.pop("pred"),
        }
        rows.append(
            {
                "physical_manufacturer": physical_manufacturer,
                "teacher": teacher_name,
                **result,
            }
        )
        print(
            f"stage2 score={result['stage2_score']:.6f} "
            f"(nmae={result['stage2_nmae']:.6f}, ficr={result['stage2_ficr']:.6f})"
        )

    results = pd.DataFrame(rows).sort_values("stage2_score", ascending=False).reset_index(drop=True)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    print("\n=== Summary ===")
    print(results.to_string(index=False))

    transfer_tree = group2_transfer_tree_candidates()
    blend_rows = []
    weights = np.linspace(0, 1, 21)
    capacity = GROUP_CAPACITY_KWH[GROUP3]
    for pred_key, pinn_data in pred_by_recipe.items():
        for tree_name, tree_data in transfer_tree.items():
            if not pinn_data["time"].equals(tree_data["time"].reset_index(drop=True)):
                raise ValueError(f"time mismatch: {pred_key} vs {tree_name}")
            for w in weights:
                pred = np.clip(w * pinn_data["pred"] + (1 - w) * tree_data["pred"], 0, capacity)
                score, nmae, ficr = evaluate_predictions(pinn_data["y"], pred, capacity)
                blend_rows.append(
                    {
                        "pinn_recipe": pred_key,
                        "tree_proxy": tree_name,
                        "pinn_weight": w,
                        "score": score,
                        "nmae": nmae,
                        "ficr": ficr,
                    }
                )
    blend_results = pd.DataFrame(blend_rows).sort_values("score", ascending=False).reset_index(drop=True)
    blend_results.to_csv(BLEND_RESULTS_PATH, index=False, encoding="utf-8-sig")
    print("\n=== Best proxy blends ===")
    print(blend_results.head(20).to_string(index=False))
    return results


if __name__ == "__main__":
    main()

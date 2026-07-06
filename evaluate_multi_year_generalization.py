import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold, cross_val_predict
from xgboost import XGBRegressor

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
from utils.power_curve import GROUP_N_TURBINES as TREE_GROUP_N_TURBINES
from utils.power_curve import add_power_curve_feature, fit_group_power_curve
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS, build_group_dataset, build_weather_features

MODELS = {
    "random_forest": RandomForestRegressor(random_state=42, n_jobs=-1),
    "lgbm": LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
    "xgb": XGBRegressor(random_state=42, n_jobs=-1),
}
CV_SPLITTER = KFold(n_splits=3, shuffle=True, random_state=42)
GROUP1 = "kpx_group_1"
GROUP2 = "kpx_group_2"
GROUP3 = "kpx_group_3"
GROUPS = [GROUP1, GROUP2, GROUP3]

YEARLY_FOLDS = [
    {"fold": "year_2023", "val_start": "2023-01-01 01:00:00", "val_end": "2024-01-01 01:00:00"},
    {"fold": "year_2024", "val_start": "2024-01-01 01:00:00", "val_end": "2025-01-01 01:00:00"},
]
QUARTER_TREE_FOLDS = [
    {"fold": "q2024_1", "val_start": "2024-01-01 01:00:00", "val_end": "2024-04-01 01:00:00"},
    {"fold": "q2024_2", "val_start": "2024-04-01 01:00:00", "val_end": "2024-07-01 01:00:00"},
    {"fold": "q2024_3", "val_start": "2024-07-01 01:00:00", "val_end": "2024-10-01 01:00:00"},
    {"fold": "q2024_4", "val_start": "2024-10-01 01:00:00", "val_end": "2025-01-01 01:00:00"},
]

RESULTS_PATH = "results/multi_year_generalization_scores.csv"
SUMMARY_PATH = "results/multi_year_generalization_summary.csv"
MIN_TRAIN_ROWS = 1000
MIN_VAL_ROWS = 200


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def score_arrays(y, pred, capacity):
    pred = np.clip(pred, 0, capacity)
    nmae, ficr = group_nmae_ficr(y, pred, capacity)
    return 0.5 * (1 - nmae) + 0.5 * ficr, nmae, ficr


def time_split_arrays(weather_df, labels_df, group, val_start, val_end):
    weather_df = weather_df.copy()
    weather_df["forecast_kst_dtm"] = pd.to_datetime(weather_df["forecast_kst_dtm"])
    merged = weather_df.merge(
        labels_df[["kst_dtm", group]],
        left_on="forecast_kst_dtm",
        right_on="kst_dtm",
        how="inner",
    )
    merged = merged.dropna(subset=[group]).reset_index(drop=True)
    val_start = pd.Timestamp(val_start)
    val_end = pd.Timestamp(val_end)
    train_mask = merged["forecast_kst_dtm"] < val_start
    val_mask = (merged["forecast_kst_dtm"] >= val_start) & (merged["forecast_kst_dtm"] < val_end)
    feature_cols = [c for c in weather_df.columns if c not in TIME_KEY_COLS]
    return (
        merged.loc[train_mask, feature_cols].reset_index(drop=True),
        merged.loc[val_mask, feature_cols].reset_index(drop=True),
        merged.loc[train_mask, group].reset_index(drop=True),
        merged.loc[val_mask, group].reset_index(drop=True),
        merged.loc[val_mask, "forecast_kst_dtm"].reset_index(drop=True),
    )


def ensemble_fit_predict(x_train, y_train, x_pred):
    preds = []
    for model in MODELS.values():
        fitted = clone(model)
        fitted.fit(x_train, y_train)
        preds.append(fitted.predict(x_pred))
    return sum(preds) / len(preds)


def filter_scada_before(scada_df, val_start):
    scada = scada_df.copy()
    scada["kst_dtm"] = pd.to_datetime(scada["kst_dtm"])
    return scada[scada["kst_dtm"] < pd.Timestamp(val_start)].reset_index(drop=True)


def evaluate_tree_fold(fold, weather, labels, scada_vestas, scada_unison):
    val_start, val_end = fold["val_start"], fold["val_end"]
    scada_by_group = {
        GROUP1: filter_scada_before(scada_vestas, val_start),
        GROUP2: filter_scada_before(scada_vestas, val_start),
        GROUP3: filter_scada_before(scada_unison, val_start),
    }
    per_group = {}
    pooled_pred_pct, pooled_actual_pct = [], []

    for group, capacity in GROUP_CAPACITY_KWH.items():
        if len(scada_by_group[group]) == 0:
            continue
        curve_fn = fit_group_power_curve(scada_by_group[group], group)
        group_weather = add_power_curve_feature(weather, HUB_HEIGHT_PROXY_COL, curve_fn, TREE_GROUP_N_TURBINES[group])
        x_train, x_val, y_train, y_val, val_time = time_split_arrays(group_weather, labels, group, val_start, val_end)
        if len(x_train) < MIN_TRAIN_ROWS or len(x_val) < MIN_VAL_ROWS:
            continue

        oof = np.zeros(len(y_train))
        for model in MODELS.values():
            oof += cross_val_predict(clone(model), x_train, y_train, cv=CV_SPLITTER, n_jobs=1)
        oof /= len(MODELS)
        raw_val = ensemble_fit_predict(x_train, y_train, x_val)
        per_group[group] = {
            "time": val_time,
            "y": y_val.to_numpy(),
            "raw": np.clip(raw_val, 0, capacity),
            "capacity": capacity,
        }
        pooled_pred_pct.append(oof / capacity)
        pooled_actual_pct.append(y_train.to_numpy() / capacity)

    if pooled_pred_pct:
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(np.concatenate(pooled_pred_pct), np.concatenate(pooled_actual_pct))
        for group, data in per_group.items():
            capacity = data["capacity"]
            data["calibrated"] = np.clip(calibrator.predict(data["raw"] / capacity) * capacity, 0, capacity)
    return per_group


def build_teacher_weather(weather_raw, scada_df, group, val_start):
    teacher = fit_scada_wind_teacher(weather_raw, scada_df, group, fit_before=val_start)
    return apply_scada_wind_teacher(weather_raw, teacher)


def blend_weather(weather_a, weather_b, weight_a):
    out = weather_a.copy()
    blend_cols = ["v", "v_std", "scada_ws_mean", "scada_ws_std", "scada_ws_p10", "scada_ws_p50", "scada_ws_p90"]
    for col in blend_cols:
        if col in weather_a.columns and col in weather_b.columns:
            out[col] = weight_a * weather_a[col].to_numpy() + (1 - weight_a) * weather_b[col].to_numpy()
    return out


def train_pinn_fold(physical_manufacturer, group_weather, labels, val_start, val_end):
    seed_everything(42)
    groups = list(group_weather)
    c_max = C_MAX_BY_MANUFACTURER[physical_manufacturer]
    area = MANUFACTURER_AREA[physical_manufacturer]
    turbine_capacity_w = SINGLE_TURBINE_CAPACITY_W[physical_manufacturer]
    model = PowerCurvePINN(c_max, area).to(DEVICE)

    group_data = {}
    v_pool_parts = []
    for group in groups:
        ds = build_group_pinn_dataset(group_weather[group], labels, group)
        val_start_ts = pd.Timestamp(val_start)
        val_end_ts = pd.Timestamp(val_end)
        train_df = ds[ds["forecast_kst_dtm"] < val_start_ts].reset_index(drop=True)
        val_df = ds[(ds["forecast_kst_dtm"] >= val_start_ts) & (ds["forecast_kst_dtm"] < val_end_ts)].reset_index(drop=True)
        if len(train_df) < MIN_TRAIN_ROWS or len(val_df) < MIN_VAL_ROWS:
            continue

        train_years = sorted(train_df["forecast_kst_dtm"].dt.year.unique())
        year_to_idx = {year: idx for idx, year in enumerate(train_years)}
        train_df = train_df.copy()
        train_df["year_idx"] = train_df["forecast_kst_dtm"].dt.year.map(year_to_idx)
        bias = TurbineGroupBias(
            GROUP_CAPACITY_KWH[group],
            n_train_rows=len(train_df) if USE_TRAIN_ONLY_HOUR_BIAS else None,
            n_train_years=len(train_years) if USE_TRAIN_ONLY_YEAR_BIAS else None,
        ).to(DEVICE)
        group_data[group] = {
            "train": train_df,
            "val": val_df,
            "bias": bias,
            "row_idx": torch.arange(len(train_df), dtype=torch.long, device=DEVICE),
            "year_idx": torch.tensor(train_df["year_idx"].to_numpy(), dtype=torch.long, device=DEVICE),
            "capacity": GROUP_CAPACITY_KWH[group],
        }
        v_pool_parts.append(train_df["v"].to_numpy())

    if not group_data:
        return {}

    v_pool = np.concatenate(v_pool_parts)
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
            print(f"[{physical_manufacturer}:{val_start[:10]}] stage1 {epoch}: loss={loss.item():.4f}")

    for p in model.parameters():
        p.requires_grad_(False)
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
        if epoch % 500 == 0 or epoch == STAGE2_EPOCHS - 1:
            bias_diag = sum(bias_l2(data["bias"])["hod"].item() for data in group_data.values())
            print(f"[{physical_manufacturer}:{val_start[:10]}] stage2 {epoch}: data={l_data_sum.item():.4f} hod={bias_diag:.6f}")

    out = {}
    for group, data in group_data.items():
        with torch.no_grad():
            pred = group_prediction(
                model,
                data["val"],
                GROUP_N_TURBINES[group],
                bias=data["bias"],
                use_wind_distribution=True,
            )
            pred = torch.clamp(pred, min=0.0, max=data["capacity"]).cpu().numpy()
        out[group] = {
            "time": pd.to_datetime(data["val"]["forecast_kst_dtm"]).reset_index(drop=True),
            "y": data["val"]["y"].to_numpy(),
            "pred": pred,
            "capacity": data["capacity"],
        }
    return out


def evaluate_yearly_pinn_fold(fold, weather_raw, labels, scada_vestas, scada_unison, use_mixed_group3):
    val_start, val_end = fold["val_start"], fold["val_end"]
    g1_weather = build_teacher_weather(weather_raw, scada_vestas, GROUP1, val_start)
    g2_weather = build_teacher_weather(weather_raw, scada_vestas, GROUP2, val_start)
    vestas_pred = train_pinn_fold("vestas", {GROUP1: g1_weather, GROUP2: g2_weather}, labels, val_start, val_end)

    g3_unison_weather = build_teacher_weather(weather_raw, scada_unison, GROUP3, val_start)
    if use_mixed_group3:
        g3_vestas_weather = build_teacher_weather(weather_raw, scada_vestas, GROUP2, val_start)
        g3_weather = blend_weather(g3_unison_weather, g3_vestas_weather, 0.30)
    else:
        g3_weather = g3_unison_weather
    unison_pred = train_pinn_fold("unison", {GROUP3: g3_weather}, labels, val_start, val_end)
    return {**vestas_pred, **unison_pred}


def append_candidate_rows(rows, fold_name, fold_type, candidate, pred_by_group):
    for group, data in pred_by_group.items():
        score, nmae, ficr = score_arrays(data["y"], data["pred"], data["capacity"])
        rows.append(
            {
                "fold": fold_name,
                "fold_type": fold_type,
                "candidate": candidate,
                "group": group,
                "score": score,
                "nmae": nmae,
                "ficr": ficr,
                "n": len(data["y"]),
            }
        )


def add_fold_means(results):
    group_rows = results[results["group"] != "mean"].copy()
    means = (
        group_rows.groupby(["fold", "fold_type", "candidate"], as_index=False)
        .agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"), n=("n", "sum"))
    )
    means["group"] = "mean"
    return pd.concat([group_rows, means[results.columns]], ignore_index=True)


def summarize(results):
    means = results[results["group"] == "mean"].copy()
    summary = (
        means.groupby(["fold_type", "candidate"], as_index=False)
        .agg(
            mean_score=("score", "mean"),
            std_score=("score", "std"),
            worst_fold=("score", "min"),
            best_fold=("score", "max"),
            n_folds=("score", "count"),
        )
        .sort_values(["fold_type", "mean_score"], ascending=[True, False])
    )
    return summary


def main():
    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    weather_tree = build_weather_features(ldaps_train, gfs_train)
    weather_pinn = build_pinn_weather(ldaps_train, gfs_train)
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")

    rows = []
    for fold in YEARLY_FOLDS:
        print(f"\n=== yearly fold {fold['fold']} ===")
        tree = evaluate_tree_fold(fold, weather_tree, labels, scada_vestas, scada_unison)
        append_candidate_rows(rows, fold["fold"], "yearly", "tree_raw", {
            g: {**d, "pred": d["raw"]} for g, d in tree.items()
        })
        append_candidate_rows(rows, fold["fold"], "yearly", "tree_calibrated", {
            g: {**d, "pred": d["calibrated"]} for g, d in tree.items()
        })

        pinn_standard = evaluate_yearly_pinn_fold(fold, weather_pinn, labels, scada_vestas, scada_unison, False)
        append_candidate_rows(rows, fold["fold"], "yearly", "pinn_standard", pinn_standard)

        pinn_mixed = evaluate_yearly_pinn_fold(fold, weather_pinn, labels, scada_vestas, scada_unison, True)
        append_candidate_rows(rows, fold["fold"], "yearly", "pinn_mixed_g3", pinn_mixed)

        for w_pinn in [0.1, 0.2, 0.3, 0.5]:
            blended = {}
            for group in sorted(set(tree) & set(pinn_mixed)):
                if not tree[group]["time"].reset_index(drop=True).equals(pinn_mixed[group]["time"].reset_index(drop=True)):
                    raise ValueError(f"time mismatch {fold['fold']} {group}")
                capacity = GROUP_CAPACITY_KWH[group]
                pred = np.clip(w_pinn * pinn_mixed[group]["pred"] + (1 - w_pinn) * tree[group]["calibrated"], 0, capacity)
                blended[group] = {**tree[group], "pred": pred}
            append_candidate_rows(rows, fold["fold"], "yearly", f"blend_tree_cal_pinn{w_pinn:.1f}", blended)

    for fold in QUARTER_TREE_FOLDS:
        print(f"\n=== quarter tree fold {fold['fold']} ===")
        tree = evaluate_tree_fold(fold, weather_tree, labels, scada_vestas, scada_unison)
        append_candidate_rows(rows, fold["fold"], "quarter_tree", "tree_raw", {
            g: {**d, "pred": d["raw"]} for g, d in tree.items()
        })
        append_candidate_rows(rows, fold["fold"], "quarter_tree", "tree_calibrated", {
            g: {**d, "pred": d["calibrated"]} for g, d in tree.items()
        })

    results = add_fold_means(pd.DataFrame(rows))
    summary = summarize(results)
    results.to_csv(RESULTS_PATH, index=False, encoding="utf-8-sig")
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8-sig")

    print("\n=== Yearly candidate summary ===")
    print(summary[summary["fold_type"] == "yearly"].to_string(index=False))
    print("\n=== Quarter tree summary ===")
    print(summary[summary["fold_type"] == "quarter_tree"].to_string(index=False))
    print("\n=== Yearly fold means ===")
    print(results[(results["fold_type"] == "yearly") & (results["group"] == "mean")].to_string(index=False))
    return results, summary


if __name__ == "__main__":
    main()

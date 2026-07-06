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
from utils.metrics import GROUP_CAPACITY_KWH
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
SUBMISSION_PATH = "results/submission.csv"


def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensemble_fit_predict(x_train, y_train, x_pred):
    preds = []
    for model in MODELS.values():
        fitted = clone(model)
        fitted.fit(x_train, y_train)
        preds.append(fitted.predict(x_pred))
    return sum(preds) / len(preds)


def blend_weather(weather_a, weather_b, weight_a):
    out = weather_a.copy()
    blend_cols = ["v", "v_std", "scada_ws_mean", "scada_ws_std", "scada_ws_p10", "scada_ws_p50", "scada_ws_p90"]
    for col in blend_cols:
        if col in weather_a.columns and col in weather_b.columns:
            out[col] = weight_a * weather_a[col].to_numpy() + (1 - weight_a) * weather_b[col].to_numpy()
    return out


def build_tree_test_predictions():
    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    weather_train = build_weather_features(ldaps_train, gfs_train)
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")

    ldaps_test = pd.read_csv("data/test/ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv("data/test/gfs_test.csv", encoding="utf-8-sig")
    weather_test = build_weather_features(ldaps_test, gfs_test)

    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada_by_group = {GROUP1: scada_vestas, GROUP2: scada_vestas, GROUP3: scada_unison}

    out = {}
    pooled_oof_pred_pct, pooled_oof_actual_pct = [], []
    per_group = {}
    for group, capacity in GROUP_CAPACITY_KWH.items():
        curve_fn = fit_group_power_curve(scada_by_group[group], group)
        n_turbines = TREE_GROUP_N_TURBINES[group]
        group_weather_train = add_power_curve_feature(weather_train, HUB_HEIGHT_PROXY_COL, curve_fn, n_turbines)
        group_weather_test = add_power_curve_feature(weather_test, HUB_HEIGHT_PROXY_COL, curve_fn, n_turbines)

        x_train, y_train = build_group_dataset(group_weather_train, labels, group)
        feature_cols = [c for c in group_weather_test.columns if c not in TIME_KEY_COLS]
        x_test = group_weather_test[feature_cols]

        oof = np.zeros(len(y_train))
        for model in MODELS.values():
            oof += cross_val_predict(clone(model), x_train, y_train, cv=CV_SPLITTER, n_jobs=1)
        oof /= len(MODELS)
        raw_test = ensemble_fit_predict(x_train, y_train, x_test)

        per_group[group] = {
            "time": pd.to_datetime(group_weather_test["forecast_kst_dtm"]).reset_index(drop=True),
            "raw": np.clip(raw_test, 0, capacity),
            "capacity": capacity,
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
        }
        pooled_oof_pred_pct.append(oof / capacity)
        pooled_oof_actual_pct.append(y_train.to_numpy() / capacity)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(np.concatenate(pooled_oof_pred_pct), np.concatenate(pooled_oof_actual_pct))
    for group, data in per_group.items():
        capacity = data["capacity"]
        out[group] = {
            "time": data["time"],
            "raw": data["raw"],
            "calibrated": np.clip(calibrator.predict(data["raw"] / capacity) * capacity, 0, capacity),
        }

    group2_curve = fit_group_power_curve(scada_vestas, GROUP2)
    group3_curve = fit_group_power_curve(scada_unison, GROUP3)
    group2_weather_train = add_power_curve_feature(
        weather_train, HUB_HEIGHT_PROXY_COL, group2_curve, TREE_GROUP_N_TURBINES[GROUP2]
    )
    group3_weather_test = add_power_curve_feature(
        weather_test, HUB_HEIGHT_PROXY_COL, group3_curve, TREE_GROUP_N_TURBINES[GROUP3]
    )
    x2_train, y2_train = build_group_dataset(group2_weather_train, labels, GROUP2)
    feature_cols = [c for c in group3_weather_test.columns if c not in TIME_KEY_COLS]
    group2_transfer = ensemble_fit_predict(x2_train, y2_train, group3_weather_test[feature_cols])
    out["group2_transfer_on_group3"] = {
        "time": pd.to_datetime(group3_weather_test["forecast_kst_dtm"]).reset_index(drop=True),
        "pred": np.clip(
            group2_transfer / GROUP_CAPACITY_KWH[GROUP2] * GROUP_CAPACITY_KWH[GROUP3],
            0,
            GROUP_CAPACITY_KWH[GROUP3],
        ),
    }
    return out


def build_pinn_weather_train_test():
    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    ldaps_test = pd.read_csv("data/test/ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv("data/test/gfs_test.csv", encoding="utf-8-sig")

    weather_train_raw = build_pinn_weather(ldaps_train, gfs_train)
    weather_test_raw = build_pinn_weather(ldaps_test, gfs_test)

    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")

    def teacher_weather(scada_df, group):
        teacher = fit_scada_wind_teacher(weather_train_raw, scada_df, group, fit_before=None)
        return apply_scada_wind_teacher(weather_train_raw, teacher), apply_scada_wind_teacher(weather_test_raw, teacher)

    g1_train, g1_test = teacher_weather(scada_vestas, GROUP1)
    g2_train, g2_test = teacher_weather(scada_vestas, GROUP2)
    g3_unison_train, g3_unison_test = teacher_weather(scada_unison, GROUP3)
    g3_vestas_train, g3_vestas_test = teacher_weather(scada_vestas, GROUP2)

    return {
        "train": {
            GROUP1: g1_train,
            GROUP2: g2_train,
            GROUP3: blend_weather(g3_unison_train, g3_vestas_train, 0.30),
        },
        "test": {
            GROUP1: g1_test,
            GROUP2: g2_test,
            GROUP3: blend_weather(g3_unison_test, g3_vestas_test, 0.30),
        },
    }


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
            print(f"[{physical_manufacturer}] stage2 epoch {epoch}: data={l_data_sum.item():.4f} hod_l2={bias_diag:.6f}")

    return model, {group: data["bias"] for group, data in group_data.items()}


def predict_pinn(model, bias, group, weather_test):
    with torch.no_grad():
        pred = group_prediction(model, weather_test, GROUP_N_TURBINES[group], bias=bias, use_wind_distribution=True)
        return torch.clamp(pred, min=0.0, max=GROUP_CAPACITY_KWH[group]).cpu().numpy()


def build_pinn_test_predictions():
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    weather = build_pinn_weather_train_test()

    vestas_model, vestas_bias = train_full_pinn(
        "vestas",
        {GROUP1: weather["train"][GROUP1], GROUP2: weather["train"][GROUP2]},
        labels,
    )
    unison_model, unison_bias = train_full_pinn("unison", {GROUP3: weather["train"][GROUP3]}, labels)

    return {
        GROUP1: {
            "time": pd.to_datetime(weather["test"][GROUP1]["forecast_kst_dtm"]).reset_index(drop=True),
            "pred": predict_pinn(vestas_model, vestas_bias[GROUP1], GROUP1, weather["test"][GROUP1]),
        },
        GROUP2: {
            "time": pd.to_datetime(weather["test"][GROUP2]["forecast_kst_dtm"]).reset_index(drop=True),
            "pred": predict_pinn(vestas_model, vestas_bias[GROUP2], GROUP2, weather["test"][GROUP2]),
        },
        GROUP3: {
            "time": pd.to_datetime(weather["test"][GROUP3]["forecast_kst_dtm"]).reset_index(drop=True),
            "pred": predict_pinn(unison_model, unison_bias[GROUP3], GROUP3, weather["test"][GROUP3]),
        },
    }


def main():
    tree = build_tree_test_predictions()
    pinn = build_pinn_test_predictions()

    pred = {}
    for group in GROUPS:
        if not tree[group]["time"].equals(pinn[group]["time"]):
            raise ValueError(f"time mismatch for {group}")

    pred[GROUP1] = np.clip(
        0.70 * pinn[GROUP1]["pred"] + 0.30 * tree[GROUP1]["calibrated"],
        0,
        GROUP_CAPACITY_KWH[GROUP1],
    )

    group2_anchor = np.clip(
        0.70 * pinn[GROUP2]["pred"] + 0.30 * tree[GROUP2]["calibrated"],
        0,
        GROUP_CAPACITY_KWH[GROUP2],
    )
    pred[GROUP2] = np.clip(0.85 * group2_anchor + 0.15 * tree[GROUP2]["raw"], 0, GROUP_CAPACITY_KWH[GROUP2])

    if not tree[GROUP3]["time"].equals(tree["group2_transfer_on_group3"]["time"]):
        raise ValueError("group3 transfer tree time mismatch")
    group3_anchor = np.clip(
        0.70 * pinn[GROUP3]["pred"] + 0.30 * tree["group2_transfer_on_group3"]["pred"],
        0,
        GROUP_CAPACITY_KWH[GROUP3],
    )
    pred[GROUP3] = np.clip(0.90 * group3_anchor + 0.10 * tree[GROUP3]["calibrated"], 0, GROUP_CAPACITY_KWH[GROUP3])

    prediction = pd.DataFrame({"forecast_kst_dtm": tree[GROUP1]["time"]})
    for group in GROUPS:
        prediction[group] = pred[group]

    submission = pd.read_csv("data/sample_submission.csv", encoding="utf-8-sig")
    submission["forecast_kst_dtm"] = pd.to_datetime(submission["forecast_kst_dtm"])
    merged = submission[["forecast_id", "forecast_kst_dtm"]].merge(prediction, on="forecast_kst_dtm", how="left")

    if merged[GROUPS].isna().any().any():
        missing = merged[merged[GROUPS].isna().any(axis=1)].head()
        raise ValueError(f"submission has missing predictions:\n{missing}")
    merged = merged[["forecast_id", "forecast_kst_dtm", GROUP1, GROUP2, GROUP3]]
    if len(merged) != len(submission):
        raise ValueError(f"row count mismatch: {len(merged)} vs {len(submission)}")

    merged.to_csv(SUBMISSION_PATH, index=False, encoding="utf-8-sig")
    print(f"saved {SUBMISSION_PATH}: {merged.shape}")
    print(merged.head())
    print(merged.tail())
    return merged


if __name__ == "__main__":
    main()

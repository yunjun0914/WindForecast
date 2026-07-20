from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import DataLoader, TensorDataset

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # noqa: F401
from models.group_unified import (
    BoundedResidualMLP,
    GroupPhysicsPINN,
    normalized_metric_loss,
)
from models.seqnn import TCNPowerRegressor
from utils.metrics import (
    GROUP_CAPACITY_KWH,
    TARGET_COLS,
    group_nmae_ficr,
    pooled_oof_summary,
)
from utils.per_turbine_features import GROUP_WAKE_FEATURES, build_group_wake_features
from utils.per_turbine_scada import align_scada_hour, build_turbine_scada_hourly
from utils.pinn_data import C_MAX_BY_MANUFACTURER
from utils.pinn_physics import MANUFACTURER_AREA
from utils.power_curve import GROUP_MANUFACTURER, GROUP_N_TURBINES, add_power_curve_feature_oof
from utils.preprocessing import HUB_HEIGHT_PROXY_COL, TIME_KEY_COLS
from utils.seq_dataset import SequenceStandardScaler, make_sequences
from utils.tree_feature_profiles import (
    FEATURE_PROFILE_FULL_V2,
    GROUP_FAMILY_QUOTA65_V1_FEATURES,
    build_tree_features,
)
from utils.weather_time_features import add_weather_time_features


YEARS = [2022, 2023, 2024]
BASE_EXTRA_FEATURES = [*GROUP_WAKE_FEATURES, "phys_gfs_air_density", "power_curve_est"]
TEACHER_TARGETS = ["teacher_ws_cubic", "teacher_ws_std", "teacher_wd_sin", "teacher_wd_cos"]
PREDICTION_COLS = ["pinn_pred", "tree_pred", "tcn_pred"]


@dataclass
class GroupTables:
    base_static: pd.DataFrame
    residual_static: pd.DataFrame
    residual_cols: list[str]
    labels: pd.DataFrame
    scada: pd.DataFrame
    teacher_targets: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/group_unified_v1.json"))
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--stem", default="group_unified_v1")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-outer-folds", type=int, default=0)
    return parser.parse_args()


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: Path, smoke_test: bool) -> dict:
    config = json.loads(path.read_text(encoding="utf-8"))
    if smoke_test:
        config["teacher_trees"] = min(24, int(config["teacher_trees"]))
        config["tree"]["n_estimators"] = min(80, int(config["tree"]["n_estimators"]))
        config["pinn"]["epochs"] = 2
        config["tcn"]["epochs"] = 2
        config["residual"]["epochs"] = 2
    weights = config["ensemble_weights"]
    if abs(sum(float(weights[name]) for name in ["pinn", "tree", "tcn"]) - 1.0) > 1e-9:
        raise ValueError("ensemble weights must sum to 1")
    return config


def read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }
    return ldaps, gfs, labels, scada


def build_group_scada_targets(scada: pd.DataFrame, group: str) -> pd.DataFrame:
    hourly = build_turbine_scada_hourly(scada, group)
    n_turbines = GROUP_N_TURBINES[group]

    def summarize(part: pd.DataFrame) -> pd.Series:
        cubic = pd.to_numeric(part["scada_ws_cubic"], errors="coerce").to_numpy(float)
        wd_sin = pd.to_numeric(part["scada_wd_sin"], errors="coerce").to_numpy(float)
        wd_cos = pd.to_numeric(part["scada_wd_cos"], errors="coerce").to_numpy(float)
        valid = np.isfinite(cubic)
        if int(valid.sum()) < n_turbines:
            return pd.Series({name: np.nan for name in TEACHER_TARGETS})
        return pd.Series(
            {
                "teacher_ws_cubic": float(np.cbrt(np.mean(np.clip(cubic[valid], 0, None) ** 3))),
                "teacher_ws_std": float(np.std(cubic[valid], ddof=0)),
                "teacher_wd_sin": float(np.nanmean(wd_sin)),
                "teacher_wd_cos": float(np.nanmean(wd_cos)),
            }
        )

    return (
        hourly.groupby("forecast_kst_dtm", sort=True)
        .apply(summarize, include_groups=False)
        .reset_index()
        .dropna(subset=TEACHER_TARGETS)
    )


def build_group_tables(
    ldaps: pd.DataFrame,
    gfs: pd.DataFrame,
    labels: pd.DataFrame,
    scada: pd.DataFrame,
    group: str,
    include_residual: bool = True,
) -> GroupTables:
    full = build_tree_features(ldaps, gfs, group, feature_profile=FEATURE_PROFILE_FULL_V2)
    expanded = add_weather_time_features(full)
    for col in TIME_KEY_COLS:
        expanded[col] = pd.to_datetime(expanded[col])

    quota_cols = GROUP_FAMILY_QUOTA65_V1_FEATURES[group]
    required = [*quota_cols, "phys_gfs_air_density"]
    missing = [col for col in required if col not in expanded.columns]
    if missing:
        raise ValueError(f"{group} base features missing: {missing}")

    wake = build_group_wake_features(ldaps, gfs, group)
    for col in TIME_KEY_COLS:
        wake[col] = pd.to_datetime(wake[col])
    base_static = expanded[TIME_KEY_COLS + required].merge(
        wake[TIME_KEY_COLS + GROUP_WAKE_FEATURES], on=TIME_KEY_COLS, how="inner", validate="one_to_one"
    )

    if include_residual:
        excluded = set(TIME_KEY_COLS + quota_cols + ["phys_gfs_air_density"])
        residual_cols = [
            col
            for col in expanded.columns
            if col not in excluded and pd.api.types.is_numeric_dtype(expanded[col])
        ]
        if not residual_cols:
            raise ValueError(f"{group} has no residual-only features")
        residual_static = expanded[TIME_KEY_COLS + residual_cols].copy()
    else:
        residual_cols = []
        residual_static = expanded[TIME_KEY_COLS].copy()
    label_table = labels[["kst_dtm", group]].rename(columns={group: "target"}).copy()
    return GroupTables(
        base_static=base_static,
        residual_static=residual_static,
        residual_cols=residual_cols,
        labels=label_table,
        scada=scada,
        teacher_targets=build_group_scada_targets(scada, group),
    )


def filter_scada_to_times(scada: pd.DataFrame, group: str, times: pd.Series) -> pd.DataFrame:
    available = set(pd.to_datetime(times).to_numpy(dtype="datetime64[ns]"))
    aligned = align_scada_hour(scada["kst_dtm"], group).to_numpy(dtype="datetime64[ns]")
    return scada.loc[np.isin(aligned, list(available))].reset_index(drop=True)


def select_times(table: pd.DataFrame, times: pd.Series | np.ndarray) -> pd.DataFrame:
    selected = set(pd.to_datetime(times).to_numpy(dtype="datetime64[ns]"))
    values = pd.to_datetime(table["forecast_kst_dtm"]).to_numpy(dtype="datetime64[ns]")
    return table.loc[np.isin(values, list(selected))].copy().reset_index(drop=True)


def build_fold_feature_tables(
    tables: GroupTables,
    group: str,
    train_times: pd.Series | np.ndarray,
    val_times: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_base = select_times(tables.base_static, train_times)
    val_base = select_times(tables.base_static, val_times)
    fit_scada = filter_scada_to_times(tables.scada, group, train_base["forecast_kst_dtm"])
    if len(fit_scada) < 1000:
        raise ValueError(f"{group} fold has insufficient SCADA rows: {len(fit_scada)}")
    train_base, val_base = add_power_curve_feature_oof(
        train_base,
        val_base,
        fit_scada,
        group,
        HUB_HEIGHT_PROXY_COL,
        GROUP_N_TURBINES[group],
        out_col="power_curve_est",
        clean=True,
    )
    base_cols = [*GROUP_FAMILY_QUOTA65_V1_FEATURES[group], *BASE_EXTRA_FEATURES]
    missing = [col for col in base_cols if col not in train_base.columns]
    if missing:
        raise ValueError(f"{group} fold base columns missing: {missing}")

    return train_base.reset_index(drop=True), val_base.reset_index(drop=True), base_cols


def build_fold_tables(
    tables: GroupTables,
    group: str,
    train_times: pd.Series | np.ndarray,
    val_times: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    train_base, val_base, base_cols = build_fold_feature_tables(
        tables,
        group,
        train_times,
        val_times,
    )

    labels = tables.labels.copy()
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    train = train_base.merge(
        labels, left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner", validate="one_to_one"
    ).dropna(subset=["target"])
    val = val_base.merge(
        labels, left_on="forecast_kst_dtm", right_on="kst_dtm", how="inner", validate="one_to_one"
    ).dropna(subset=["target"])
    return train.reset_index(drop=True), val.reset_index(drop=True), base_cols


def teacher_matrix(table: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    values = table[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return values.to_numpy(np.float32)


def fit_teacher(
    train: pd.DataFrame,
    val: pd.DataFrame,
    targets: pd.DataFrame,
    feature_cols: list[str],
    trees: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    target_table = targets.copy()
    target_table["forecast_kst_dtm"] = pd.to_datetime(target_table["forecast_kst_dtm"])
    fit = train.merge(target_table, on="forecast_kst_dtm", how="inner").dropna(subset=TEACHER_TARGETS)
    if len(fit) < 1000:
        raise ValueError(f"Teacher has insufficient rows: {len(fit)}")
    model = RandomForestRegressor(
        n_estimators=trees,
        min_samples_leaf=20,
        max_features=0.70,
        max_samples=0.80,
        bootstrap=True,
        oob_score=True,
        random_state=seed,
        n_jobs=8,
    )
    model.fit(teacher_matrix(fit, feature_cols), fit[TEACHER_TARGETS].to_numpy(float))
    train_prediction = model.predict(teacher_matrix(train, feature_cols))
    val_prediction = model.predict(teacher_matrix(val, feature_cols))

    oob = pd.DataFrame(model.oob_prediction_, columns=TEACHER_TARGETS)
    oob["forecast_kst_dtm"] = pd.to_datetime(fit["forecast_kst_dtm"]).to_numpy()
    oob = oob.drop_duplicates("forecast_kst_dtm")
    aligned = train[["forecast_kst_dtm"]].merge(oob, on="forecast_kst_dtm", how="left")
    oob_values = aligned[TEACHER_TARGETS].to_numpy(float)
    use_oob = np.isfinite(oob_values).all(axis=1)
    train_prediction[use_oob] = oob_values[use_oob]
    return train_prediction.astype(np.float32), val_prediction.astype(np.float32)


def calendar_arrays(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    forecast = pd.to_datetime(table["forecast_kst_dtm"])
    available = pd.to_datetime(table["data_available_kst_dtm"])
    doy = forecast.dt.dayofyear.to_numpy(np.float32)
    lead = ((forecast - available).dt.total_seconds() / 3600.0).to_numpy(np.float32)
    return doy, lead


def predict_pinn(
    model: GroupPhysicsPINN,
    teacher: np.ndarray,
    rho: np.ndarray,
    doy: np.ndarray,
    lead: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(teacher), batch_size):
            stop = start + batch_size
            prediction = model(
                torch.from_numpy(teacher[start:stop]).to(device),
                torch.from_numpy(rho[start:stop]).to(device),
                torch.from_numpy(doy[start:stop]).to(device),
                torch.from_numpy(lead[start:stop]).to(device),
            )
            parts.append(prediction.cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def fit_predict_pinn(
    train: pd.DataFrame,
    val: pd.DataFrame,
    teacher_train: np.ndarray,
    teacher_val: np.ndarray,
    group: str,
    config: dict,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict]:
    set_seed(seed)
    capacity = GROUP_CAPACITY_KWH[group]
    manufacturer = GROUP_MANUFACTURER[group]
    model = GroupPhysicsPINN(
        n_turbines=GROUP_N_TURBINES[group],
        rotor_area_m2=MANUFACTURER_AREA[manufacturer],
        rated_power_w=capacity * 1000.0,
        c_max=C_MAX_BY_MANUFACTURER[manufacturer],
        hidden_size=int(config["hidden_size"]),
        residual_amplitude=float(config["residual_amplitude"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    y = (train["target"].to_numpy(np.float32) / capacity).astype(np.float32)
    rho = train["phys_gfs_air_density"].to_numpy(np.float32)
    doy, lead = calendar_arrays(train)
    keep = np.isfinite(y) & (y >= 0.10)
    dataset = TensorDataset(
        torch.from_numpy(teacher_train[keep]),
        torch.from_numpy(rho[keep]),
        torch.from_numpy(doy[keep]),
        torch.from_numpy(lead[keep]),
        torch.from_numpy(y[keep]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    last_loss = np.nan
    for _ in range(int(config["epochs"])):
        model.train()
        losses = []
        for teacher_b, rho_b, doy_b, lead_b, target_b in loader:
            teacher_b = teacher_b.to(device, non_blocking=True)
            rho_b = rho_b.to(device, non_blocking=True)
            doy_b = doy_b.to(device, non_blocking=True)
            lead_b = lead_b.to(device, non_blocking=True)
            target_b = target_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(teacher_b, rho_b, doy_b, lead_b)
            data_loss, _, _ = normalized_metric_loss(
                prediction,
                target_b,
                gamma=float(config["gamma"]),
                nmae_weight=float(config["nmae_weight"]),
                ficr_weight=float(config["ficr_weight"]),
            )
            regularization = model.physics_regularization(device)
            loss = (
                data_loss
                + float(config["boundary_weight"]) * regularization["boundary"]
                + float(config["smoothness_weight"]) * regularization["smoothness"]
                + float(config["flatness_weight"]) * regularization["flatness"]
                + float(config["residual_l2_weight"]) * regularization["residual_l2"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses))

    val_rho = val["phys_gfs_air_density"].to_numpy(np.float32)
    val_doy, val_lead = calendar_arrays(val)
    pred_norm = predict_pinn(
        model,
        teacher_val,
        val_rho,
        val_doy,
        val_lead,
        device,
        int(config["batch_size"]) * 4,
    )
    return pred_norm * capacity, {"train_loss": last_loss, "epochs": int(config["epochs"])}


def fit_predict_tree(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    group: str,
    config: dict,
    seed: int,
) -> tuple[np.ndarray, dict]:
    capacity = GROUP_CAPACITY_KWH[group]
    keep = train["target"].to_numpy(float) >= capacity * 0.10
    x_train = train.loc[keep, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_val = val[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    params = dict(config)
    params.update(
        {
            "objective": "regression",
            "random_state": seed,
            "n_jobs": 8,
            "verbose": -1,
        }
    )
    model = LGBMRegressor(**params)
    model.fit(x_train, train.loc[keep, "target"].to_numpy(float))
    prediction = np.clip(model.predict(x_val), 0.0, capacity)
    return prediction.astype(np.float32), {"n_train": int(keep.sum()), "trees": int(params["n_estimators"])}


def tcn_predict(
    model: TCNPowerRegressor,
    features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            parts.append(torch.sigmoid(model(batch)).cpu().numpy())
    return np.concatenate(parts).astype(np.float32)


def fit_predict_tcn(
    train: pd.DataFrame,
    val: pd.DataFrame,
    feature_cols: list[str],
    group: str,
    config: dict,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    set_seed(seed)
    capacity = GROUP_CAPACITY_KWH[group]
    x_train, y_train, _ = make_sequences(
        train, feature_cols, window=int(config["window"]), target_col="target"
    )
    x_val, _, val_times = make_sequences(
        val, feature_cols, window=int(config["window"]), target_col="target"
    )
    scaler = SequenceStandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    y_norm = (y_train / capacity).astype(np.float32)
    keep = np.isfinite(y_norm) & (y_norm >= 0.10)
    weights = (0.5 + np.sqrt(np.clip(y_norm[keep], 0.0, 1.0))).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(x_train[keep]),
        torch.from_numpy(y_norm[keep]),
        torch.from_numpy(weights),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    model = TCNPowerRegressor(
        input_size=len(feature_cols),
        hidden_size=int(config["hidden_size"]),
        num_layers=int(config["num_layers"]),
        kernel_size=int(config["kernel_size"]),
        dropout=float(config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    last_loss = np.nan
    for _ in range(int(config["epochs"])):
        model.train()
        losses = []
        for features_b, target_b, weight_b in loader:
            features_b = features_b.to(device, non_blocking=True)
            target_b = target_b.to(device, non_blocking=True)
            weight_b = weight_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = torch.sigmoid(model(features_b))
            loss, _, _ = normalized_metric_loss(
                prediction,
                target_b,
                gamma=float(config["gamma"]),
                nmae_weight=float(config["nmae_weight"]),
                ficr_weight=float(config["ficr_weight"]),
                sample_weight=weight_b,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses))
    prediction = tcn_predict(model, x_val, device, int(config["batch_size"]) * 4) * capacity
    return (
        pd.DataFrame({"forecast_kst_dtm": pd.to_datetime(val_times), "tcn_pred": prediction}),
        {"train_loss": last_loss, "epochs": int(config["epochs"]), "n_train": int(keep.sum())},
    )


def run_base_fold(
    tables: GroupTables,
    group: str,
    train_times: pd.Series | np.ndarray,
    val_times: pd.Series | np.ndarray,
    config: dict,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, list[dict]]:
    train, val, feature_cols = build_fold_tables(tables, group, train_times, val_times)
    teacher_train, teacher_val = fit_teacher(
        train,
        val,
        tables.teacher_targets,
        feature_cols,
        int(config["teacher_trees"]),
        seed,
    )
    pinn_pred, pinn_stats = fit_predict_pinn(
        train, val, teacher_train, teacher_val, group, config["pinn"], device, seed + 11
    )
    tree_pred, tree_stats = fit_predict_tree(
        train, val, feature_cols, group, config["tree"], seed + 23
    )
    tcn_frame, tcn_stats = fit_predict_tcn(
        train, val, feature_cols, group, config["tcn"], device, seed + 37
    )
    output = val[["forecast_kst_dtm", "target"]].copy()
    output["pinn_pred"] = pinn_pred
    output["tree_pred"] = tree_pred
    output = output.merge(tcn_frame, on="forecast_kst_dtm", how="inner", validate="one_to_one")
    diagnostics = [
        {"model": "pinn", **pinn_stats},
        {"model": "tree", **tree_stats},
        {"model": "tcn", **tcn_stats},
    ]
    return output, diagnostics


def available_target_times(tables: GroupTables) -> pd.DataFrame:
    labels = tables.labels.dropna(subset=["target"]).copy()
    labels["forecast_kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    available = tables.base_static[["forecast_kst_dtm", "data_available_kst_dtm"]].merge(
        labels[["forecast_kst_dtm", "target"]], on="forecast_kst_dtm", how="inner", validate="one_to_one"
    )
    available["year"] = available["forecast_kst_dtm"].dt.year
    return available.loc[available["year"].isin(YEARS)].reset_index(drop=True)


def inner_splits(outer_train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, str]]:
    years = sorted(outer_train["year"].unique().tolist())
    if len(years) >= 2:
        return [
            (
                outer_train.loc[outer_train["year"].ne(year), "forecast_kst_dtm"].to_numpy(),
                outer_train.loc[outer_train["year"].eq(year), "forecast_kst_dtm"].to_numpy(),
                f"year_{year}",
            )
            for year in years
        ]

    issues = np.asarray(sorted(pd.to_datetime(outer_train["data_available_kst_dtm"]).unique()))
    halves = np.array_split(issues, 2)
    splits = []
    for index, val_issues in enumerate(halves):
        val_mask = outer_train["data_available_kst_dtm"].isin(val_issues)
        splits.append(
            (
                outer_train.loc[~val_mask, "forecast_kst_dtm"].to_numpy(),
                outer_train.loc[val_mask, "forecast_kst_dtm"].to_numpy(),
                f"issue_half_{index}",
            )
        )
    return splits


def ensemble_prediction(frame: pd.DataFrame, weights: dict) -> np.ndarray:
    return (
        float(weights["pinn"]) * frame["pinn_pred"].to_numpy(float)
        + float(weights["tree"]) * frame["tree_pred"].to_numpy(float)
        + float(weights["tcn"]) * frame["tcn_pred"].to_numpy(float)
    )


def residual_matrix(
    predictions: pd.DataFrame,
    residual_static: pd.DataFrame,
    residual_cols: list[str],
    capacity: float,
    weights: dict,
) -> np.ndarray:
    table = predictions.merge(
        residual_static[["forecast_kst_dtm", *residual_cols]],
        on="forecast_kst_dtm",
        how="inner",
        validate="one_to_one",
    )
    if not np.array_equal(
        pd.to_datetime(table["forecast_kst_dtm"]).to_numpy(),
        pd.to_datetime(predictions["forecast_kst_dtm"]).to_numpy(),
    ):
        raise ValueError("Residual feature merge changed prediction row order")
    branch = table[PREDICTION_COLS].to_numpy(float) / capacity
    ensemble = ensemble_prediction(table, weights) / capacity
    prediction_features = np.column_stack(
        [
            branch,
            ensemble,
            branch.mean(axis=1),
            branch.std(axis=1),
            branch.max(axis=1) - branch.min(axis=1),
            branch[:, 0] - branch[:, 1],
            branch[:, 0] - branch[:, 2],
            branch[:, 1] - branch[:, 2],
        ]
    )
    raw = table[residual_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(float)
    return np.column_stack([raw, prediction_features]).astype(np.float32)


def fit_predict_residual(
    train_predictions: pd.DataFrame,
    val_predictions: pd.DataFrame,
    tables: GroupTables,
    group: str,
    weights: dict,
    config: dict,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict]:
    set_seed(seed)
    capacity = GROUP_CAPACITY_KWH[group]
    x_train = residual_matrix(
        train_predictions, tables.residual_static, tables.residual_cols, capacity, weights
    )
    x_val = residual_matrix(
        val_predictions, tables.residual_static, tables.residual_cols, capacity, weights
    )
    mean = np.nanmean(x_train, axis=0).astype(np.float32)
    std = np.nanstd(x_train, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    x_train = np.nan_to_num((x_train - mean) / std).astype(np.float32)
    x_val = np.nan_to_num((x_val - mean) / std).astype(np.float32)
    y_train = (train_predictions["target"].to_numpy(np.float32) / capacity).astype(np.float32)
    base_train = (ensemble_prediction(train_predictions, weights) / capacity).astype(np.float32)
    base_val = (ensemble_prediction(val_predictions, weights) / capacity).astype(np.float32)
    keep = np.isfinite(y_train) & (y_train >= 0.10)
    dataset = TensorDataset(
        torch.from_numpy(x_train[keep]),
        torch.from_numpy(base_train[keep]),
        torch.from_numpy(y_train[keep]),
    )
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    model = BoundedResidualMLP(
        input_size=x_train.shape[1],
        hidden_size=int(config["hidden_size"]),
        dropout=float(config["dropout"]),
        max_delta=float(config["max_delta"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    last_loss = np.nan
    for _ in range(int(config["epochs"])):
        model.train()
        losses = []
        for features_b, base_b, target_b in loader:
            features_b = features_b.to(device, non_blocking=True)
            base_b = base_b.to(device, non_blocking=True)
            target_b = target_b.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            delta = model(features_b)
            prediction = torch.clamp(base_b + delta, 0.0, 1.0)
            metric_loss, _, _ = normalized_metric_loss(
                prediction,
                target_b,
                gamma=float(config["gamma"]),
                nmae_weight=float(config["nmae_weight"]),
                ficr_weight=float(config["ficr_weight"]),
            )
            loss = metric_loss + float(config["delta_l2_weight"]) * delta.pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last_loss = float(np.mean(losses))

    model.eval()
    delta_parts = []
    with torch.no_grad():
        for start in range(0, len(x_val), int(config["batch_size"]) * 4):
            batch = torch.from_numpy(x_val[start : start + int(config["batch_size"]) * 4]).to(device)
            delta_parts.append(model(batch).cpu().numpy())
    delta = np.concatenate(delta_parts).astype(np.float32)
    corrected = np.clip(base_val + delta, 0.0, 1.0) * capacity
    return corrected, {
        "train_loss": last_loss,
        "epochs": int(config["epochs"]),
        "n_train": int(keep.sum()),
        "n_features": int(x_train.shape[1]),
        "mean_abs_delta": float(np.mean(np.abs(delta))),
    }


def metric_row(actual: np.ndarray, prediction: np.ndarray, group: str) -> dict:
    nmae, ficr = group_nmae_ficr(actual, prediction, GROUP_CAPACITY_KWH[group])
    return {"score": 0.5 * (1.0 - nmae) + 0.5 * ficr, "nmae": nmae, "ficr": ficr}


def score_variants(oof: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mapping = {
        "pinn": "pinn_pred",
        "tree": "tree_pred",
        "tcn": "tcn_pred",
        "ensemble": "ensemble_pred",
        "ensemble_residual": "residual_pred",
    }
    long_parts = []
    fold_rows = []
    for variant, pred_col in mapping.items():
        part = oof[["forecast_kst_dtm", "pred_year", "group", "target", pred_col]].rename(
            columns={"target": "actual", pred_col: "pred"}
        )
        part["variant"] = variant
        long_parts.append(part)
        for (pred_year, group), fold in part.groupby(["pred_year", "group"]):
            fold_rows.append(
                {
                    "variant": variant,
                    "pred_year": int(pred_year),
                    "group": group,
                    **metric_row(fold["actual"].to_numpy(), fold["pred"].to_numpy(), group),
                    "n_rows": len(fold),
                }
            )
    long = pd.concat(long_parts, ignore_index=True)
    present_groups = set(long["group"].unique())
    if present_groups == set(TARGET_COLS):
        summary, group_scores = pooled_oof_summary(
            long,
            actual_col="actual",
            forecast_col="pred",
        )
    else:
        group_rows = []
        for (variant, group), part in long.groupby(["variant", "group"]):
            metrics = metric_row(part["actual"].to_numpy(), part["pred"].to_numpy(), group)
            group_rows.append(
                {"variant": variant, "group": group, **metrics, "n_rows": len(part)}
            )
        group_scores = pd.DataFrame(group_rows)
        summary = (
            group_scores.groupby("variant", as_index=False)
            .agg(
                mean_nmae=("nmae", "mean"),
                mean_ficr=("ficr", "mean"),
                worst_group_score=("score", "min"),
                std_group_score=("score", lambda values: values.std(ddof=0)),
                n_groups=("group", "nunique"),
            )
        )
        summary["mean_score"] = 0.5 * (1.0 - summary["mean_nmae"]) + 0.5 * summary["mean_ficr"]
    folds = pd.DataFrame(fold_rows)
    diagnostics = (
        folds.groupby("variant", as_index=False)
        .agg(worst_fold=("score", "min"), std_fold=("score", lambda values: values.std(ddof=0)))
    )
    summary = summary.merge(diagnostics, on="variant", how="left").sort_values(
        "mean_score", ascending=False
    )
    return summary.reset_index(drop=True), pd.concat(
        [folds.assign(row_type="fold"), group_scores.assign(pred_year=np.nan, row_type="pooled_group")],
        ignore_index=True,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config, args.smoke_test)
    groups = parse_csv(args.groups)
    unknown = [group for group in groups if group not in TARGET_COLS]
    if unknown:
        raise ValueError(f"Unknown groups: {unknown}")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} groups={groups} smoke={args.smoke_test}", flush=True)
    ldaps, gfs, labels, scada_by_group = read_inputs()

    oof_parts = []
    training_rows = []
    outer_count = 0
    for group_index, group in enumerate(groups):
        print(f"\n=== build unified group data: {group} ===", flush=True)
        tables = build_group_tables(ldaps, gfs, labels, scada_by_group[group], group)
        available = available_target_times(tables)
        print(
            f"{group}: base_features={len(GROUP_FAMILY_QUOTA65_V1_FEATURES[group]) + len(BASE_EXTRA_FEATURES)} "
            f"residual_features={len(tables.residual_cols)} years={sorted(available['year'].unique())}",
            flush=True,
        )
        for pred_year in YEARS:
            outer_train = available.loc[available["year"].ne(pred_year)].copy()
            outer_val = available.loc[available["year"].eq(pred_year)].copy()
            if len(outer_train) < 500 or len(outer_val) < 200:
                continue
            if args.max_outer_folds and outer_count >= args.max_outer_folds:
                break
            print(
                f"\n--- {group} outer={pred_year} train={len(outer_train)} val={len(outer_val)} ---",
                flush=True,
            )
            inner_parts = []
            for inner_index, (inner_train_times, inner_val_times, inner_name) in enumerate(
                inner_splits(outer_train)
            ):
                print(f"inner {inner_name}: train={len(inner_train_times)} val={len(inner_val_times)}", flush=True)
                inner_prediction, diagnostics = run_base_fold(
                    tables,
                    group,
                    inner_train_times,
                    inner_val_times,
                    config,
                    device,
                    int(config["seed"]) + group_index * 10000 + pred_year * 10 + inner_index,
                )
                inner_parts.append(inner_prediction)
                for row in diagnostics:
                    training_rows.append(
                        {
                            "group": group,
                            "pred_year": pred_year,
                            "stage": f"inner_{inner_name}",
                            **row,
                        }
                    )
            residual_train = pd.concat(inner_parts, ignore_index=True).sort_values(
                "forecast_kst_dtm"
            )
            if residual_train["forecast_kst_dtm"].duplicated().any():
                raise ValueError(f"Duplicate inner OOF rows for {group} outer={pred_year}")

            outer_prediction, diagnostics = run_base_fold(
                tables,
                group,
                outer_train["forecast_kst_dtm"].to_numpy(),
                outer_val["forecast_kst_dtm"].to_numpy(),
                config,
                device,
                int(config["seed"]) + group_index * 10000 + pred_year * 10 + 9,
            )
            for row in diagnostics:
                training_rows.append(
                    {"group": group, "pred_year": pred_year, "stage": "outer", **row}
                )
            residual_prediction, residual_stats = fit_predict_residual(
                residual_train,
                outer_prediction,
                tables,
                group,
                config["ensemble_weights"],
                config["residual"],
                device,
                int(config["seed"]) + group_index * 10000 + pred_year * 10 + 99,
            )
            training_rows.append(
                {"group": group, "pred_year": pred_year, "stage": "outer", "model": "residual", **residual_stats}
            )
            outer_prediction["ensemble_pred"] = np.clip(
                ensemble_prediction(outer_prediction, config["ensemble_weights"]),
                0.0,
                GROUP_CAPACITY_KWH[group],
            )
            outer_prediction["residual_pred"] = residual_prediction
            outer_prediction["pred_year"] = pred_year
            outer_prediction["group"] = group
            oof_parts.append(outer_prediction)
            before = metric_row(
                outer_prediction["target"].to_numpy(), outer_prediction["ensemble_pred"].to_numpy(), group
            )
            after = metric_row(
                outer_prediction["target"].to_numpy(), outer_prediction["residual_pred"].to_numpy(), group
            )
            print(
                f"outer={pred_year} ensemble={before['score']:.6f} residual={after['score']:.6f}",
                flush=True,
            )
            outer_count += 1
        if args.max_outer_folds and outer_count >= args.max_outer_folds:
            break

    if not oof_parts:
        raise RuntimeError("No OOF predictions were generated")
    oof = pd.concat(oof_parts, ignore_index=True).sort_values(
        ["group", "forecast_kst_dtm"]
    )
    summary, diagnostics = score_variants(oof)
    oof_path = args.results_dir / f"{args.stem}_oof.csv"
    summary_path = args.results_dir / f"{args.stem}_summary.csv"
    diagnostics_path = args.results_dir / f"{args.stem}_diagnostics.csv"
    oof.to_csv(oof_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.concat(
        [diagnostics, pd.DataFrame(training_rows).assign(row_type="training")],
        ignore_index=True,
    ).to_csv(diagnostics_path, index=False, encoding="utf-8-sig")
    print("\n=== pooled official OOF ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"saved {oof_path}", flush=True)
    print(f"saved {summary_path}", flush=True)
    print(f"saved {diagnostics_path}", flush=True)


if __name__ == "__main__":
    main()

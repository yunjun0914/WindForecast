from __future__ import annotations

import argparse
import copy
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import _bootstrap  # noqa: F401
from models.seqnn import TCNPowerRegressor
from utils.metrics import (
    GROUP_CAPACITY_KWH,
    TARGET_COLS,
    group_nmae_ficr,
    pooled_oof_summary,
)
from utils.per_turbine_features import get_or_build_group_feature_cache, tree_feature_columns
from utils.per_turbine_optimal_grid import (
    OPTIMAL_GRID_CUBIC_ISSUE_CONTEXT_TAG,
    OPTIMAL_GRID_ISSUE_CONTEXT_TAG,
    OPTIMAL_GRID_ISSUE_RELATIVE_TAG,
    OPTIMAL_GRID_REPLACE_TAG,
    load_optimal_grid_fold_features,
    optimal_grid_input_columns,
)
from utils.per_turbine_power_grid import (
    POWER_GRID_TEACHER_TAG,
    load_power_grid_fold_features,
    power_grid_input_columns,
)
from utils.per_turbine_scada import (
    apply_turbine_share_shrinkage,
    build_official_aligned_turbine_targets,
    build_static_turbine_share_priors,
    turbine_capacity_kwh,
)
from utils.per_turbine_sequence import SequenceStandardScaler, make_per_turbine_sequences
from utils.per_turbine_raw_source_features import (
    SOURCES,
    raw_source_input_columns,
)
from utils.per_turbine_teacher import TEACHER_FEATURE_COLS, get_or_build_teacher_cache
from utils.per_turbine_terrain import add_per_turbine_terrain_features, terrain_feature_columns
from utils.power_curve import GROUP_TURBINE_PREFIXES


YEARS = [2022, 2023, 2024]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--window", type=int, default=72)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=4096)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--target-min-output-ratio", type=float, default=0.10)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--teacher-trees", type=int, default=80)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--stem", default="per_turbine_tcn_w72_oof_v1")
    parser.add_argument(
        "--feature-variant",
        choices=[
            "baseline",
            "optimal_grid_replace_local16",
            "optimal_grid_issue_context",
            "optimal_grid_issue_relative",
            "cubic_grid_issue_context",
            "power_grid_pair12",
        ],
        default="baseline",
    )
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument("--rebuild-teacher-cache", action="store_true")
    parser.add_argument(
        "--terrain-variant", choices=["none", "directional2"], default="none"
    )
    parser.add_argument(
        "--terrain-turbine-cache",
        type=Path,
        default=Path("results/external_terrain_audit_v1_turbines.csv"),
    )
    parser.add_argument(
        "--terrain-sector-cache",
        type=Path,
        default=Path(
            "results/terrain_directional_residual_audit_v1_sector_features.csv"
        ),
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--interval-ratio", type=float, default=0.07)
    parser.add_argument("--interval-head-epochs", type=int, default=5)
    parser.add_argument("--interval-head-lr", type=float, default=1e-4)
    parser.add_argument("--interval-head-weight-decay", type=float, default=1e-4)
    parser.add_argument("--joint-finetune-epochs", type=int, default=0)
    parser.add_argument("--joint-finetune-lr", type=float, default=1e-4)
    parser.add_argument("--joint-finetune-weight-decay", type=float, default=1e-4)
    parser.add_argument("--joint-finetune-batch-size", type=int, default=64)
    parser.add_argument("--joint-anchor-weight", type=float, default=0.03)
    parser.add_argument("--joint-gamma", type=float, default=0.00682273)
    parser.add_argument(
        "--input-ablation",
        choices=["all", "weather_only", "teacher_only"],
        default="all",
    )
    parser.add_argument("--skip-interval-head", action="store_true")
    parser.add_argument(
        "--weather-source",
        choices=["mixed", *SOURCES],
        default="mixed",
    )
    parser.add_argument("--drop-highwind-lowpower", action="store_true")
    parser.add_argument("--drop-wind-col", default="optgrid_ws_calibrated")
    parser.add_argument("--drop-wind-threshold", type=float, default=8.0)
    parser.add_argument("--drop-power-ratio-threshold", type=float, default=0.10)
    parser.add_argument("--target-share-alpha", type=float, default=1.0)
    return parser.parse_args()


def parse_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def aligned_row_values(table: pd.DataFrame, column: str) -> np.ndarray:
    if column not in table.columns:
        raise ValueError(f"Missing column for row-value alignment: {column}")
    work = table.sort_values("forecast_kst_dtm").copy()
    work["forecast_kst_dtm"] = pd.to_datetime(work["forecast_kst_dtm"])
    parts = []
    for _, year_df in work.groupby(work["forecast_kst_dtm"].dt.year, sort=True):
        parts.append(pd.to_numeric(year_df[column], errors="coerce").to_numpy(np.float32))
    return np.concatenate(parts) if parts else np.empty((0,), dtype=np.float32)


def turbine_metric(
    actual: np.ndarray,
    pred: np.ndarray,
    capacity: float,
) -> tuple[float, float, float]:
    actual = np.asarray(actual, dtype=float)
    pred = np.clip(np.asarray(pred, dtype=float), 0, capacity)
    error_rate = np.abs(pred - actual) / capacity
    nmae = float(error_rate.mean())
    unit_price = np.select(
        [error_rate <= 0.06, error_rate <= 0.08],
        [4.0, 3.0],
        default=0.0,
    )
    denominator = float(np.sum(actual * 4.0))
    ficr = float(np.sum(actual * unit_price) / denominator) if denominator > 0 else 0.0
    return 0.5 * (1.0 - nmae) + 0.5 * ficr, nmae, ficr


def group_metric(actual: np.ndarray, pred: np.ndarray, group: str) -> tuple[float, float, float]:
    capacity = GROUP_CAPACITY_KWH[group]
    pred = np.clip(np.asarray(pred, dtype=float), 0, capacity)
    nmae, ficr = group_nmae_ficr(actual, pred, capacity)
    return 0.5 * (1.0 - nmae) + 0.5 * ficr, nmae, ficr


def predict_normalized(
    model: torch.nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            parts.append(model(xb).detach().cpu().numpy())
    return np.concatenate(parts) if parts else np.empty((0,), dtype=np.float32)


def encode_hidden(
    model: TCNPowerRegressor,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.from_numpy(x[start : start + batch_size]).to(device)
            encoded = model.network(xb.transpose(1, 2))[:, :, -1]
            parts.append(encoded.cpu().numpy())
    return np.concatenate(parts) if parts else np.empty((0, 0), dtype=np.float32)


def train_turbine_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    official_train: np.ndarray,
    wind_train: np.ndarray | None,
    x_val_all: np.ndarray,
    y_val_all: np.ndarray,
    official_val: np.ndarray,
    time_train: pd.Series,
    time_val: pd.Series,
    group: str,
    turbine: str,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict[str, object], dict[str, object]]:
    set_seed(seed)
    turbine_capacity = turbine_capacity_kwh(group)
    group_capacity = GROUP_CAPACITY_KWH[group]
    train_keep = (
        np.isfinite(y_train)
        & np.isfinite(official_train)
        & (official_train >= group_capacity * args.target_min_output_ratio)
    )
    raw_train_keep = train_keep.copy()
    n_drop_highwind_lowpower = 0
    if args.drop_highwind_lowpower:
        if wind_train is None:
            raise ValueError("--drop-highwind-lowpower requires wind_train values")
        if len(wind_train) != len(y_train):
            raise ValueError(
                f"wind_train length mismatch for {turbine}: {len(wind_train)} != {len(y_train)}"
            )
        power_ratio = y_train / turbine_capacity
        highwind_lowpower = (
            np.isfinite(wind_train)
            & np.isfinite(power_ratio)
            & (wind_train >= args.drop_wind_threshold)
            & (power_ratio <= args.drop_power_ratio_threshold)
        )
        n_drop_highwind_lowpower = int((raw_train_keep & highwind_lowpower).sum())
        train_keep = train_keep & ~highwind_lowpower
    val_eval_keep = (
        np.isfinite(y_val_all)
        & np.isfinite(official_val)
        & (official_val >= group_capacity * args.target_min_output_ratio)
    )
    if int(train_keep.sum()) < 500 or int(val_eval_keep.sum()) < 200:
        raise ValueError(
            f"Insufficient TCN targets {turbine}: train={int(train_keep.sum())} "
            f"val={int(val_eval_keep.sum())}"
        )

    x_train = x_train[train_keep]
    y_train = y_train[train_keep]
    official_train = official_train[train_keep]
    scaler = SequenceStandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val_all)
    del x_train, x_val_all
    gc.collect()

    y_train_norm = np.clip(y_train / turbine_capacity, 0, 1).astype(np.float32)
    weights = (
        0.5 + np.sqrt(np.clip(official_train / group_capacity, 0, 1))
    ).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(x_train_scaled),
        torch.from_numpy(y_train_norm),
        torch.from_numpy(weights),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    model = TCNPowerRegressor(
        input_size=x_train_scaled.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -np.inf
    best_epoch = -1
    best_state = None
    bad_epochs = 0
    val_eval_indices = np.flatnonzero(val_eval_keep)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb, wb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = (torch.abs(pred - yb) * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        pred_eval_norm = predict_normalized(
            model,
            x_val_scaled[val_eval_indices],
            device,
            args.eval_batch_size,
        )
        pred_eval = np.clip(pred_eval_norm, 0, 1) * turbine_capacity
        score, nmae, ficr = turbine_metric(
            y_val_all[val_eval_keep],
            pred_eval,
            turbine_capacity,
        )
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    best_eval_norm = predict_normalized(
        model,
        x_val_scaled[val_eval_indices],
        device,
        args.eval_batch_size,
    )
    best_eval = np.clip(best_eval_norm, 0, 1) * turbine_capacity
    best_score, best_nmae, best_ficr = turbine_metric(
        y_val_all[val_eval_keep],
        best_eval,
        turbine_capacity,
    )
    pred_all_norm = predict_normalized(model, x_val_scaled, device, args.eval_batch_size)
    pred_all = np.clip(pred_all_norm, 0, 1) * turbine_capacity
    train_pretrained_norm = predict_normalized(
        model, x_train_scaled, device, args.eval_batch_size
    ).astype(np.float32)
    artifact = {
        "turbine_id": turbine,
        "head": copy.deepcopy(model.head).cpu(),
        "model_state": {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        },
        "scaler_mean": scaler.mean_.copy(),
        "scaler_std": scaler.std_.copy(),
        "train_time": pd.to_datetime(np.asarray(time_train)[train_keep]),
        "train_pretrained": train_pretrained_norm,
        "train_hidden": encode_hidden(model, x_train_scaled, device, args.eval_batch_size),
        "train_official": official_train.astype(np.float32, copy=True),
        "val_time": pd.to_datetime(np.asarray(time_val)),
        "val_hidden": encode_hidden(model, x_val_scaled, device, args.eval_batch_size),
        "val_official": official_val.astype(np.float32, copy=True),
    }
    stats = {
        "group": group,
        "turbine_id": turbine,
        "best_epoch": best_epoch,
        "turbine_val_score": best_score,
        "turbine_val_nmae": best_nmae,
        "turbine_val_ficr": best_ficr,
        "n_train": len(y_train),
        "n_train_before_drop": int(raw_train_keep.sum()),
        "n_drop_highwind_lowpower": n_drop_highwind_lowpower,
        "drop_wind_threshold": args.drop_wind_threshold if args.drop_highwind_lowpower else np.nan,
        "drop_power_ratio_threshold": (
            args.drop_power_ratio_threshold if args.drop_highwind_lowpower else np.nan
        ),
        "n_val_all": len(y_val_all),
        "n_val_target": int(val_eval_keep.sum()),
    }
    del model, optimizer, loader, dataset, x_train_scaled, x_val_scaled
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pred_all, stats, artifact


class GroupIntervalHeads(torch.nn.Module):
    def __init__(self, heads: list[torch.nn.Module], turbine_ratio: float) -> None:
        super().__init__()
        self.heads = torch.nn.ModuleList(heads)
        self.turbine_ratio = float(turbine_ratio)

    def forward(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        turbine_prediction = torch.stack(
            [
                head(hidden[:, index, :]).squeeze(-1)
                for index, head in enumerate(self.heads)
            ],
            dim=1,
        )
        turbine_prediction = torch.clamp(turbine_prediction, 0.0, 1.0)
        group_prediction = turbine_prediction.sum(dim=1) * self.turbine_ratio
        return group_prediction, turbine_prediction


def align_head_artifacts(
    artifacts: list[dict[str, object]],
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common = pd.Index(artifacts[0][f"{split}_time"])
    for artifact in artifacts[1:]:
        common = common.intersection(pd.Index(artifact[f"{split}_time"]), sort=False)
    common = common.sort_values()
    hidden_parts = []
    official_parts = []
    for artifact in artifacts:
        times = pd.Index(artifact[f"{split}_time"])
        positions = times.get_indexer(common)
        if np.any(positions < 0):
            raise ValueError(f"Failed to align {split} hidden rows")
        hidden_parts.append(np.asarray(artifact[f"{split}_hidden"])[positions])
        official_parts.append(np.asarray(artifact[f"{split}_official"])[positions])
    official = official_parts[0]
    for other in official_parts[1:]:
        if not np.allclose(official, other, equal_nan=True):
            raise ValueError(f"Official target mismatch in {split} artifacts")
    return common.to_numpy(), np.stack(hidden_parts, axis=1), official


def fine_tune_interval_heads(
    artifacts: list[dict[str, object]],
    group: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, dict[str, object]]:
    train_time, train_hidden, train_official = align_head_artifacts(artifacts, "train")
    val_time, val_hidden, _ = align_head_artifacts(artifacts, "val")
    group_capacity = float(GROUP_CAPACITY_KWH[group])
    turbine_ratio = float(turbine_capacity_kwh(group)) / group_capacity
    model = GroupIntervalHeads(
        [copy.deepcopy(artifact["head"]) for artifact in artifacts],
        turbine_ratio,
    ).to(device)
    train_target = np.clip(train_official / group_capacity, 0.0, 1.0).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(train_hidden.astype(np.float32, copy=False)),
        torch.from_numpy(train_target),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.interval_head_lr,
        weight_decay=args.interval_head_weight_decay,
    )
    history = []
    for epoch in range(1, args.interval_head_epochs + 1):
        model.train()
        losses = []
        band_rates = []
        for hidden, target in loader:
            hidden = hidden.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction, _ = model(hidden)
            error = torch.abs(prediction - target)
            loss = torch.relu(error - args.interval_ratio).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            band_rates.append(float((error <= args.interval_ratio).float().mean().cpu()))
        history.append(
            {
                "group": group,
                "epoch": epoch,
                "train_interval_loss": float(np.mean(losses)),
                "train_band_rate": float(np.mean(band_rates)),
                "n_train": len(train_time),
            }
        )

    model.eval()
    prediction_parts = []
    with torch.no_grad():
        for start in range(0, len(val_hidden), args.eval_batch_size):
            hidden = torch.from_numpy(
                val_hidden[start : start + args.eval_batch_size].astype(
                    np.float32, copy=False
                )
            ).to(device)
            _, turbine_prediction = model(hidden)
            prediction_parts.append(turbine_prediction.cpu().numpy())
    prediction = np.concatenate(prediction_parts, axis=0)
    rows = []
    turbine_capacity = turbine_capacity_kwh(group)
    for turbine_index, artifact in enumerate(artifacts):
        rows.append(
            pd.DataFrame(
                {
                    "forecast_kst_dtm": pd.to_datetime(val_time),
                    "group": group,
                    "turbine_id": artifact["turbine_id"],
                    "pred": np.clip(prediction[:, turbine_index], 0.0, 1.0)
                    * turbine_capacity,
                }
            )
        )
    del model, optimizer, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.concat(rows, ignore_index=True), history[-1]


def scale_with_artifact(
    values: np.ndarray, artifact: dict[str, object]
) -> np.ndarray:
    mean = np.asarray(artifact["scaler_mean"], dtype=np.float32)
    std = np.asarray(artifact["scaler_std"], dtype=np.float32)
    scaled = (values - mean) / std
    return np.nan_to_num(
        scaled, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32)


def common_artifact_times(
    artifacts: list[dict[str, object]], split: str
) -> pd.DatetimeIndex:
    common = pd.DatetimeIndex(artifacts[0][f"{split}_time"])
    for artifact in artifacts[1:]:
        common = common.intersection(
            pd.DatetimeIndex(artifact[f"{split}_time"]), sort=False
        )
    return common.sort_values()


def aligned_artifact_values(
    artifact: dict[str, object], split: str, field: str, common: pd.DatetimeIndex
) -> np.ndarray:
    times = pd.DatetimeIndex(artifact[f"{split}_time"])
    positions = times.get_indexer(common)
    if np.any(positions < 0):
        raise ValueError(f"Failed to align {split} {field} for {artifact['turbine_id']}")
    return np.asarray(artifact[field])[positions]


def soft_group_score_loss(
    target: torch.Tensor, prediction: torch.Tensor, gamma: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    error = torch.abs(prediction - target)
    nmae = error.mean()
    unit_price = 4.0 - torch.sigmoid((error - 0.06) / gamma) - 3.0 * torch.sigmoid(
        (error - 0.08) / gamma
    )
    ficr = (target * unit_price).sum() / torch.clamp((target * 4.0).sum(), min=1e-6)
    return 0.5 * nmae + 0.5 * (1.0 - ficr), nmae, ficr


def fine_tune_group_backbones(
    table: pd.DataFrame,
    artifacts: list[dict[str, object]],
    feature_cols: list[str],
    group: str,
    train_years: list[int],
    pred_year: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    train_time = common_artifact_times(artifacts, "train")
    val_time = common_artifact_times(artifacts, "val")
    train_inputs = []
    val_inputs = []
    pretrained_parts = []
    models = []

    train_official = aligned_artifact_values(
        artifacts[0], "train", "train_official", train_time
    ).astype(np.float32)
    for artifact in artifacts[1:]:
        other = aligned_artifact_values(
            artifact, "train", "train_official", train_time
        )
        if not np.allclose(train_official, other, equal_nan=True):
            raise ValueError(f"Official group target mismatch for {group} pred_year={pred_year}")

    for artifact in artifacts:
        turbine = str(artifact["turbine_id"])
        turbine_table = table.loc[table["turbine_id"].eq(turbine)]
        train_table = turbine_table.loc[
            turbine_table["forecast_kst_dtm"].dt.year.isin(train_years)
        ]
        val_table = turbine_table.loc[
            turbine_table["forecast_kst_dtm"].dt.year.eq(pred_year)
        ]
        x_train_all, _, _, x_train_time, _ = make_per_turbine_sequences(
            train_table, feature_cols, window=args.window
        )
        x_val_all, _, _, x_val_time, _ = make_per_turbine_sequences(
            val_table, feature_cols, window=args.window
        )
        train_positions = pd.DatetimeIndex(x_train_time).get_indexer(train_time)
        val_positions = pd.DatetimeIndex(x_val_time).get_indexer(val_time)
        if np.any(train_positions < 0) or np.any(val_positions < 0):
            raise ValueError(f"Sequence alignment failed for {group}/{turbine}/{pred_year}")
        train_inputs.append(
            scale_with_artifact(x_train_all[train_positions], artifact)
        )
        val_inputs.append(scale_with_artifact(x_val_all[val_positions], artifact))
        pretrained_parts.append(
            aligned_artifact_values(
                artifact, "train", "train_pretrained", train_time
            ).astype(np.float32)
        )
        model = TCNPowerRegressor(
            input_size=len(feature_cols),
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        ).to(device)
        model.load_state_dict(artifact["model_state"])
        models.append(model)

    group_capacity = float(GROUP_CAPACITY_KWH[group])
    turbine_capacity = float(turbine_capacity_kwh(group))
    turbine_ratio = turbine_capacity / group_capacity
    target = np.clip(train_official / group_capacity, 0.0, 1.0).astype(np.float32)
    pretrained = np.stack(pretrained_parts, axis=1)
    parameters = [parameter for model in models for parameter in model.parameters()]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.joint_finetune_lr,
        weight_decay=args.joint_finetune_weight_decay,
    )
    history = []
    generator = np.random.default_rng(args.seed + pred_year * 1000 + len(models))
    for epoch in range(1, args.joint_finetune_epochs + 1):
        for model in models:
            model.train()
        epoch_losses = []
        epoch_score_losses = []
        epoch_anchors = []
        indices = generator.permutation(len(train_time))
        for start in range(0, len(indices), args.joint_finetune_batch_size):
            batch_indices = indices[start : start + args.joint_finetune_batch_size]
            target_batch = torch.from_numpy(target[batch_indices]).to(device)
            pretrained_batch = torch.from_numpy(pretrained[batch_indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            turbine_prediction = torch.stack(
                [
                    torch.clamp(
                        model(torch.from_numpy(values[batch_indices]).to(device)),
                        0.0,
                        1.0,
                    )
                    for model, values in zip(models, train_inputs)
                ],
                dim=1,
            )
            group_prediction = turbine_prediction.sum(dim=1) * turbine_ratio
            score_loss, _, _ = soft_group_score_loss(
                target_batch, group_prediction, args.joint_gamma
            )
            anchor = (turbine_prediction - pretrained_batch).pow(2).mean()
            loss = score_loss + args.joint_anchor_weight * anchor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_score_losses.append(float(score_loss.detach().cpu()))
            epoch_anchors.append(float(anchor.detach().cpu()))
        history.append(
            {
                "group": group,
                "pred_year": pred_year,
                "epoch": epoch,
                "loss": float(np.mean(epoch_losses)),
                "score_loss": float(np.mean(epoch_score_losses)),
                "anchor_mse": float(np.mean(epoch_anchors)),
                "n_train": len(train_time),
                "n_turbines": len(models),
            }
        )

    prediction_parts = [[] for _ in models]
    for model in models:
        model.eval()
    with torch.no_grad():
        for start in range(0, len(val_time), args.eval_batch_size):
            stop = start + args.eval_batch_size
            for turbine_index, (model, values) in enumerate(zip(models, val_inputs)):
                batch = torch.from_numpy(values[start:stop]).to(device)
                prediction_parts[turbine_index].append(
                    torch.clamp(model(batch), 0.0, 1.0).cpu().numpy()
                )

    rows = []
    for artifact, parts in zip(artifacts, prediction_parts):
        prediction = np.concatenate(parts)
        rows.append(
            pd.DataFrame(
                {
                    "forecast_kst_dtm": pd.to_datetime(val_time),
                    "group": group,
                    "turbine_id": artifact["turbine_id"],
                    "pred": prediction * turbine_capacity,
                }
            )
        )
    del models, optimizer, train_inputs, val_inputs, pretrained
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.concat(rows, ignore_index=True), history


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.target_share_alpha <= 1.0:
        raise ValueError("--target-share-alpha must be in [0, 1]")
    if args.joint_finetune_epochs > 0 and not args.skip_interval_head:
        raise ValueError("Joint backbone fine-tune requires --skip-interval-head")
    if args.weather_source != "mixed":
        if args.feature_variant != "baseline":
            raise ValueError("Raw source experts require --feature-variant baseline")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} window={args.window} feature_variant={args.feature_variant} "
        f"weather_source={args.weather_source} epochs={args.epochs} patience={args.patience}",
        flush=True,
    )

    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    scada_vestas = pd.read_csv("data/train/scada_vestas_train.csv", encoding="utf-8-sig")
    scada_unison = pd.read_csv("data/train/scada_unison_train.csv", encoding="utf-8-sig")
    scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }

    score_rows = []
    training_rows = []
    interval_training_rows = []
    joint_training_rows = []
    turbine_prediction_parts = []
    baseline_turbine_prediction_parts = []
    group_prediction_parts = []
    model_counter = 0

    for group in parse_list(args.groups):
        base_features = get_or_build_group_feature_cache(
            ldaps,
            gfs,
            group,
            cache_root=args.cache_root,
            rebuild=args.rebuild_feature_cache,
        )
        targets = build_official_aligned_turbine_targets(scada_by_group[group], labels, group)
        label_one = labels[["kst_dtm", group]].rename(
            columns={"kst_dtm": "forecast_kst_dtm", group: "official_target"}
        )
        label_one["forecast_kst_dtm"] = pd.to_datetime(label_one["forecast_kst_dtm"])

        for pred_year in YEARS:
            if labels.loc[labels["kst_dtm"].dt.year.eq(pred_year), group].notna().sum() < 200:
                continue
            train_years = [year for year in YEARS if year != pred_year]
            static_shares = build_static_turbine_share_priors(
                targets,
                group,
                train_years,
                target_min_output_ratio=args.target_min_output_ratio,
            )
            fold_targets = apply_turbine_share_shrinkage(
                targets,
                static_shares,
                dynamic_weight=args.target_share_alpha,
            )
            target_one = fold_targets[
                ["forecast_kst_dtm", "turbine_id", "turbine_target"]
            ].copy()
            if args.weather_source != "mixed":
                features = base_features
                teacher_input_cols = raw_source_input_columns(group, args.weather_source)
                teacher_cache_tag = f"raw_{args.weather_source}"
            elif args.feature_variant in {
                "optimal_grid_replace_local16",
                "optimal_grid_issue_context",
                "optimal_grid_issue_relative",
                "cubic_grid_issue_context",
            }:
                include_issue_context = args.feature_variant != "optimal_grid_replace_local16"
                include_issue_relative = (
                    args.feature_variant == "optimal_grid_issue_relative"
                )
                wind_target = (
                    "cubic" if args.feature_variant == "cubic_grid_issue_context" else "mean"
                )
                features = load_optimal_grid_fold_features(
                    base_features,
                    args.cache_root,
                    group,
                    pred_year,
                    wind_target=wind_target,
                    include_issue_context=include_issue_context,
                    include_issue_relative=include_issue_relative,
                )
                teacher_input_cols = optimal_grid_input_columns(
                    group,
                    include_issue_context=include_issue_context,
                    include_issue_relative=include_issue_relative,
                )
                if args.feature_variant == "optimal_grid_issue_context":
                    teacher_cache_tag = OPTIMAL_GRID_ISSUE_CONTEXT_TAG
                elif args.feature_variant == "optimal_grid_issue_relative":
                    teacher_cache_tag = OPTIMAL_GRID_ISSUE_RELATIVE_TAG
                elif args.feature_variant == "cubic_grid_issue_context":
                    teacher_cache_tag = OPTIMAL_GRID_CUBIC_ISSUE_CONTEXT_TAG
                else:
                    teacher_cache_tag = OPTIMAL_GRID_REPLACE_TAG
            elif args.feature_variant == "power_grid_pair12":
                features = load_power_grid_fold_features(
                    base_features, args.cache_root, group, pred_year
                )
                teacher_input_cols = power_grid_input_columns(group)
                teacher_cache_tag = POWER_GRID_TEACHER_TAG
            else:
                features = base_features
                teacher_input_cols = tree_feature_columns(group)
                teacher_cache_tag = None
            teacher = None
            if args.input_ablation != "weather_only":
                teacher = get_or_build_teacher_cache(
                    features=features,
                    targets=targets,
                    scada=scada_by_group[group],
                    group=group,
                    train_years=train_years,
                    pred_year=pred_year,
                    cache_root=args.cache_root,
                    rebuild=args.rebuild_teacher_cache,
                    n_estimators=args.teacher_trees,
                    input_feature_cols=teacher_input_cols,
                    cache_tag=teacher_cache_tag,
                )
            features = add_per_turbine_terrain_features(
                features,
                args.terrain_variant,
                turbine_cache=args.terrain_turbine_cache,
                sector_cache=args.terrain_sector_cache,
            )
            keys = ["forecast_kst_dtm", "turbine_id"]
            if teacher is None:
                table = features.copy()
            else:
                table = features.merge(
                    teacher[keys + TEACHER_FEATURE_COLS], on=keys, how="inner"
                )
            table = table.merge(target_one, on=keys, how="left")
            table = table.merge(label_one, on="forecast_kst_dtm", how="left")
            table["forecast_kst_dtm"] = pd.to_datetime(table["forecast_kst_dtm"])
            weather_cols = [
                *teacher_input_cols,
                *terrain_feature_columns(args.terrain_variant),
            ]
            if args.input_ablation == "weather_only":
                feature_cols = weather_cols
            elif args.input_ablation == "teacher_only":
                feature_cols = list(TEACHER_FEATURE_COLS)
            else:
                feature_cols = [*weather_cols, *TEACHER_FEATURE_COLS]
            print(
                f"\n{group} pred_year={pred_year} train={train_years} "
                f"variant={args.feature_variant} source={args.weather_source} "
                f"features={len(feature_cols)} share_alpha={args.target_share_alpha:.2f}",
                flush=True,
            )

            fold_turbine_parts = []
            fold_artifacts = []
            for turbine_index, turbine in enumerate(GROUP_TURBINE_PREFIXES[group]):
                turbine_table = table.loc[table["turbine_id"].eq(turbine)].copy()
                train_table = turbine_table.loc[
                    turbine_table["forecast_kst_dtm"].dt.year.isin(train_years)
                ]
                val_table = turbine_table.loc[
                    turbine_table["forecast_kst_dtm"].dt.year.eq(pred_year)
                ]
                x_train, y_train, official_train, time_train, _ = make_per_turbine_sequences(
                    train_table,
                    feature_cols,
                    window=args.window,
                )
                wind_train = (
                    aligned_row_values(train_table, args.drop_wind_col)
                    if args.drop_highwind_lowpower
                    else None
                )
                x_val, y_val, official_val, time_val, _ = make_per_turbine_sequences(
                    val_table,
                    feature_cols,
                    window=args.window,
                )
                pred, stats, artifact = train_turbine_fold(
                    x_train=x_train,
                    y_train=y_train,
                    official_train=official_train,
                    wind_train=wind_train,
                    x_val_all=x_val,
                    y_val_all=y_val,
                    official_val=official_val,
                    time_train=time_train,
                    time_val=time_val,
                    group=group,
                    turbine=turbine,
                    args=args,
                    device=device,
                    seed=args.seed + pred_year * 100 + turbine_index,
                )
                model_counter += 1
                stats.update(
                    {
                        "pred_year": pred_year,
                        "train_years": ",".join(map(str, train_years)),
                        "window": args.window,
                        "n_features": len(feature_cols),
                        "weather_source": args.weather_source,
                        "target_share_alpha": args.target_share_alpha,
                        "static_share": float(static_shares.loc[turbine]),
                    }
                )
                training_rows.append(stats)
                fold_artifacts.append(artifact)
                turbine_pred = pd.DataFrame(
                    {
                        "forecast_kst_dtm": pd.to_datetime(time_val),
                        "group": group,
                        "turbine_id": turbine,
                        "pred_year": pred_year,
                        "pred": pred,
                    }
                )
                fold_turbine_parts.append(turbine_pred)
                baseline_turbine_prediction_parts.append(turbine_pred)
                drop_text = (
                    f" drop={stats['n_drop_highwind_lowpower']}/"
                    f"{stats['n_train_before_drop']}"
                    if args.drop_highwind_lowpower
                    else ""
                )
                print(
                    f"  {turbine}: epoch={stats['best_epoch']} "
                    f"turbine_score={stats['turbine_val_score']:.5f}{drop_text}",
                    flush=True,
                )
            baseline_turbines = pd.concat(fold_turbine_parts, ignore_index=True)
            if args.joint_finetune_epochs > 0:
                joint_turbines, joint_history = fine_tune_group_backbones(
                    table=table,
                    artifacts=fold_artifacts,
                    feature_cols=feature_cols,
                    group=group,
                    train_years=train_years,
                    pred_year=pred_year,
                    args=args,
                    device=device,
                )
                joint_turbines["pred_year"] = pred_year
                turbine_prediction_parts.append(joint_turbines)
                joint_training_rows.extend(joint_history)
                variants = {
                    "baseline_retrained": baseline_turbines,
                    "group_joint_score": joint_turbines,
                }
            elif args.skip_interval_head:
                turbine_prediction_parts.append(baseline_turbines)
                variants = {"baseline_retrained": baseline_turbines}
            else:
                interval_turbines, interval_stats = fine_tune_interval_heads(
                    fold_artifacts, group, args, device
                )
                interval_turbines["pred_year"] = pred_year
                turbine_prediction_parts.append(interval_turbines)
                interval_stats.update(
                    {
                        "pred_year": pred_year,
                        "train_years": ",".join(map(str, train_years)),
                        "interval_ratio": args.interval_ratio,
                        "head_lr": args.interval_head_lr,
                    }
                )
                interval_training_rows.append(interval_stats)
                variants = {
                    "baseline_retrained": baseline_turbines,
                    "interval7_head": interval_turbines,
                }
            expected_turbines = len(GROUP_TURBINE_PREFIXES[group])
            for variant, fold_turbines in variants.items():
                group_pred = (
                    fold_turbines.groupby("forecast_kst_dtm", as_index=False)
                    .agg(pred=("pred", "sum"), n_turbines=("turbine_id", "nunique"))
                    .merge(label_one, on="forecast_kst_dtm", how="inner")
                    .dropna(subset=["official_target"])
                )
                if not group_pred["n_turbines"].eq(expected_turbines).all():
                    raise ValueError(
                        f"Incomplete turbine sum for {group} pred_year={pred_year} {variant}"
                    )
                group_pred["pred"] = group_pred["pred"].clip(
                    0, GROUP_CAPACITY_KWH[group]
                )
                score, nmae, ficr = group_metric(
                    group_pred["official_target"].to_numpy(float),
                    group_pred["pred"].to_numpy(float),
                    group,
                )
                score_rows.append(
                    {
                        "variant": variant,
                        "group": group,
                        "pred_year": pred_year,
                        "train_years": ",".join(map(str, train_years)),
                        "score": score,
                        "nmae": nmae,
                        "ficr": ficr,
                        "n_rows": len(group_pred),
                        "n_features": len(feature_cols),
                        "window": args.window,
                        "weather_source": args.weather_source,
                        "target_share_alpha": args.target_share_alpha,
                    }
                )
                group_pred["group"] = group
                group_pred["pred_year"] = pred_year
                group_pred["variant"] = variant
                group_prediction_parts.append(group_pred)
                print(
                    f"{group} {pred_year} {variant}: group_score={score:.6f} "
                    f"nMAE={nmae:.6f} FiCR={ficr:.6f}",
                    flush=True,
                )
            del fold_artifacts
            gc.collect()
            if args.smoke_test:
                print("smoke test complete after one group-fold", flush=True)
                return

    scores = pd.DataFrame(score_rows)
    group_predictions = pd.concat(group_prediction_parts, ignore_index=True)
    summary, pooled_group_scores = pooled_oof_summary(group_predictions)
    fold_means = (
        scores.groupby(["variant", "pred_year"], as_index=False)
        .agg(score=("score", "mean"), nmae=("nmae", "mean"), ficr=("ficr", "mean"))
    )
    fold_diagnostics = (
        fold_means.groupby("variant", as_index=False)
        .agg(
            worst_fold=("score", "min"),
            std_score=("score", lambda values: values.std(ddof=0)),
            n_folds=("score", "count"),
        )
    )
    summary = summary.merge(fold_diagnostics, on="variant", how="left")
    summary = summary.sort_values("mean_score", ascending=False).reset_index(drop=True)
    summary["model"] = args.stem
    summary["n_models"] = model_counter
    summary["n_features"] = scores["n_features"].max()
    summary["window"] = args.window
    pd.DataFrame(training_rows).to_csv(
        args.results_dir / f"{args.stem}_training.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(interval_training_rows).to_csv(
        args.results_dir / f"{args.stem}_interval_training.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(joint_training_rows).to_csv(
        args.results_dir / f"{args.stem}_joint_training.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scores.to_csv(args.results_dir / f"{args.stem}_scores.csv", index=False, encoding="utf-8-sig")
    pooled_group_scores.to_csv(
        args.results_dir / f"{args.stem}_pooled_group_scores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(args.results_dir / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")
    group_predictions.to_csv(
        args.results_dir / f"{args.stem}_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(turbine_prediction_parts, ignore_index=True).to_csv(
        args.results_dir / f"{args.stem}_turbine_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(baseline_turbine_prediction_parts, ignore_index=True).to_csv(
        args.results_dir / f"{args.stem}_baseline_turbine_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\n=== summary ===", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

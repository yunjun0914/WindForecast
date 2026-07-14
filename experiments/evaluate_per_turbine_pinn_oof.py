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
from models.per_turbine_pinn import PerTurbineResidualPINN, PerTurbineResidualTCNPINN
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr
from utils.per_turbine_features import get_or_build_group_feature_cache, tree_feature_columns
from utils.per_turbine_optimal_grid import (
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
from utils.per_turbine_sequence import make_per_turbine_sequences
from utils.per_turbine_teacher import TEACHER_FEATURE_COLS, get_or_build_teacher_cache
from utils.per_turbine_terrain import add_per_turbine_terrain_features, terrain_feature_columns
from utils.pinn_scada_teacher_config import BEST_SCADA_TEACHER_PINN_GAMMA
from utils.power_curve import GROUP_TURBINE_PREFIXES


YEARS = [2022, 2023, 2024]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--backbone", choices=["mlp", "tcn"], default="mlp")
    parser.add_argument("--window", type=int, default=72)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--residual-amplitude", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--target-min-output-ratio", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=BEST_SCADA_TEACHER_PINN_GAMMA)
    parser.add_argument("--lambda-anchor", type=float, default=0.03)
    parser.add_argument("--lambda-residual", type=float, default=0.001)
    parser.add_argument("--lambda-bias", type=float, default=0.001)
    parser.add_argument("--lambda-scale", type=float, default=0.01)
    parser.add_argument("--min-delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--teacher-trees", type=int, default=80)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--stem", default="per_turbine_pinn_oof_v1")
    parser.add_argument(
        "--feature-variant",
        choices=["baseline", "optimal_grid_replace_local16", "power_grid_pair12"],
        default="baseline",
    )
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument("--rebuild-teacher-cache", action="store_true")
    parser.add_argument("--joint-finetune-epochs", type=int, default=0)
    parser.add_argument("--joint-finetune-lr", type=float, default=1e-4)
    parser.add_argument("--joint-finetune-weight-decay", type=float, default=1e-4)
    parser.add_argument("--joint-finetune-batch-size", type=int, default=64)
    parser.add_argument("--joint-anchor-weight", type=float, default=0.03)
    parser.add_argument("--joint-gamma", type=float, default=BEST_SCADA_TEACHER_PINN_GAMMA)
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
    parser.add_argument("--target-share-alpha", type=float, default=1.0)
    return parser.parse_args()


def parse_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_unit_price(error_rate: torch.Tensor, gamma: float) -> torch.Tensor:
    return 4.0 - torch.sigmoid((error_rate - 0.06) / gamma) - 3.0 * torch.sigmoid(
        (error_rate - 0.08) / gamma
    )


def turbine_metric(actual: np.ndarray, pred: np.ndarray, capacity: float) -> tuple[float, float, float]:
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


def standardize(
    x_train: np.ndarray,
    x_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    reduce_axes = tuple(range(x_train.ndim - 1))
    mean = np.nanmean(x_train, axis=reduce_axes).astype(np.float32)
    std = np.nanstd(x_train, axis=reduce_axes).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std)
    train = np.nan_to_num((x_train - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    val = np.nan_to_num((x_val - mean) / std, nan=0.0, posinf=0.0, neginf=0.0)
    return train.astype(np.float32), val.astype(np.float32), mean, std


def predict_model(
    model: PerTurbineResidualPINN,
    features: np.ndarray,
    physical_norm: np.ndarray,
    lead_hour: np.ndarray,
    month_index: np.ndarray,
    device: torch.device,
    batch_size: int,
    return_parts: bool = False,
):
    model.eval()
    pred_parts = []
    physical_parts = []
    calendar_parts = []
    residual_parts = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            stop = start + batch_size
            xb = torch.from_numpy(features[start:stop]).to(device)
            pb = torch.from_numpy(physical_norm[start:stop]).to(device)
            lb = torch.from_numpy(lead_hour[start:stop]).to(device)
            mb = torch.from_numpy(month_index[start:stop]).to(device)
            if return_parts:
                pred, parts = model(xb, pb, lb, mb, return_parts=True)
                physical_parts.append(parts["physical"].cpu().numpy())
                calendar_parts.append(parts["calendar"].cpu().numpy())
                residual_parts.append(parts["residual"].cpu().numpy())
            else:
                pred = model(xb, pb, lb, mb)
            pred_parts.append(pred.cpu().numpy())
    prediction = np.concatenate(pred_parts) if pred_parts else np.empty((0,), dtype=np.float32)
    if not return_parts:
        return prediction
    return prediction, {
        "physical": np.concatenate(physical_parts),
        "calendar": np.concatenate(calendar_parts),
        "residual": np.concatenate(residual_parts),
    }


def prepare_arrays(
    table: pd.DataFrame,
    feature_cols: list[str],
    turbine_capacity: float,
) -> dict[str, np.ndarray]:
    times = pd.to_datetime(table["forecast_kst_dtm"])
    lead = (
        (times - pd.to_datetime(table["data_available_kst_dtm"]))
        .dt.total_seconds()
        .div(3600)
        .round()
        .clip(0, 240)
        .to_numpy(np.int64)
    )
    return {
        "features": table[feature_cols].to_numpy(np.float32),
        "physical": np.clip(
            table["teacher_power_curve_kwh"].to_numpy(np.float32) / turbine_capacity,
            0,
            1,
        ),
        "lead": lead,
        "month": (times.dt.month.to_numpy(np.int64) - 1),
        "target": table["turbine_target"].to_numpy(np.float32),
        "official": table["official_target"].to_numpy(np.float32),
        "time": times.to_numpy(),
    }


def train_turbine_fold(
    train_table: pd.DataFrame,
    val_table: pd.DataFrame,
    feature_cols: list[str],
    group: str,
    turbine: str,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, object]]:
    set_seed(seed)
    train_table = train_table.sort_values("forecast_kst_dtm").reset_index(drop=True)
    val_table = val_table.sort_values("forecast_kst_dtm").reset_index(drop=True)
    turbine_capacity = turbine_capacity_kwh(group)
    group_capacity = GROUP_CAPACITY_KWH[group]
    train = prepare_arrays(train_table, feature_cols, turbine_capacity)
    val = prepare_arrays(val_table, feature_cols, turbine_capacity)
    train_keep = (
        np.isfinite(train["target"])
        & np.isfinite(train["official"])
        & (train["official"] >= group_capacity * args.target_min_output_ratio)
    )
    val_eval_keep = (
        np.isfinite(val["target"])
        & np.isfinite(val["official"])
        & (val["official"] >= group_capacity * args.target_min_output_ratio)
    )
    if int(train_keep.sum()) < 500 or int(val_eval_keep.sum()) < 200:
        raise ValueError(
            f"Insufficient PINN targets {turbine}: train={int(train_keep.sum())} "
            f"val={int(val_eval_keep.sum())}"
        )
    if args.backbone == "tcn":
        train_sequence, _, _, train_time, _ = make_per_turbine_sequences(
            train_table, feature_cols, window=args.window
        )
        val_sequence, _, _, val_time, _ = make_per_turbine_sequences(
            val_table, feature_cols, window=args.window
        )
        if not np.array_equal(pd.to_datetime(train_time), pd.to_datetime(train["time"])):
            raise ValueError(f"Train sequence alignment failed for {turbine}")
        if not np.array_equal(pd.to_datetime(val_time), pd.to_datetime(val["time"])):
            raise ValueError(f"Validation sequence alignment failed for {turbine}")
        x_train_raw = train_sequence[train_keep]
        x_val_raw = val_sequence
    else:
        x_train_raw = train["features"][train_keep]
        x_val_raw = val["features"]
    x_train, x_val, scaler_mean, scaler_std = standardize(x_train_raw, x_val_raw)
    y_train = (train["target"][train_keep] / turbine_capacity).astype(np.float32)
    official_train = train["official"][train_keep]
    weights = (0.5 + np.sqrt(np.clip(official_train / group_capacity, 0, 1))).astype(np.float32)
    physical_train = train["physical"][train_keep].astype(np.float32)
    lead_train = train["lead"][train_keep]
    month_train = train["month"][train_keep]

    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(physical_train),
        torch.from_numpy(lead_train),
        torch.from_numpy(month_train),
        torch.from_numpy(y_train),
        torch.from_numpy(weights),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    if args.backbone == "tcn":
        model = PerTurbineResidualTCNPINN(
            input_size=len(feature_cols),
            hidden_size=args.hidden_size,
            residual_amplitude=args.residual_amplitude,
            num_layers=args.num_layers,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        ).to(device)
    else:
        model = PerTurbineResidualPINN(
            input_size=len(feature_cols),
            hidden_size=args.hidden_size,
            residual_amplitude=args.residual_amplitude,
        ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_score = -np.inf
    best_epoch = -1
    best_state = None
    bad_epochs = 0
    val_indices = np.flatnonzero(val_eval_keep)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, pb, lb, mb, yb, wb in loader:
            xb = xb.to(device, non_blocking=True)
            pb = pb.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred, parts = model(xb, pb, lb, mb, return_parts=True)
            error_rate = torch.abs(pred - yb)
            weighted_l1 = (error_rate * wb).mean()
            price = soft_unit_price(error_rate, args.gamma)
            ficr_soft = (yb * price).sum() / torch.clamp((yb * 4.0).sum(), min=1e-6)
            data_loss = 0.5 * weighted_l1 + 0.5 * (1.0 - ficr_soft)
            anchor_loss = (pred - torch.clamp(parts["physical"], 0, 1)).pow(2).mean()
            reg = model.regularization()
            loss = (
                data_loss
                + args.lambda_anchor * anchor_loss
                + args.lambda_residual * reg["residual_l2"]
                + args.lambda_bias * reg["bias_l2"]
                + args.lambda_scale * reg["scale_l2"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        pred_eval_norm = predict_model(
            model,
            x_val[val_indices],
            val["physical"][val_indices].astype(np.float32),
            val["lead"][val_indices],
            val["month"][val_indices],
            device,
            args.eval_batch_size,
        )
        pred_eval = pred_eval_norm * turbine_capacity
        score, _, _ = turbine_metric(
            val["target"][val_eval_keep],
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
    train_pretrained = predict_model(
        model,
        x_train,
        physical_train,
        lead_train,
        month_train,
        device,
        args.eval_batch_size,
    )
    prediction_norm, parts = predict_model(
        model,
        x_val,
        val["physical"].astype(np.float32),
        val["lead"],
        val["month"],
        device,
        args.eval_batch_size,
        return_parts=True,
    )
    prediction = prediction_norm * turbine_capacity
    output = pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(val["time"]),
            "turbine_id": turbine,
            "pred": prediction,
            "physics_pred": np.clip(val["physical"], 0, 1) * turbine_capacity,
            "learned_physics_pred": np.clip(parts["physical"], 0, 1) * turbine_capacity,
            "calendar_kwh": parts["calendar"] * turbine_capacity,
            "residual_kwh": parts["residual"] * turbine_capacity,
        }
    )
    eval_pred = prediction[val_eval_keep]
    eval_score, eval_nmae, eval_ficr = turbine_metric(
        val["target"][val_eval_keep],
        eval_pred,
        turbine_capacity,
    )
    stats = {
        "group": group,
        "turbine_id": turbine,
        "best_epoch": best_epoch,
        "turbine_val_score": eval_score,
        "turbine_val_nmae": eval_nmae,
        "turbine_val_ficr": eval_ficr,
        "physics_scale": float(model.physics_scale.detach().cpu()),
        "mean_abs_calendar_kwh": float(np.mean(np.abs(output["calendar_kwh"]))),
        "mean_abs_residual_kwh": float(np.mean(np.abs(output["residual_kwh"]))),
        "n_train": int(train_keep.sum()),
        "n_val_all": len(val_table),
        "n_val_target": int(val_eval_keep.sum()),
        "backbone": args.backbone,
        "window": args.window if args.backbone == "tcn" else 1,
        "num_layers": args.num_layers if args.backbone == "tcn" else 0,
    }
    artifact = {
        "turbine_id": turbine,
        "model_state": {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        },
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "train_time": pd.to_datetime(train["time"][train_keep]),
        "val_time": pd.to_datetime(val["time"]),
        "train_features": x_train,
        "val_features": x_val,
        "train_physical": physical_train,
        "val_physical": val["physical"].astype(np.float32),
        "train_lead": lead_train,
        "val_lead": val["lead"],
        "train_month": month_train,
        "val_month": val["month"],
        "train_official": official_train.astype(np.float32),
        "train_pretrained": train_pretrained.astype(np.float32),
    }
    del model, optimizer, loader, dataset, x_train, x_val
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output, stats, artifact


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
    unit_price = soft_unit_price(error, gamma)
    ficr = (target * unit_price).sum() / torch.clamp((target * 4.0).sum(), min=1e-6)
    return 0.5 * nmae + 0.5 * (1.0 - ficr), nmae, ficr


def build_artifact_model(
    input_size: int, args: argparse.Namespace, device: torch.device
) -> PerTurbineResidualPINN:
    if args.backbone == "tcn":
        return PerTurbineResidualTCNPINN(
            input_size=input_size,
            hidden_size=args.hidden_size,
            residual_amplitude=args.residual_amplitude,
            num_layers=args.num_layers,
            kernel_size=args.kernel_size,
            dropout=args.dropout,
        ).to(device)
    return PerTurbineResidualPINN(
        input_size=input_size,
        hidden_size=args.hidden_size,
        residual_amplitude=args.residual_amplitude,
    ).to(device)


def fine_tune_group_backbones(
    artifacts: list[dict[str, object]],
    feature_cols: list[str],
    group: str,
    pred_year: int,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    train_time = common_artifact_times(artifacts, "train")
    val_time = common_artifact_times(artifacts, "val")
    train_official = aligned_artifact_values(
        artifacts[0], "train", "train_official", train_time
    ).astype(np.float32)
    for artifact in artifacts[1:]:
        other = aligned_artifact_values(
            artifact, "train", "train_official", train_time
        )
        if not np.allclose(train_official, other, equal_nan=True):
            raise ValueError(
                f"Official group target mismatch for {group} pred_year={pred_year}"
            )

    train_inputs = []
    val_inputs = []
    pretrained_parts = []
    models = []
    for artifact in artifacts:
        train_inputs.append(
            {
                field: aligned_artifact_values(
                    artifact, "train", f"train_{field}", train_time
                )
                for field in ["features", "physical", "lead", "month"]
            }
        )
        val_inputs.append(
            {
                field: aligned_artifact_values(
                    artifact, "val", f"val_{field}", val_time
                )
                for field in ["features", "physical", "lead", "month"]
            }
        )
        pretrained_parts.append(
            aligned_artifact_values(
                artifact, "train", "train_pretrained", train_time
            ).astype(np.float32)
        )
        model = build_artifact_model(len(feature_cols), args, device)
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
                    model(
                        torch.from_numpy(values["features"][batch_indices]).to(device),
                        torch.from_numpy(values["physical"][batch_indices]).to(device),
                        torch.from_numpy(values["lead"][batch_indices]).to(device),
                        torch.from_numpy(values["month"][batch_indices]).to(device),
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
                prediction_parts[turbine_index].append(
                    model(
                        torch.from_numpy(values["features"][start:stop]).to(device),
                        torch.from_numpy(values["physical"][start:stop]).to(device),
                        torch.from_numpy(values["lead"][start:stop]).to(device),
                        torch.from_numpy(values["month"][start:stop]).to(device),
                    ).cpu().numpy()
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
                    "pred": np.clip(prediction, 0.0, 1.0) * turbine_capacity,
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
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} epochs={args.epochs} patience={args.patience} "
        f"residual_amp={args.residual_amplitude} gamma={args.gamma} "
        f"feature_variant={args.feature_variant} backbone={args.backbone} "
        f"window={args.window} layers={args.num_layers}",
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
    joint_training_rows = []
    turbine_parts = []
    baseline_turbine_parts = []
    group_parts = []
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
            if args.feature_variant == "optimal_grid_replace_local16":
                features = load_optimal_grid_fold_features(
                    base_features, args.cache_root, group, pred_year
                )
                teacher_input_cols = optimal_grid_input_columns(group)
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
            table = features.merge(teacher[keys + TEACHER_FEATURE_COLS], on=keys, how="inner")
            table = table.merge(target_one, on=keys, how="left")
            table = table.merge(label_one, on="forecast_kst_dtm", how="left")
            table["forecast_kst_dtm"] = pd.to_datetime(table["forecast_kst_dtm"])
            table["data_available_kst_dtm"] = pd.to_datetime(table["data_available_kst_dtm"])
            feature_cols = [
                *teacher_input_cols,
                *TEACHER_FEATURE_COLS,
                *terrain_feature_columns(args.terrain_variant),
            ]
            print(
                f"\n{group} pred_year={pred_year} train={train_years} "
                f"variant={args.feature_variant} features={len(feature_cols)} "
                f"share_alpha={args.target_share_alpha:.2f}",
                flush=True,
            )
            fold_turbines = []
            fold_artifacts = []
            for turbine_index, turbine in enumerate(GROUP_TURBINE_PREFIXES[group]):
                turbine_table = table.loc[table["turbine_id"].eq(turbine)]
                train_table = turbine_table.loc[
                    turbine_table["forecast_kst_dtm"].dt.year.isin(train_years)
                ].copy()
                val_table = turbine_table.loc[
                    turbine_table["forecast_kst_dtm"].dt.year.eq(pred_year)
                ].copy()
                output, stats, artifact = train_turbine_fold(
                    train_table,
                    val_table,
                    feature_cols,
                    group,
                    turbine,
                    args,
                    device,
                    args.seed + pred_year * 100 + turbine_index,
                )
                model_counter += 1
                output["group"] = group
                output["pred_year"] = pred_year
                fold_turbines.append(output)
                fold_artifacts.append(artifact)
                baseline_turbine_parts.append(output)
                stats.update(
                    {
                        "pred_year": pred_year,
                        "train_years": ",".join(map(str, train_years)),
                        "n_features": len(feature_cols),
                        "target_share_alpha": args.target_share_alpha,
                        "static_share": float(static_shares.loc[turbine]),
                    }
                )
                training_rows.append(stats)
                print(
                    f"  {turbine}: epoch={stats['best_epoch']} "
                    f"score={stats['turbine_val_score']:.5f} "
                    f"scale={stats['physics_scale']:.3f} "
                    f"res={stats['mean_abs_residual_kwh']:.1f}kWh",
                    flush=True,
                )
                if args.smoke_test and args.joint_finetune_epochs <= 0:
                    print("smoke test complete", flush=True)
                    return

            baseline_turbines = pd.concat(fold_turbines, ignore_index=True)
            if args.joint_finetune_epochs > 0:
                joint_turbines, joint_history = fine_tune_group_backbones(
                    artifacts=fold_artifacts,
                    feature_cols=feature_cols,
                    group=group,
                    pred_year=pred_year,
                    args=args,
                    device=device,
                )
                joint_turbines["pred_year"] = pred_year
                turbine_parts.append(joint_turbines)
                joint_training_rows.extend(joint_history)
                variants = [
                    ("per_turbine_physics_curve", baseline_turbines, "physics_pred"),
                    ("baseline_retrained", baseline_turbines, "pred"),
                    ("group_joint_score", joint_turbines, "pred"),
                ]
            else:
                turbine_parts.append(baseline_turbines)
                variants = [
                    ("per_turbine_physics_curve", baseline_turbines, "physics_pred"),
                    ("per_turbine_pinn", baseline_turbines, "pred"),
                ]

            expected_turbines = len(GROUP_TURBINE_PREFIXES[group])
            for variant, fold_variant, pred_col in variants:
                group_table = (
                    fold_variant.groupby("forecast_kst_dtm", as_index=False)
                    .agg(
                        pred=(pred_col, "sum"),
                        n_turbines=("turbine_id", "nunique"),
                    )
                    .merge(label_one, on="forecast_kst_dtm", how="inner")
                    .dropna(subset=["official_target"])
                )
                if not group_table["n_turbines"].eq(expected_turbines).all():
                    raise ValueError(
                        f"Incomplete PINN turbine sum {group} pred_year={pred_year} {variant}"
                    )
                group_table["pred"] = group_table["pred"].clip(
                    0, GROUP_CAPACITY_KWH[group]
                )
                score, nmae, ficr = group_metric(
                    group_table["official_target"].to_numpy(float),
                    group_table["pred"].to_numpy(float),
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
                        "n_rows": len(group_table),
                        "n_features": len(feature_cols),
                        "target_share_alpha": args.target_share_alpha,
                    }
                )
                part = group_table[
                    ["forecast_kst_dtm", "official_target", "n_turbines", "pred"]
                ].copy()
                part["variant"] = variant
                part["group"] = group
                part["pred_year"] = pred_year
                group_parts.append(part)
                print(
                    f"{group} {pred_year} {variant}: score={score:.6f} "
                    f"nMAE={nmae:.6f} FiCR={ficr:.6f}",
                    flush=True,
                )
            del fold_artifacts
            gc.collect()
            if args.smoke_test:
                print("smoke test complete after one group-fold", flush=True)
                return

    scores = pd.DataFrame(score_rows)
    summary_rows = []
    for variant, variant_scores in scores.groupby("variant"):
        fold_means = variant_scores.groupby("pred_year", as_index=False)[
            ["score", "nmae", "ficr"]
        ].mean()
        summary_rows.append(
            {
                "variant": variant,
                "mean_score": fold_means["score"].mean(),
                "mean_nmae": fold_means["nmae"].mean(),
                "mean_ficr": fold_means["ficr"].mean(),
                "worst_fold": fold_means["score"].min(),
                "std_score": fold_means["score"].std(ddof=0),
                "n_folds": len(fold_means),
                "n_models": model_counter,
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("mean_score", ascending=False)
    pd.DataFrame(training_rows).to_csv(
        args.results_dir / f"{args.stem}_training.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(joint_training_rows).to_csv(
        args.results_dir / f"{args.stem}_joint_training.csv",
        index=False,
        encoding="utf-8-sig",
    )
    scores.to_csv(args.results_dir / f"{args.stem}_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.results_dir / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(group_parts, ignore_index=True).to_csv(
        args.results_dir / f"{args.stem}_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(turbine_parts, ignore_index=True).to_csv(
        args.results_dir / f"{args.stem}_turbine_predictions.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(baseline_turbine_parts, ignore_index=True).to_csv(
        args.results_dir / f"{args.stem}_baseline_turbine_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print("\n=== summary ===", flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

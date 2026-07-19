from __future__ import annotations

import argparse
import copy
import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader, TensorDataset

import _bootstrap  # noqa: F401
from experiments.evaluate_per_turbine_bin_moe_tcn_oof import (
    YEARS,
    build_fold_features,
    group_score,
    parse_csv,
    set_seed,
)
from models.group_decision_q import IssueDecisionQTCN
from models.issue_block_tcn import IssueBlockTCN
from utils.decision_reward import (
    hard_score_reward_matrix,
    smooth_ficr_reward,
    smooth_six_reward,
)
from utils.extra_trees_scada_stack import (
    build_extra_trees_scada_stack,
    cubic_feature_channels,
    scada_wind_matrix,
)
from utils.group_local_panel import build_group_local_panel
from utils.issue_block_dataset import IssueBlockData, make_issue_blocks
from utils.metrics import (
    GROUP_CAPACITY_KWH,
    TARGET_COLS,
    group_nmae_ficr,
    pooled_oof_summary,
)
from utils.per_turbine_features import get_or_build_group_feature_cache
from utils.per_turbine_optimal_grid_builder import build_wind_candidate_matrix
from utils.per_turbine_scada import (
    build_official_aligned_turbine_targets,
    build_turbine_scada_hourly,
)
from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.per_turbine_sequence import SequenceStandardScaler
from utils.target_transforms import (
    inverse_normalized_prediction,
)


VARIANTS = (
    "point_power",
    "point_cuberoot",
    "pure_band",
    "pure6_band",
    "decision_q",
)

LEGACY_MODE = "legacy"
ET_SCADA_STACK_MODE = "et_scada_stack"
LEGACY_STEM = "group_decision_q_tcn_oof_v1"
ET_SCADA_STEM = "extra_trees_scada_tcn2_fixed_oof_v1"
STACK_BASELINE_VARIANT = "baseline_raw_features"
STACK_RAW_VARIANT = "et_scada_raw"
STACK_WIND_ISO_VARIANT = "et_scada_wind_isotonic"
STACK_POWER_ISO_VARIANT = "et_scada_wind_power_isotonic"
POWER_ISOTONIC_ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)


@dataclass
class FoldArrays:
    x_train: np.ndarray
    y_train: np.ndarray
    train_valid: np.ndarray
    x_validation: np.ndarray
    y_validation: np.ndarray
    validation_times: np.ndarray
    mean_train_target: float
    n_train_rows: int
    n_validation_rows: int
    x_inference: np.ndarray | None = None
    inference_times: np.ndarray | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Direct group point-regression versus expected-Score Q TCN."
    )
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--years", default=",".join(map(str, YEARS)))
    parser.add_argument(
        "--mode",
        choices=[LEGACY_MODE, ET_SCADA_STACK_MODE],
        default=LEGACY_MODE,
    )
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--q-value-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--target-min-output-ratio", type=float, default=0.10)
    parser.add_argument("--prediction-floor", type=float, default=0.10)
    parser.add_argument("--candidate-min", type=float, default=0.10)
    parser.add_argument("--candidate-max", type=float, default=1.00)
    parser.add_argument("--candidate-step", type=float, default=0.005)
    parser.add_argument("--band-temperature-start", type=float, default=0.10)
    parser.add_argument("--band-temperature-end", type=float, default=0.01)
    parser.add_argument("--band-min-epochs", type=int, default=20)
    parser.add_argument(
        "--feature-variant",
        choices=["optimal_grid_replace_local16", "optimal_grid_issue_context"],
        default="optimal_grid_replace_local16",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--within-year-folds", type=int, default=3)
    parser.add_argument("--et-estimators", type=int, default=500)
    parser.add_argument("--et-min-samples-leaf", type=int, default=2)
    parser.add_argument("--et-max-features", type=float, default=0.80)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument("--rebuild-optimal-grid-cache", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--stem", default=LEGACY_STEM)
    return parser.parse_args()


def candidate_grid(args: argparse.Namespace) -> np.ndarray:
    if not 0.0 <= args.candidate_min < args.candidate_max <= 1.0:
        raise ValueError("Candidate bounds must satisfy 0 <= min < max <= 1")
    if args.candidate_step <= 0.0:
        raise ValueError("--candidate-step must be positive")
    count = int(round((args.candidate_max - args.candidate_min) / args.candidate_step))
    candidates = args.candidate_min + np.arange(count + 1) * args.candidate_step
    if not np.isclose(candidates[-1], args.candidate_max, atol=1e-8):
        raise ValueError("Candidate step must land exactly on candidate max")
    return candidates.astype(np.float32)


def prepare_fold_arrays(
    blocks: IssueBlockData,
    pred_year: int,
    capacity: float,
    min_output_ratio: float,
    inference_blocks: IssueBlockData | None = None,
) -> FoldArrays:
    train_issue_mask = blocks.years != pred_year
    validation_issue_mask = blocks.years == pred_year
    train_features = blocks.features[train_issue_mask]
    train_target = blocks.targets[train_issue_mask, :, 0]
    validation_features = blocks.features[validation_issue_mask]
    validation_target = blocks.targets[validation_issue_mask, :, 0]
    train_valid = (
        np.isfinite(train_target)
        & (train_target >= capacity * float(min_output_ratio))
    )
    validation_valid = (
        np.isfinite(validation_target)
        & (validation_target >= capacity * float(min_output_ratio))
    )
    train_issue_keep = train_valid.any(axis=1)
    train_features = train_features[train_issue_keep]
    train_target = train_target[train_issue_keep]
    train_valid = train_valid[train_issue_keep]
    if int(train_valid.sum()) < 1000 or int(validation_valid.sum()) < 200:
        raise ValueError(
            f"Insufficient group targets for pred{pred_year}: "
            f"train={int(train_valid.sum())} validation={int(validation_valid.sum())}"
        )

    scaler = SequenceStandardScaler()
    scaler.fit(train_features[train_valid])
    x_train = scaler.transform(train_features)
    x_validation = scaler.transform(validation_features)
    x_inference = None
    inference_times = None
    if inference_blocks is not None:
        if inference_blocks.feature_cols != blocks.feature_cols:
            raise ValueError("Inference issue-block features differ from training features")
        x_inference = scaler.transform(inference_blocks.features)
        inference_times = inference_blocks.forecast_times.copy()
    y_train = np.maximum(train_target / capacity, 0.0).astype(np.float32)
    y_validation = np.maximum(validation_target / capacity, 0.0).astype(
        np.float32
    )
    mean_train_target = float(y_train[train_valid].mean())
    return FoldArrays(
        x_train=x_train,
        y_train=np.nan_to_num(y_train, nan=0.0),
        train_valid=train_valid,
        x_validation=x_validation,
        y_validation=y_validation,
        validation_times=blocks.forecast_times[validation_issue_mask],
        mean_train_target=mean_train_target,
        n_train_rows=int(train_valid.sum()),
        n_validation_rows=int(validation_valid.sum()),
        x_inference=x_inference,
        inference_times=inference_times,
    )


def prepare_indexed_fold_arrays(
    blocks: IssueBlockData,
    train_issue_indices: np.ndarray,
    validation_issue_indices: np.ndarray,
    capacity: float,
    min_output_ratio: float,
) -> FoldArrays:
    train_issue_indices = np.asarray(train_issue_indices, dtype=int)
    validation_issue_indices = np.asarray(validation_issue_indices, dtype=int)
    train_features = blocks.features[train_issue_indices]
    train_target = blocks.targets[train_issue_indices, :, 0]
    validation_features = blocks.features[validation_issue_indices]
    validation_target = blocks.targets[validation_issue_indices, :, 0]
    train_valid = np.isfinite(train_target) & (
        train_target >= capacity * float(min_output_ratio)
    )
    validation_valid = np.isfinite(validation_target) & (
        validation_target >= capacity * float(min_output_ratio)
    )
    train_issue_keep = train_valid.any(axis=1)
    train_features = train_features[train_issue_keep]
    train_target = train_target[train_issue_keep]
    train_valid = train_valid[train_issue_keep]
    if int(train_valid.sum()) < 1000 or int(validation_valid.sum()) < 200:
        raise ValueError(
            "Insufficient indexed group targets: "
            f"train={int(train_valid.sum())} validation={int(validation_valid.sum())}"
        )

    scaler = SequenceStandardScaler()
    scaler.fit(train_features[train_valid])
    x_train = scaler.transform(train_features)
    x_validation = scaler.transform(validation_features)
    y_train = np.maximum(train_target / capacity, 0.0).astype(np.float32)
    y_validation = np.maximum(validation_target / capacity, 0.0).astype(np.float32)
    return FoldArrays(
        x_train=x_train,
        y_train=np.nan_to_num(y_train, nan=0.0),
        train_valid=train_valid,
        x_validation=x_validation,
        y_validation=y_validation,
        validation_times=blocks.forecast_times[validation_issue_indices],
        mean_train_target=float(y_train[train_valid].mean()),
        n_train_rows=int(train_valid.sum()),
        n_validation_rows=int(validation_valid.sum()),
    )


def make_loader(
    arrays: FoldArrays,
    args: argparse.Namespace,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(arrays.x_train),
        torch.from_numpy(arrays.y_train),
        torch.from_numpy(arrays.train_valid.astype(np.float32)),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def predict_point(
    model: IssueBlockTCN,
    features: np.ndarray,
    target_transform: str,
    prediction_floor: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            parts.append(model(batch)[..., 0].cpu().numpy())
    transformed = np.concatenate(parts, axis=0)
    prediction = inverse_normalized_prediction(transformed, target_transform)
    return np.clip(prediction, prediction_floor, 1.0)


def predict_q(
    model: IssueDecisionQTCN,
    features: np.ndarray,
    candidates: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    prediction_parts = []
    value_parts = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            values = model(batch, candidates)
            selected = values.argmax(dim=-1)
            prediction_parts.append(candidates[selected].cpu().numpy())
            value_parts.append(values.max(dim=-1).values.cpu().numpy())
    return (
        np.concatenate(prediction_parts, axis=0).astype(np.float32),
        np.concatenate(value_parts, axis=0).astype(np.float32),
    )


def bounded_band_prediction(raw_prediction: torch.Tensor, floor: float) -> torch.Tensor:
    return float(floor) + (1.0 - float(floor)) * torch.sigmoid(raw_prediction)


def predict_band(
    model: IssueBlockTCN,
    features: np.ndarray,
    prediction_floor: float,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            raw_prediction = model(batch)[..., 0]
            parts.append(
                bounded_band_prediction(raw_prediction, prediction_floor)
                .cpu()
                .numpy()
            )
    return np.concatenate(parts, axis=0).astype(np.float32)


def train_point_fold(
    arrays: FoldArrays,
    group: str,
    target_transform: str,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]]]:
    set_seed(seed)
    loader = make_loader(arrays, args, seed)
    model = IssueBlockTCN(
        input_size=arrays.x_train.shape[-1],
        output_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        full_context=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_score = -np.inf
    best_epoch = 0
    best_prediction = None
    best_nmae = np.nan
    best_ficr = np.nan
    bad_epochs = 0
    history = []
    capacity = float(GROUP_CAPACITY_KWH[group])
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for xb, yb_power, mb in loader:
            xb = xb.to(device, non_blocking=True)
            yb_power = yb_power.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            if target_transform == "power":
                yb = yb_power
            elif target_transform == "cuberoot":
                yb = torch.pow(torch.clamp(yb_power, 0.0, 1.0), 1.0 / 3.0)
            else:
                raise ValueError(f"Unknown target transform: {target_transform}")
            weights = 0.5 + torch.sqrt(torch.clamp(yb_power, 0.0, 1.0))
            weighted_mask = weights * mb
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)[..., 0]
            loss = (
                torch.abs(prediction - yb) * weighted_mask
            ).sum() / torch.clamp(weighted_mask.sum(), min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        validation_prediction = predict_point(
            model,
            arrays.x_validation,
            target_transform,
            args.prediction_floor,
            device,
            args.eval_batch_size,
        )
        score, nmae, ficr = group_score(
            arrays.y_validation.reshape(-1) * capacity,
            validation_prediction.reshape(-1) * capacity,
            group,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "score": score,
                "nmae": nmae,
                "ficr": ficr,
            }
        )
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            best_prediction = validation_prediction.copy()
            best_nmae = nmae
            best_ficr = ficr
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    if best_prediction is None:
        raise RuntimeError(f"No point checkpoint selected for {group}/{target_transform}")
    stats = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_nmae": best_nmae,
        "best_ficr": best_ficr,
        "epochs_trained": len(history),
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
    del model, optimizer, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_prediction, stats, history


def _train_band_fold(
    arrays: FoldArrays,
    group: str,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    loss_mode: str = "pure68",
    inference_features: np.ndarray | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray | None,
    dict[str, object],
    list[dict[str, object]],
]:
    set_seed(seed)
    loader = make_loader(arrays, args, seed)
    model = IssueBlockTCN(
        input_size=arrays.x_train.shape[-1],
        output_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        full_context=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_ficr = -np.inf
    best_epoch = 0
    best_prediction = None
    best_state = None
    best_score = np.nan
    best_nmae = np.nan
    bad_epochs = 0
    history = []
    capacity = float(GROUP_CAPACITY_KWH[group])
    temperature_ratio = args.band_temperature_end / args.band_temperature_start
    for epoch in range(1, args.epochs + 1):
        progress = (epoch - 1) / max(args.epochs - 1, 1)
        temperature = args.band_temperature_start * temperature_ratio**progress
        model.train()
        epoch_losses = []
        for xb, yb, mb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            raw_prediction = model(xb)[..., 0]
            prediction = bounded_band_prediction(
                raw_prediction, args.prediction_floor
            )
            reward_function = (
                smooth_ficr_reward if loss_mode == "pure68" else smooth_six_reward
            )
            if loss_mode not in {"pure68", "pure6"}:
                raise ValueError(f"Unknown band loss mode: {loss_mode}")
            reward = reward_function(
                yb,
                prediction,
                arrays.mean_train_target,
                temperature,
            )
            loss = -(reward * mb).sum() / torch.clamp(mb.sum(), min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        validation_prediction = predict_band(
            model,
            arrays.x_validation,
            args.prediction_floor,
            device,
            args.eval_batch_size,
        )
        score, nmae, ficr = group_score(
            arrays.y_validation.reshape(-1) * capacity,
            validation_prediction.reshape(-1) * capacity,
            group,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "temperature": temperature,
                "score": score,
                "nmae": nmae,
                "ficr": ficr,
            }
        )
        if ficr > best_ficr + args.min_delta:
            best_ficr = ficr
            best_epoch = epoch
            best_prediction = validation_prediction.copy()
            if inference_features is not None:
                best_state = copy.deepcopy(model.state_dict())
            best_score = score
            best_nmae = nmae
            bad_epochs = 0
        elif epoch >= args.band_min_epochs:
            bad_epochs += 1
        if epoch >= args.band_min_epochs and bad_epochs >= args.patience:
            break

    if best_prediction is None:
        raise RuntimeError(f"No pure-band checkpoint selected for {group}")
    inference_prediction = None
    if inference_features is not None:
        if best_state is None:
            raise RuntimeError(f"No pure-band state selected for {group}")
        model.load_state_dict(best_state)
        inference_prediction = predict_band(
            model,
            inference_features,
            args.prediction_floor,
            device,
            args.eval_batch_size,
        )
    stats = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_nmae": best_nmae,
        "best_ficr": best_ficr,
        "epochs_trained": len(history),
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "mean_train_target": arrays.mean_train_target,
        "loss_mode": loss_mode,
    }
    del model, optimizer, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_prediction, inference_prediction, stats, history


def train_band_fold(
    arrays: FoldArrays,
    group: str,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    loss_mode: str = "pure68",
) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]]]:
    prediction, _, stats, history = _train_band_fold(
        arrays,
        group,
        args,
        device,
        seed,
        loss_mode=loss_mode,
    )
    return prediction, stats, history


def train_band_fixed_fold(
    arrays: FoldArrays,
    group: str,
    fixed_epoch: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict[str, object], list[dict[str, object]]]:
    if fixed_epoch < 1:
        raise ValueError("fixed_epoch must be positive")
    set_seed(seed)
    loader = make_loader(arrays, args, seed)
    model = IssueBlockTCN(
        input_size=arrays.x_train.shape[-1],
        output_size=1,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        full_context=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history: list[dict[str, object]] = []
    temperature_ratio = args.band_temperature_end / args.band_temperature_start
    for epoch in range(1, fixed_epoch + 1):
        progress = (epoch - 1) / max(args.epochs - 1, 1)
        temperature = args.band_temperature_start * temperature_ratio**progress
        model.train()
        epoch_losses = []
        for xb, yb, mb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = bounded_band_prediction(
                model(xb)[..., 0], args.prediction_floor
            )
            reward = smooth_ficr_reward(
                yb,
                prediction,
                arrays.mean_train_target,
                temperature,
            )
            loss = -(reward * mb).sum() / torch.clamp(mb.sum(), min=1.0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(epoch_losses)),
                "temperature": float(temperature),
            }
        )

    prediction = predict_band(
        model,
        arrays.x_validation,
        args.prediction_floor,
        device,
        args.eval_batch_size,
    )
    capacity = float(GROUP_CAPACITY_KWH[group])
    score, nmae, ficr = group_score(
        arrays.y_validation.reshape(-1) * capacity,
        prediction.reshape(-1) * capacity,
        group,
    )
    stats = {
        "fixed_epoch": int(fixed_epoch),
        "score": float(score),
        "nmae": float(nmae),
        "ficr": float(ficr),
        "n_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "n_train_rows": int(arrays.n_train_rows),
        "n_validation_rows": int(arrays.n_validation_rows),
    }
    del model, optimizer, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction, stats, history


def train_band_fold_with_inference(
    arrays: FoldArrays,
    inference_features: np.ndarray,
    group: str,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
    loss_mode: str = "pure68",
) -> tuple[np.ndarray, np.ndarray, dict[str, object], list[dict[str, object]]]:
    prediction, inference_prediction, stats, history = _train_band_fold(
        arrays,
        group,
        args,
        device,
        seed,
        loss_mode=loss_mode,
        inference_features=inference_features,
    )
    if inference_prediction is None:
        raise RuntimeError(f"No pure-band inference prediction for {group}")
    return prediction, inference_prediction, stats, history


def train_q_fold(
    arrays: FoldArrays,
    group: str,
    candidates_array: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object], list[dict[str, object]]]:
    set_seed(seed)
    loader = make_loader(arrays, args, seed)
    candidates = torch.from_numpy(candidates_array).to(device)
    model = IssueDecisionQTCN(
        input_size=arrays.x_train.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        value_size=args.q_value_size,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best_score = -np.inf
    best_epoch = 0
    best_prediction = None
    best_values = None
    best_nmae = np.nan
    best_ficr = np.nan
    bad_epochs = 0
    history = []
    capacity = float(GROUP_CAPACITY_KWH[group])
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for xb, yb, mb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb, candidates)
            reward = hard_score_reward_matrix(
                yb, candidates, arrays.mean_train_target
            )
            squared_error = torch.square(prediction - reward) * mb.unsqueeze(-1)
            loss = squared_error.sum() / torch.clamp(
                mb.sum() * candidates.numel(), min=1.0
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        validation_prediction, validation_values = predict_q(
            model,
            arrays.x_validation,
            candidates,
            device,
            args.eval_batch_size,
        )
        score, nmae, ficr = group_score(
            arrays.y_validation.reshape(-1) * capacity,
            validation_prediction.reshape(-1) * capacity,
            group,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "score": score,
                "nmae": nmae,
                "ficr": ficr,
                "mean_selected_value": float(validation_values.mean()),
            }
        )
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch
            best_prediction = validation_prediction.copy()
            best_values = validation_values.copy()
            best_nmae = nmae
            best_ficr = ficr
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            break

    if best_prediction is None or best_values is None:
        raise RuntimeError(f"No Q checkpoint selected for {group}")
    stats = {
        "best_epoch": best_epoch,
        "best_score": best_score,
        "best_nmae": best_nmae,
        "best_ficr": best_ficr,
        "epochs_trained": len(history),
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "mean_train_target": arrays.mean_train_target,
    }
    del model, optimizer, loader
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return best_prediction, best_values, stats, history


def _remove_issue_overlap(
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    forecast_times: np.ndarray,
) -> np.ndarray:
    heldout_times = np.unique(forecast_times[validation_indices].reshape(-1))
    overlap = np.isin(forecast_times[train_indices], heldout_times).any(axis=1)
    return np.asarray(train_indices, dtype=int)[~overlap]


def _power_inner_holdouts(
    blocks: IssueBlockData,
    outer_train_indices: np.ndarray,
    capacity: float,
    min_output_ratio: float,
    within_year_folds: int,
) -> list[tuple[str, np.ndarray]]:
    target = blocks.targets[outer_train_indices, :, 0]
    usable = np.isfinite(target) & (target >= capacity * min_output_ratio)
    eligible = outer_train_indices[usable.any(axis=1)]
    present_years = sorted(np.unique(blocks.years[eligible]).astype(int).tolist())
    if len(present_years) >= 2:
        return [
            (f"year{year}", eligible[blocks.years[eligible] == year])
            for year in present_years
        ]
    ordered = eligible[np.argsort(blocks.issue_times[eligible])]
    chunks = [
        chunk
        for chunk in np.array_split(ordered, int(within_year_folds))
        if len(chunk)
    ]
    return [(f"chunk{index + 1}", chunk) for index, chunk in enumerate(chunks)]


def _prediction_part(
    arrays: FoldArrays,
    group: str,
    pred_year: int,
    prediction: np.ndarray,
    variant: str,
) -> pd.DataFrame:
    capacity = float(GROUP_CAPACITY_KWH[group])
    return pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(arrays.validation_times.reshape(-1)),
            "group": group,
            "pred_year": int(pred_year),
            "official_target": arrays.y_validation.reshape(-1) * capacity,
            "pred": prediction.reshape(-1) * capacity,
            "variant": variant,
        }
    ).dropna(subset=["official_target", "pred"])


def _nested_power_oof(
    blocks: IssueBlockData,
    group: str,
    pred_year: int,
    fixed_epoch: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    capacity = float(GROUP_CAPACITY_KWH[group])
    outer_train = np.flatnonzero(blocks.years != pred_year)
    holdouts = _power_inner_holdouts(
        blocks,
        outer_train,
        capacity,
        args.target_min_output_ratio,
        args.within_year_folds,
    )
    parts = []
    rows: list[dict[str, object]] = []
    for fold_index, (fold_name, validation_indices) in enumerate(holdouts):
        train_indices = outer_train[~np.isin(outer_train, validation_indices)]
        train_indices = _remove_issue_overlap(
            train_indices,
            validation_indices,
            blocks.forecast_times,
        )
        arrays = prepare_indexed_fold_arrays(
            blocks,
            train_indices,
            validation_indices,
            capacity,
            args.target_min_output_ratio,
        )
        prediction, stats, _ = train_band_fixed_fold(
            arrays,
            group,
            fixed_epoch,
            args,
            device,
            seed + fold_index * 1009,
        )
        part = _prediction_part(
            arrays,
            group,
            int(blocks.years[validation_indices][0]),
            prediction,
            STACK_WIND_ISO_VARIANT,
        )
        part["inner_fold"] = fold_name
        parts.append(part)
        rows.append(
            {
                "stage": "power_isotonic_inner_tcn_oof",
                "group": group,
                "pred_year": int(pred_year),
                "inner_fold": fold_name,
                **stats,
            }
        )
    if not parts:
        raise ValueError(f"No nested power OOF predictions for {group}/val{pred_year}")
    return pd.concat(parts, ignore_index=True), rows


def _fit_power_isotonic(
    frame: pd.DataFrame,
    capacity: float,
    min_output_ratio: float,
) -> IsotonicRegression:
    valid = (
        np.isfinite(frame["pred"])
        & np.isfinite(frame["official_target"])
        & frame["official_target"].ge(capacity * min_output_ratio)
    )
    if int(valid.sum()) < 500:
        raise ValueError(f"Too few power Isotonic rows: {int(valid.sum())}")
    calibrator = IsotonicRegression(
        increasing=True,
        y_min=capacity * min_output_ratio,
        y_max=capacity,
        out_of_bounds="clip",
    )
    calibrator.fit(
        frame.loc[valid, "pred"].to_numpy(float),
        frame.loc[valid, "official_target"].to_numpy(float),
    )
    return calibrator


def _safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 2:
        return np.nan
    if np.unique(left[valid]).size < 2 or np.unique(right[valid]).size < 2:
        return np.nan
    return float(spearmanr(left[valid], right[valid]).statistic)


def _calibrate_power_outer(
    inner_oof: pd.DataFrame,
    outer_prediction: pd.DataFrame,
    group: str,
    pred_year: int,
    min_output_ratio: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    capacity = float(GROUP_CAPACITY_KWH[group])
    crossfit = np.full(len(inner_oof), np.nan, dtype=float)
    for fold_name in inner_oof["inner_fold"].unique():
        heldout = inner_oof["inner_fold"].eq(fold_name).to_numpy()
        calibrator = _fit_power_isotonic(
            inner_oof.loc[~heldout], capacity, min_output_ratio
        )
        crossfit[heldout] = calibrator.predict(
            inner_oof.loc[heldout, "pred"].to_numpy(float)
        )
    if not np.isfinite(crossfit).all():
        raise ValueError(f"Incomplete power Isotonic cross-fit: {group}/val{pred_year}")

    actual = inner_oof["official_target"].to_numpy(float)
    raw = inner_oof["pred"].to_numpy(float)
    alpha_scores: dict[float, float] = {}
    for alpha in POWER_ISOTONIC_ALPHAS:
        blended = np.clip(
            raw + alpha * (crossfit - raw),
            capacity * min_output_ratio,
            capacity,
        )
        nmae, ficr = group_nmae_ficr(actual, blended, capacity)
        alpha_scores[alpha] = 0.5 * (1.0 - nmae) + 0.5 * ficr
    selected_alpha = max(
        POWER_ISOTONIC_ALPHAS,
        key=lambda alpha: (alpha_scores[alpha], -alpha),
    )

    final_calibrator = _fit_power_isotonic(inner_oof, capacity, min_output_ratio)
    calibrated = outer_prediction.copy()
    raw_outer = calibrated["pred"].to_numpy(float)
    iso_outer = final_calibrator.predict(raw_outer)
    calibrated["pred"] = np.clip(
        raw_outer + selected_alpha * (iso_outer - raw_outer),
        capacity * min_output_ratio,
        capacity,
    )
    calibrated["variant"] = STACK_POWER_ISO_VARIANT
    calibrated["power_isotonic_alpha"] = float(selected_alpha)

    valid = (
        np.isfinite(calibrated["official_target"])
        & calibrated["official_target"].ge(capacity * min_output_ratio)
    )
    raw_valid = raw_outer[valid]
    calibrated_valid = calibrated.loc[valid, "pred"].to_numpy(float)
    actual_valid = calibrated.loc[valid, "official_target"].to_numpy(float)
    mapping = (
        pd.DataFrame({"raw": raw_valid, "calibrated": calibrated_valid})
        .groupby("raw", sort=True)["calibrated"]
        .first()
        .to_numpy(float)
    )
    diagnostic: dict[str, object] = {
        "stage": "power_isotonic",
        "group": group,
        "pred_year": int(pred_year),
        "selected_alpha": float(selected_alpha),
        "raw_spearman": _safe_spearman(raw_valid, actual_valid),
        "calibrated_spearman": _safe_spearman(calibrated_valid, actual_valid),
        "mapping_spearman": _safe_spearman(raw_valid, calibrated_valid),
        "unique_retention": int(np.unique(calibrated_valid).size)
        / max(int(np.unique(raw_valid).size), 1),
        "monotonic_inversions": int(np.sum(np.diff(mapping) < -1e-8)),
    }
    for alpha, score in alpha_scores.items():
        diagnostic[f"alpha_{alpha:.2f}_inner_score"] = float(score)
    return calibrated, diagnostic


def _stack_score_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (variant, group), part in predictions.groupby(["variant", "group"], sort=False):
        capacity = float(GROUP_CAPACITY_KWH[group])
        actual = part["official_target"].to_numpy(float)
        forecast = part["pred"].to_numpy(float)
        valid = np.isfinite(actual) & np.isfinite(forecast) & (
            actual >= 0.10 * capacity
        )
        actual_valid = actual[valid]
        forecast_valid = forecast[valid]
        errors = np.abs(forecast_valid - actual_valid) / capacity
        weight = float(actual_valid.sum())
        nmae, ficr = group_nmae_ficr(actual_valid, forecast_valid, capacity)
        rows.append(
            {
                "scope": "group",
                "variant": variant,
                "group": group,
                "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
                "nmae": float(nmae),
                "ficr": float(ficr),
                "weighted_within_6pct": float(
                    actual_valid[errors <= 0.06].sum() / weight
                ),
                "weighted_within_8pct": float(
                    actual_valid[errors <= 0.08].sum() / weight
                ),
                "n_rows": int(valid.sum()),
            }
        )
    group_rows = pd.DataFrame(rows)
    means = []
    for variant, part in group_rows.groupby("variant", sort=False):
        mean_nmae = float(part["nmae"].mean())
        mean_ficr = float(part["ficr"].mean())
        means.append(
            {
                "scope": "group_equal_mean",
                "variant": variant,
                "group": "__mean__",
                "score": 0.5 * (1.0 - mean_nmae) + 0.5 * mean_ficr,
                "nmae": mean_nmae,
                "ficr": mean_ficr,
                "weighted_within_6pct": float(part["weighted_within_6pct"].mean()),
                "weighted_within_8pct": float(part["weighted_within_8pct"].mean()),
                "n_rows": int(part["n_rows"].sum()),
            }
        )
    return pd.concat([group_rows, pd.DataFrame(means)], ignore_index=True)


def _stack_feature_tables(
    panel_table: pd.DataFrame,
    feature_cols: list[str],
    scada_hourly: pd.DataFrame,
    group: str,
    pred_year: int,
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, tuple[pd.DataFrame, list[str]]], pd.DataFrame]:
    turbines = tuple(GROUP_TURBINE_PREFIXES[group])
    row_years = pd.to_datetime(panel_table["forecast_kst_dtm"]).dt.year.to_numpy()
    train_indices = np.flatnonzero(row_years != pred_year)
    validation_indices = np.flatnonzero(row_years == pred_year)
    target_wind = scada_wind_matrix(panel_table, scada_hourly, turbines)
    stack = build_extra_trees_scada_stack(
        panel_table[feature_cols].to_numpy(np.float32),
        target_wind,
        train_indices,
        validation_indices,
        row_years,
        pd.to_datetime(panel_table["data_available_kst_dtm"]).to_numpy(
            dtype="datetime64[ns]"
        ),
        turbines,
        seed,
        n_estimators=args.et_estimators,
        min_samples_leaf=args.et_min_samples_leaf,
        max_features=args.et_max_features,
        within_year_folds=args.within_year_folds,
        n_jobs=args.n_jobs,
    )
    raw_wind = np.full_like(target_wind, np.nan, dtype=np.float32)
    iso_wind = np.full_like(target_wind, np.nan, dtype=np.float32)
    raw_wind[train_indices] = stack.raw_train_wind
    raw_wind[validation_indices] = stack.raw_validation_wind
    iso_wind[train_indices] = stack.calibrated_train_wind
    iso_wind[validation_indices] = stack.calibrated_validation_wind
    if not np.isfinite(raw_wind).all() or not np.isfinite(iso_wind).all():
        raise ValueError(f"Incomplete stacked wind channels for {group}/val{pred_year}")

    raw_channels = cubic_feature_channels(raw_wind, stack.wind_scales)
    iso_channels = cubic_feature_channels(iso_wind, stack.wind_scales)
    raw_table = panel_table.copy()
    iso_table = panel_table.copy()
    raw_cols = []
    iso_cols = []
    for turbine_index, turbine in enumerate(turbines):
        raw_col = f"{turbine}__et_scada_raw_cube"
        iso_col = f"{turbine}__et_scada_isotonic_cube"
        raw_table[raw_col] = raw_channels[:, turbine_index]
        iso_table[iso_col] = iso_channels[:, turbine_index]
        raw_cols.append(raw_col)
        iso_cols.append(iso_col)
    return (
        {
            STACK_BASELINE_VARIANT: (panel_table, list(feature_cols)),
            STACK_RAW_VARIANT: (raw_table, [*feature_cols, *raw_cols]),
            STACK_WIND_ISO_VARIANT: (iso_table, [*feature_cols, *iso_cols]),
        },
        stack.diagnostics,
    )


def run_et_scada_stack(args: argparse.Namespace) -> None:
    groups = parse_csv(args.groups)
    pred_years = [int(value) for value in parse_csv(args.years)]
    if args.stem == LEGACY_STEM:
        args.stem = ET_SCADA_STEM
    if args.smoke_test:
        groups = groups[:1]
        pred_years = pred_years[:1]
        args.epochs = 2
        args.patience = 2
        args.et_estimators = min(args.et_estimators, 10)
        if not args.stem.endswith("_smoke"):
            args.stem = f"{args.stem}_smoke"
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"mode={ET_SCADA_STACK_MODE} device={device} groups={groups} "
        f"years={pred_years} h={args.hidden_size} layers={args.num_layers} "
        f"ET={args.et_estimators}x leaf={args.et_min_samples_leaf}",
        flush=True,
    )

    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    scada_vestas = pd.read_csv(
        "data/train/scada_vestas_train.csv", encoding="utf-8-sig"
    )
    scada_unison = pd.read_csv(
        "data/train/scada_unison_train.csv", encoding="utf-8-sig"
    )
    scada_raw_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }
    wind_candidates = build_wind_candidate_matrix(ldaps, gfs)
    training_rows: list[dict[str, object]] = []
    selected_epochs: list[int] = []

    print("\n=== shared fixed-epoch discovery: baseline raw panel ===", flush=True)
    for group_index, group in enumerate(groups):
        capacity = float(GROUP_CAPACITY_KWH[group])
        base_features = get_or_build_group_feature_cache(
            ldaps,
            gfs,
            group,
            cache_root=args.cache_root,
            rebuild=args.rebuild_feature_cache,
        )
        targets = build_official_aligned_turbine_targets(
            scada_raw_by_group[group], labels, group
        )
        for pred_year in pred_years:
            if labels.loc[
                labels["kst_dtm"].dt.year.eq(pred_year), group
            ].notna().sum() < 200:
                continue
            train_years = [year for year in YEARS if year != pred_year]
            features, _, _ = build_fold_features(
                base_features,
                wind_candidates,
                targets,
                group,
                pred_year,
                train_years,
                args,
            )
            panel = build_group_local_panel(features, group)
            blocks = make_issue_blocks(
                panel.table,
                labels,
                list(panel.full_feature_cols),
                target_cols=[group],
            )
            arrays = prepare_fold_arrays(
                blocks,
                pred_year,
                capacity,
                args.target_min_output_ratio,
            )
            fold_seed = args.seed + group_index * 100000 + pred_year * 100
            _, stats, history = train_band_fold(
                arrays,
                group,
                args,
                device,
                fold_seed,
                loss_mode="pure68",
            )
            selected_epochs.append(int(stats["best_epoch"]))
            training_rows.extend(
                {
                    "stage": "epoch_discovery_curve",
                    "group": group,
                    "pred_year": int(pred_year),
                    **row,
                }
                for row in history
            )
            print(
                f"  {group} val{pred_year}: best_epoch={stats['best_epoch']} "
                f"score={stats['best_score']:.6f}",
                flush=True,
            )
            del features, panel, blocks, arrays
            gc.collect()
        del base_features, targets
        gc.collect()
    if not selected_epochs:
        raise ValueError("No fixed epochs were discovered")
    fixed_epoch = max(1, int(np.rint(np.median(np.asarray(selected_epochs, float)))))
    training_rows.append(
        {
            "stage": "fixed_epoch_selection",
            "fixed_epoch": int(fixed_epoch),
            "fold_best_epochs": ",".join(map(str, selected_epochs)),
        }
    )
    print(f"shared fixed epoch = {fixed_epoch} from {selected_epochs}", flush=True)

    prediction_parts = []
    print("\n=== fresh fixed-epoch stacked OOF ===", flush=True)
    for group_index, group in enumerate(groups):
        capacity = float(GROUP_CAPACITY_KWH[group])
        base_features = get_or_build_group_feature_cache(
            ldaps,
            gfs,
            group,
            cache_root=args.cache_root,
            rebuild=args.rebuild_feature_cache,
        )
        targets = build_official_aligned_turbine_targets(
            scada_raw_by_group[group], labels, group
        )
        scada_hourly = build_turbine_scada_hourly(scada_raw_by_group[group], group)
        for pred_year in pred_years:
            if labels.loc[
                labels["kst_dtm"].dt.year.eq(pred_year), group
            ].notna().sum() < 200:
                continue
            train_years = [year for year in YEARS if year != pred_year]
            features, _, selections = build_fold_features(
                base_features,
                wind_candidates,
                targets,
                group,
                pred_year,
                train_years,
                args,
            )
            for selection in selections.to_dict("records"):
                training_rows.append(
                    {
                        "stage": "optimal_grid_selection",
                        "group": group,
                        "pred_year": int(pred_year),
                        **selection,
                    }
                )
            panel = build_group_local_panel(features, group)
            fold_seed = args.seed + group_index * 100000 + pred_year * 100
            tables, wind_diagnostics = _stack_feature_tables(
                panel.table,
                list(panel.full_feature_cols),
                scada_hourly,
                group,
                pred_year,
                args,
                fold_seed + 700000,
            )
            for diagnostic in wind_diagnostics.to_dict("records"):
                training_rows.append(
                    {
                        "group": group,
                        "pred_year": int(pred_year),
                        **diagnostic,
                    }
                )

            blocks_by_variant: dict[str, IssueBlockData] = {}
            outer_parts: dict[str, pd.DataFrame] = {}
            for variant in [
                STACK_BASELINE_VARIANT,
                STACK_RAW_VARIANT,
                STACK_WIND_ISO_VARIANT,
            ]:
                table, feature_cols = tables[variant]
                blocks = make_issue_blocks(
                    table,
                    labels,
                    feature_cols,
                    target_cols=[group],
                )
                blocks_by_variant[variant] = blocks
                arrays = prepare_fold_arrays(
                    blocks,
                    pred_year,
                    capacity,
                    args.target_min_output_ratio,
                )
                prediction, stats, _ = train_band_fixed_fold(
                    arrays,
                    group,
                    fixed_epoch,
                    args,
                    device,
                    fold_seed,
                )
                part = _prediction_part(
                    arrays, group, pred_year, prediction, variant
                )
                prediction_parts.append(part)
                outer_parts[variant] = part
                training_rows.append(
                    {
                        "stage": "outer_fixed_tcn",
                        "variant": variant,
                        "group": group,
                        "pred_year": int(pred_year),
                        "n_features": int(len(feature_cols)),
                        **stats,
                    }
                )
                print(
                    f"  {group} val{pred_year} {variant}: "
                    f"score={stats['score']:.6f}",
                    flush=True,
                )

            inner_oof, inner_rows = _nested_power_oof(
                blocks_by_variant[STACK_WIND_ISO_VARIANT],
                group,
                pred_year,
                fixed_epoch,
                args,
                device,
                fold_seed + 500000,
            )
            training_rows.extend(inner_rows)
            calibrated, calibration_row = _calibrate_power_outer(
                inner_oof,
                outer_parts[STACK_WIND_ISO_VARIANT],
                group,
                pred_year,
                args.target_min_output_ratio,
            )
            prediction_parts.append(calibrated)
            training_rows.append(calibration_row)
            print(
                f"  {group} val{pred_year} power_iso: "
                f"alpha={calibration_row['selected_alpha']:.2f}",
                flush=True,
            )
            del features, panel, tables, blocks_by_variant, outer_parts, inner_oof
            gc.collect()
        del base_features, targets, scada_hourly
        gc.collect()

    predictions = pd.concat(prediction_parts, ignore_index=True)
    scores = _stack_score_table(predictions)
    scores["fixed_epoch"] = int(fixed_epoch)
    prefix = args.results_dir / args.stem
    predictions.to_csv(
        f"{prefix}_predictions.csv", index=False, encoding="utf-8-sig"
    )
    scores.to_csv(f"{prefix}_scores.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(training_rows).to_csv(
        f"{prefix}_training.csv", index=False, encoding="utf-8-sig"
    )
    print("\n=== pooled fixed-epoch OOF ===", flush=True)
    print(
        scores.loc[scores["scope"].eq("group_equal_mean")].to_string(index=False),
        flush=True,
    )


def run_legacy(args: argparse.Namespace) -> None:
    groups = parse_csv(args.groups)
    pred_years = [int(value) for value in parse_csv(args.years)]
    variants = parse_csv(args.variants)
    unknown = [variant for variant in variants if variant not in VARIANTS]
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("Epochs and patience must be positive")
    if not 0.0 < args.band_temperature_end <= args.band_temperature_start:
        raise ValueError(
            "Band temperatures must satisfy 0 < end <= start"
        )
    if args.band_min_epochs < 1:
        raise ValueError("--band-min-epochs must be positive")
    candidates = candidate_grid(args)
    if args.smoke_test:
        groups = groups[:1]
        pred_years = pred_years[:1]
        args.epochs = 2
        args.patience = 2

    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} groups={groups} years={pred_years} variants={variants} "
        f"epochs={args.epochs} patience={args.patience} h={args.hidden_size} "
        f"layers={args.num_layers} candidates={len(candidates)}",
        flush=True,
    )

    ldaps = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    scada_vestas = pd.read_csv(
        "data/train/scada_vestas_train.csv", encoding="utf-8-sig"
    )
    scada_unison = pd.read_csv(
        "data/train/scada_unison_train.csv", encoding="utf-8-sig"
    )
    scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }
    wind_candidates = build_wind_candidate_matrix(ldaps, gfs)

    prediction_parts = []
    fold_rows = []
    history_rows = []
    selection_parts = []
    value_parts = []
    for group_index, group in enumerate(groups):
        capacity = float(GROUP_CAPACITY_KWH[group])
        base_features = get_or_build_group_feature_cache(
            ldaps,
            gfs,
            group,
            cache_root=args.cache_root,
            rebuild=args.rebuild_feature_cache,
        )
        turbine_targets = build_official_aligned_turbine_targets(
            scada_by_group[group], labels, group
        )
        for pred_year in pred_years:
            if labels.loc[
                labels["kst_dtm"].dt.year.eq(pred_year), group
            ].notna().sum() < 200:
                continue
            train_years = [year for year in YEARS if year != pred_year]
            features, _, selections = build_fold_features(
                base_features,
                wind_candidates,
                turbine_targets,
                group,
                pred_year,
                train_years,
                args,
            )
            selections["pred_year"] = pred_year
            selection_parts.append(selections)
            panel = build_group_local_panel(features, group)
            feature_cols = list(panel.full_feature_cols)
            blocks = make_issue_blocks(
                panel.table,
                labels,
                feature_cols,
                target_cols=[group],
            )
            arrays = prepare_fold_arrays(
                blocks,
                pred_year,
                capacity,
                args.target_min_output_ratio,
            )
            print(
                f"\n{group} pred_year={pred_year} train={train_years} "
                f"features={len(feature_cols)} train_rows={arrays.n_train_rows} "
                f"validation_rows={arrays.n_validation_rows}",
                flush=True,
            )
            fold_seed = args.seed + group_index * 100000 + pred_year * 100
            for variant in variants:
                if variant == "decision_q":
                    prediction, selected_values, stats, history = train_q_fold(
                        arrays,
                        group,
                        candidates,
                        args,
                        device,
                        fold_seed,
                    )
                elif variant in {"pure_band", "pure6_band"}:
                    prediction, stats, history = train_band_fold(
                        arrays,
                        group,
                        args,
                        device,
                        fold_seed,
                        loss_mode="pure6" if variant == "pure6_band" else "pure68",
                    )
                    selected_values = np.full_like(prediction, np.nan)
                else:
                    target_transform = (
                        "power" if variant == "point_power" else "cuberoot"
                    )
                    prediction, stats, history = train_point_fold(
                        arrays,
                        group,
                        target_transform,
                        args,
                        device,
                        fold_seed,
                    )
                    selected_values = np.full_like(prediction, np.nan)

                actual = arrays.y_validation.reshape(-1) * capacity
                predicted = prediction.reshape(-1) * capacity
                times = arrays.validation_times.reshape(-1)
                part = pd.DataFrame(
                    {
                        "forecast_kst_dtm": pd.to_datetime(times),
                        "variant": variant,
                        "group": group,
                        "pred_year": pred_year,
                        "official_target": actual,
                        "pred": predicted,
                    }
                ).dropna(subset=["official_target", "pred"])
                prediction_parts.append(part)
                if variant == "decision_q":
                    value_parts.append(
                        pd.DataFrame(
                            {
                                "forecast_kst_dtm": pd.to_datetime(times),
                                "group": group,
                                "pred_year": pred_year,
                                "selected_ratio": prediction.reshape(-1),
                                "selected_value": selected_values.reshape(-1),
                            }
                        )
                    )
                score, nmae, ficr = group_score(
                    part["official_target"], part["pred"], group
                )
                fold_rows.append(
                    {
                        "variant": variant,
                        "group": group,
                        "pred_year": pred_year,
                        "score": score,
                        "nmae": nmae,
                        "ficr": ficr,
                        "best_epoch": stats["best_epoch"],
                        "epochs_trained": stats["epochs_trained"],
                        "n_features": len(feature_cols),
                        "n_parameters": stats["n_parameters"],
                        "n_rows": len(part),
                    }
                )
                for row in history:
                    history_rows.append(
                        {
                            "variant": variant,
                            "group": group,
                            "pred_year": pred_year,
                            **row,
                        }
                    )
                print(
                    f"  {variant}: epoch={stats['best_epoch']} "
                    f"trained={stats['epochs_trained']} score={score:.6f} "
                    f"nMAE={nmae:.6f} FiCR={ficr:.6f}",
                    flush=True,
                )
            if args.smoke_test:
                print("smoke test complete after one group-year fold", flush=True)
                return
            del blocks, arrays, panel, features
            gc.collect()
        del base_features, turbine_targets
        gc.collect()

    predictions = pd.concat(prediction_parts, ignore_index=True)
    fold_scores = pd.DataFrame(fold_rows)
    summary, pooled_group_scores = pooled_oof_summary(predictions)
    diagnostics = fold_scores.groupby("variant", as_index=False).agg(
        worst_group_year=("score", "min"),
        std_group_year=("score", lambda values: values.std(ddof=0)),
        median_best_epoch=("best_epoch", "median"),
        n_group_years=("score", "count"),
    )
    summary = summary.merge(diagnostics, on="variant", how="left").sort_values(
        "mean_score", ascending=False
    )
    prefix = args.results_dir / args.stem
    predictions.to_csv(f"{prefix}_predictions.csv", index=False, encoding="utf-8-sig")
    fold_scores.to_csv(f"{prefix}_fold_scores.csv", index=False, encoding="utf-8-sig")
    pooled_group_scores.to_csv(
        f"{prefix}_pooled_group_scores.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(history_rows).to_csv(
        f"{prefix}_training_history.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(selection_parts, ignore_index=True).to_csv(
        f"{prefix}_optimal_grid_selection.csv", index=False, encoding="utf-8-sig"
    )
    if value_parts:
        pd.concat(value_parts, ignore_index=True).to_csv(
            f"{prefix}_q_selections.csv", index=False, encoding="utf-8-sig"
        )
    summary.to_csv(f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== pooled OOF-selected summary ===", flush=True)
    print(summary.to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    if args.within_year_folds < 2:
        raise ValueError("--within-year-folds must be at least 2")
    if args.et_estimators < 1 or args.et_min_samples_leaf < 1:
        raise ValueError("Extra Trees size parameters must be positive")
    if not 0.0 < args.et_max_features <= 1.0:
        raise ValueError("--et-max-features must be in (0, 1]")
    if args.mode == ET_SCADA_STACK_MODE:
        run_et_scada_stack(args)
    else:
        run_legacy(args)


if __name__ == "__main__":
    main()

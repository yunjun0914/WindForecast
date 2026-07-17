from __future__ import annotations

import argparse
import copy
import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
from utils.group_local_panel import build_group_local_panel
from utils.issue_block_dataset import IssueBlockData, make_issue_blocks
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, pooled_oof_summary
from utils.per_turbine_features import get_or_build_group_feature_cache
from utils.per_turbine_optimal_grid_builder import build_wind_candidate_matrix
from utils.per_turbine_scada import build_official_aligned_turbine_targets
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
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument("--rebuild-optimal-grid-cache", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--stem", default="group_decision_q_tcn_oof_v1")
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


def main() -> None:
    args = parse_args()
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


if __name__ == "__main__":
    main()

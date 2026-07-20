from __future__ import annotations

import argparse
import gc
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # type: ignore[no-redef]  # noqa: F401

    sys.modules.setdefault("_bootstrap", _bootstrap)

from experiments.evaluate_per_turbine_pinn_oof import (
    fine_tune_group_backbones as fine_tune_pinn_group,
    prepare_arrays,
    soft_group_score_loss,
    soft_unit_price,
    standardize,
)
from experiments.evaluate_per_turbine_tcn_interval_head_oof import (
    FoldYeoJohnsonTransformer,
    decode_target_numpy,
    decode_target_tensor,
)
from models.per_turbine_pinn import PerTurbineResidualPINN
from models.seqnn import TCNPowerRegressor
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS
from utils.per_turbine_features import get_or_build_group_feature_cache
from utils.per_turbine_optimal_grid import (
    OPTIMAL_GRID_REPLACE_TAG,
    load_optimal_grid_full_features,
    optimal_grid_input_columns,
)
from utils.per_turbine_scada import (
    build_official_aligned_turbine_targets,
    turbine_capacity_kwh,
)
from utils.per_turbine_sequence import (
    SequenceStandardScaler,
    make_per_turbine_sequences,
)
from utils.per_turbine_teacher import (
    TEACHER_FEATURE_COLS,
    get_or_build_full_teacher_cache,
)
from utils.pinn_scada_teacher_config import BEST_SCADA_TEACHER_PINN_GAMMA
from utils.power_curve import GROUP_TURBINE_PREFIXES


PINN_BASE_SHARE = 0.50
TCN_BASE_SHARE = 0.25
BRANCH_WEIGHTS = {"pinn": 0.50, "tree": 0.05, "tcn": 0.45}
PINN_FLOOR = 0.20
FINAL_FLOOR = 0.10


def epoch_map(path: str | Path) -> dict[tuple[str, str], int]:
    table = pd.read_csv(path, encoding="utf-8-sig")
    required = ["group", "turbine_id", "best_epoch"]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"Epoch source {path} missing columns: {missing}")
    medians = table.groupby(["group", "turbine_id"])["best_epoch"].median()
    return {key: max(1, int(round(value))) for key, value in medians.items()}


def component_feature_columns(weather_cols: list[str]) -> dict[str, list[str]]:
    return {
        "pinn": [*weather_cols, *TEACHER_FEATURE_COLS],
        "tcn": list(weather_cols),
    }


def build_turbine_mix(turbine_predictions: pd.DataFrame) -> pd.DataFrame:
    required = {
        "group",
        "pinn_base",
        "pinn_joint",
        "tcn_base",
        "tcn_joint",
    }
    missing = sorted(required - set(turbine_predictions.columns))
    if missing:
        raise ValueError(f"Turbine predictions missing columns: {missing}")

    output = turbine_predictions.copy()
    capacity = output["group"].map(turbine_capacity_kwh).to_numpy(float)
    pinn_base_floor = np.maximum(
        output["pinn_base"].to_numpy(float), PINN_FLOOR * capacity
    )
    pinn_joint_floor = np.maximum(
        output["pinn_joint"].to_numpy(float), PINN_FLOOR * capacity
    )
    output["pinn_mix"] = (
        PINN_BASE_SHARE * pinn_base_floor
        + (1.0 - PINN_BASE_SHARE) * pinn_joint_floor
    )
    output["tcn_mix"] = (
        TCN_BASE_SHARE * output["tcn_base"]
        + (1.0 - TCN_BASE_SHARE) * output["tcn_joint"]
    )
    return output


def branch_submission(
    sample: pd.DataFrame,
    turbine_predictions: pd.DataFrame,
    branch: str,
) -> pd.DataFrame:
    grouped = (
        turbine_predictions.groupby(["forecast_kst_dtm", "group"], as_index=False)[
            branch
        ]
        .sum()
        .pivot(index="forecast_kst_dtm", columns="group", values=branch)
        .reset_index()
    )
    output = sample[["forecast_id", "forecast_kst_dtm"]].merge(
        grouped, on="forecast_kst_dtm", how="left"
    )
    for group in TARGET_COLS:
        output[group] = output[group].clip(0.0, GROUP_CAPACITY_KWH[group])
    return output[["forecast_id", "forecast_kst_dtm", *TARGET_COLS]]


def validate_submission(submission: pd.DataFrame, sample: pd.DataFrame) -> None:
    expected = ["forecast_id", "forecast_kst_dtm", *TARGET_COLS]
    if list(submission.columns) != expected:
        raise ValueError(f"Submission columns differ: {submission.columns.tolist()}")
    if len(submission) != len(sample):
        raise ValueError(f"Submission rows differ: {len(submission)} != {len(sample)}")
    if not submission["forecast_id"].equals(sample["forecast_id"]):
        raise ValueError("forecast_id order differs from sample submission")
    values = submission[TARGET_COLS].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Submission contains non-finite predictions")
    for group in TARGET_COLS:
        if not submission[group].between(0.0, GROUP_CAPACITY_KWH[group]).all():
            raise ValueError(f"Submission values out of range for {group}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pinn-oof-training",
        default="results/per_turbine_pinn_optimal_grid_replace_v1_training.csv",
    )
    parser.add_argument(
        "--tcn-oof-training",
        default=(
            "results/per_turbine_tcn_optgrid_weather_only_"
            "groupjoint_h64_l1_e3_v1_training.csv"
        ),
    )
    parser.add_argument(
        "--tree-submission",
        default="results/submission_tree_lgbm_group_quota65_complete_nested_v1.csv",
    )
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--teacher-trees", type=int, default=80)
    parser.add_argument("--rebuild-full-teacher-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--tcn-window", type=int, default=72)
    parser.add_argument("--tcn-batch-size", type=int, default=512)
    parser.add_argument("--tcn-hidden-size", type=int, default=64)
    parser.add_argument("--tcn-num-layers", type=int, default=1)
    parser.add_argument("--tcn-kernel-size", type=int, default=3)
    parser.add_argument("--tcn-dropout", type=float, default=0.10)
    parser.add_argument(
        "--tcn-feature-transform",
        choices=["standard", "yeo_johnson"],
        default="standard",
    )
    parser.add_argument(
        "--tcn-target-transform",
        choices=["identity", "sqrt"],
        default="identity",
    )
    parser.add_argument("--pinn-batch-size", type=int, default=1024)
    parser.add_argument("--joint-finetune-epochs", type=int, default=3)
    parser.add_argument("--joint-finetune-lr", type=float, default=1e-4)
    parser.add_argument("--joint-finetune-weight-decay", type=float, default=1e-4)
    parser.add_argument("--joint-finetune-batch-size", type=int, default=64)
    parser.add_argument("--joint-anchor-weight", type=float, default=0.03)
    parser.add_argument(
        "--joint-gamma", type=float, default=BEST_SCADA_TEACHER_PINN_GAMMA
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-batch-size", type=int, default=8192)
    parser.add_argument("--backbone", choices=["mlp"], default="mlp")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--residual-amplitude", type=float, default=0.15)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument(
        "--output",
        default="results/submission_jointmix_p50_t5_c45_pb50_cb25_v1.csv",
    )
    parser.add_argument(
        "--diagnostics",
        default="results/submission_jointmix_p50_t5_c45_pb50_cb25_v1_diagnostics.csv",
    )
    parser.add_argument(
        "--turbine-output",
        default="results/jointmix_full_v1_turbine_predictions.csv",
    )
    parser.add_argument(
        "--training-output", default="results/jointmix_full_v1_training.csv"
    )
    parser.add_argument("--compact-output", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def predict_tcn(
    model: TCNPowerRegressor,
    values: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.from_numpy(values[start : start + batch_size]).to(device)
            parts.append(model(batch).cpu().numpy())
    return np.concatenate(parts) if parts else np.empty((0,), dtype=np.float32)


def train_full_tcn(
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    feature_cols: list[str],
    group: str,
    turbine: str,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    set_seed(seed)
    x_train, y_train, official_train, time_train, _ = make_per_turbine_sequences(
        train_table, feature_cols, window=args.tcn_window
    )
    x_test, _, _, time_test, _ = make_per_turbine_sequences(
        test_table, feature_cols, window=args.tcn_window
    )
    keep = (
        np.isfinite(y_train)
        & np.isfinite(official_train)
        & (official_train >= GROUP_CAPACITY_KWH[group] * 0.10)
    )
    x_train = x_train[keep]
    y_train = y_train[keep]
    official_train = official_train[keep].astype(np.float32)
    time_train = pd.to_datetime(np.asarray(time_train)[keep])

    scaler = SequenceStandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    capacity = float(turbine_capacity_kwh(group))
    y_norm = np.clip(y_train / capacity, 0.0, 1.0).astype(np.float32)
    if args.tcn_target_transform == "sqrt":
        y_norm = np.sqrt(y_norm).astype(np.float32)
    weights = (
        0.5
        + np.sqrt(
            np.clip(
                official_train / float(GROUP_CAPACITY_KWH[group]), 0.0, 1.0
            )
        )
    ).astype(np.float32)
    dataset = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_norm),
        torch.from_numpy(weights),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.tcn_batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    model = TCNPowerRegressor(
        input_size=len(feature_cols),
        hidden_size=args.tcn_hidden_size,
        num_layers=args.tcn_num_layers,
        kernel_size=args.tcn_kernel_size,
        dropout=args.tcn_dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for xb, yb, wb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(xb)
            loss = (torch.abs(prediction - yb) * wb).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

    train_pretrained = predict_tcn(model, x_train, device, args.eval_batch_size)
    test_prediction = predict_tcn(model, x_test, device, args.eval_batch_size)
    output = pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(time_test),
            "group": group,
            "turbine_id": turbine,
            "pred": decode_target_numpy(
                test_prediction, args.tcn_target_transform
            )
            * capacity,
        }
    )
    artifact = {
        "turbine_id": turbine,
        "model_state": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "train_time": time_train,
        "test_time": pd.to_datetime(time_test),
        "train_features": x_train,
        "test_features": x_test,
        "train_official": official_train,
        "train_pretrained": train_pretrained.astype(np.float32),
    }
    del model, optimizer, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output, artifact


def train_full_pinn(
    train_table: pd.DataFrame,
    test_table: pd.DataFrame,
    feature_cols: list[str],
    group: str,
    turbine: str,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object], dict[str, float]]:
    set_seed(seed)
    capacity = float(turbine_capacity_kwh(group))
    group_capacity = float(GROUP_CAPACITY_KWH[group])
    train = prepare_arrays(train_table, feature_cols, capacity)
    test = prepare_arrays(test_table, feature_cols, capacity)
    keep = (
        np.isfinite(train["target"])
        & np.isfinite(train["official"])
        & (train["official"] >= group_capacity * 0.10)
    )
    x_train, x_test, scaler_mean, scaler_std = standardize(
        train["features"][keep], test["features"]
    )
    y_train = (train["target"][keep] / capacity).astype(np.float32)
    official_train = train["official"][keep].astype(np.float32)
    physical_train = train["physical"][keep].astype(np.float32)
    lead_train = train["lead"][keep]
    month_train = train["month"][keep]
    weights = (
        0.5 + np.sqrt(np.clip(official_train / group_capacity, 0.0, 1.0))
    ).astype(np.float32)
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
        batch_size=args.pinn_batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    model = PerTurbineResidualPINN(
        input_size=len(feature_cols),
        hidden_size=args.hidden_size,
        residual_amplitude=args.residual_amplitude,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for xb, pb, lb, mb, yb, wb in loader:
            xb = xb.to(device, non_blocking=True)
            pb = pb.to(device, non_blocking=True)
            lb = lb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction, pieces = model(xb, pb, lb, mb, return_parts=True)
            error_rate = torch.abs(prediction - yb)
            weighted_l1 = (error_rate * wb).mean()
            price = soft_unit_price(error_rate, BEST_SCADA_TEACHER_PINN_GAMMA)
            ficr_soft = (yb * price).sum() / torch.clamp(
                (yb * 4.0).sum(), min=1e-6
            )
            data_loss = 0.5 * weighted_l1 + 0.5 * (1.0 - ficr_soft)
            anchor = (
                prediction - torch.clamp(pieces["physical"], 0.0, 1.0)
            ).pow(2).mean()
            regularization = model.regularization()
            loss = (
                data_loss
                + 0.03 * anchor
                + 0.001 * regularization["residual_l2"]
                + 0.001 * regularization["bias_l2"]
                + 0.01 * regularization["scale_l2"]
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

    model.eval()

    def predict(
        features: np.ndarray,
        physical: np.ndarray,
        lead: np.ndarray,
        month: np.ndarray,
    ) -> np.ndarray:
        parts = []
        with torch.no_grad():
            for start in range(0, len(features), args.eval_batch_size):
                stop = start + args.eval_batch_size
                parts.append(
                    model(
                        torch.from_numpy(features[start:stop]).to(device),
                        torch.from_numpy(physical[start:stop]).to(device),
                        torch.from_numpy(lead[start:stop]).to(device),
                        torch.from_numpy(month[start:stop]).to(device),
                    )
                    .cpu()
                    .numpy()
                )
        return np.concatenate(parts) if parts else np.empty((0,), dtype=np.float32)

    train_pretrained = predict(
        x_train, physical_train, lead_train, month_train
    ).astype(np.float32)
    test_prediction = predict(
        x_test,
        test["physical"].astype(np.float32),
        test["lead"],
        test["month"],
    )
    output = pd.DataFrame(
        {
            "forecast_kst_dtm": pd.to_datetime(test["time"]),
            "group": group,
            "turbine_id": turbine,
            "pred": np.clip(test_prediction, 0.0, 1.0) * capacity,
        }
    )
    artifact = {
        "turbine_id": turbine,
        "model_state": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
        "scaler_mean": scaler_mean,
        "scaler_std": scaler_std,
        "train_time": pd.to_datetime(train["time"][keep]),
        "val_time": pd.to_datetime(test["time"]),
        "train_features": x_train,
        "val_features": x_test,
        "train_physical": physical_train,
        "val_physical": test["physical"].astype(np.float32),
        "train_lead": lead_train,
        "val_lead": test["lead"],
        "train_month": month_train,
        "val_month": test["month"],
        "train_official": official_train,
        "train_pretrained": train_pretrained,
    }
    diagnostics = {
        "physics_scale": float(model.physics_scale.detach().cpu()),
        "n_train": int(keep.sum()),
    }
    del model, optimizer, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return output, artifact, diagnostics


def common_times(
    artifacts: list[dict[str, object]], split: str
) -> pd.DatetimeIndex:
    common = pd.DatetimeIndex(artifacts[0][f"{split}_time"])
    for artifact in artifacts[1:]:
        common = common.intersection(
            pd.DatetimeIndex(artifact[f"{split}_time"]), sort=False
        )
    return common.sort_values()


def aligned(
    artifact: dict[str, object],
    split: str,
    field: str,
    times: pd.DatetimeIndex,
) -> np.ndarray:
    source_times = pd.DatetimeIndex(artifact[f"{split}_time"])
    positions = source_times.get_indexer(times)
    if np.any(positions < 0):
        raise ValueError(
            f"Failed to align {split} {field} for {artifact['turbine_id']}"
        )
    return np.asarray(artifact[field])[positions]


def fine_tune_tcn_group(
    artifacts: list[dict[str, object]],
    feature_cols: list[str],
    group: str,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    train_time = common_times(artifacts, "train")
    test_time = common_times(artifacts, "test")
    official = aligned(
        artifacts[0], "train", "train_official", train_time
    ).astype(np.float32)
    for artifact in artifacts[1:]:
        other = aligned(artifact, "train", "train_official", train_time)
        if not np.allclose(official, other, equal_nan=True):
            raise ValueError(f"Official group target mismatch for {group}")

    train_inputs = [
        aligned(artifact, "train", "train_features", train_time).astype(np.float32)
        for artifact in artifacts
    ]
    test_inputs = [
        aligned(artifact, "test", "test_features", test_time).astype(np.float32)
        for artifact in artifacts
    ]
    pretrained = np.stack(
        [
            aligned(artifact, "train", "train_pretrained", train_time).astype(
                np.float32
            )
            for artifact in artifacts
        ],
        axis=1,
    )
    models = []
    for artifact in artifacts:
        model = TCNPowerRegressor(
            input_size=len(feature_cols),
            hidden_size=args.tcn_hidden_size,
            num_layers=args.tcn_num_layers,
            kernel_size=args.tcn_kernel_size,
            dropout=args.tcn_dropout,
        ).to(device)
        model.load_state_dict(artifact["model_state"])
        models.append(model)

    parameters = [parameter for model in models for parameter in model.parameters()]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.joint_finetune_lr,
        weight_decay=args.joint_finetune_weight_decay,
    )
    target = np.clip(
        official / float(GROUP_CAPACITY_KWH[group]), 0.0, 1.0
    ).astype(np.float32)
    turbine_ratio = float(turbine_capacity_kwh(group)) / float(
        GROUP_CAPACITY_KWH[group]
    )
    generator = np.random.default_rng(args.seed + 2025000 + len(models))
    history = []
    for epoch in range(1, args.joint_finetune_epochs + 1):
        for model in models:
            model.train()
        losses = []
        score_losses = []
        anchors = []
        indices = generator.permutation(len(train_time))
        for start in range(0, len(indices), args.joint_finetune_batch_size):
            batch_indices = indices[
                start : start + args.joint_finetune_batch_size
            ]
            optimizer.zero_grad(set_to_none=True)
            turbine_model_output = torch.stack(
                [
                    model(torch.from_numpy(values[batch_indices]).to(device))
                    for model, values in zip(models, train_inputs)
                ],
                dim=1,
            )
            turbine_prediction = decode_target_tensor(
                turbine_model_output, args.tcn_target_transform
            )
            group_prediction = turbine_prediction.sum(dim=1) * turbine_ratio
            target_batch = torch.from_numpy(target[batch_indices]).to(device)
            score_loss, _, _ = soft_group_score_loss(
                target_batch, group_prediction, args.joint_gamma
            )
            pretrained_batch = torch.from_numpy(pretrained[batch_indices]).to(device)
            anchor = (turbine_model_output - pretrained_batch).pow(2).mean()
            loss = score_loss + args.joint_anchor_weight * anchor
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            score_losses.append(float(score_loss.detach().cpu()))
            anchors.append(float(anchor.detach().cpu()))
        history.append(
            {
                "branch": "tcn",
                "group": group,
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "score_loss": float(np.mean(score_losses)),
                "anchor_mse": float(np.mean(anchors)),
                "n_train": len(train_time),
            }
        )

    prediction_parts = [[] for _ in models]
    for model in models:
        model.eval()
    with torch.no_grad():
        for start in range(0, len(test_time), args.eval_batch_size):
            stop = start + args.eval_batch_size
            for index, (model, values) in enumerate(zip(models, test_inputs)):
                prediction_parts[index].append(
                    decode_target_tensor(
                        model(torch.from_numpy(values[start:stop]).to(device)),
                        args.tcn_target_transform,
                    )
                    .cpu()
                    .numpy()
                )

    capacity = float(turbine_capacity_kwh(group))
    rows = []
    for artifact, parts in zip(artifacts, prediction_parts):
        rows.append(
            pd.DataFrame(
                {
                    "forecast_kst_dtm": test_time,
                    "group": group,
                    "turbine_id": artifact["turbine_id"],
                    "pred": np.concatenate(parts) * capacity,
                }
            )
        )
    del models, optimizer, train_inputs, test_inputs, pretrained
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return pd.concat(rows, ignore_index=True), history


def load_submission(path: str | Path, sample: pd.DataFrame) -> pd.DataFrame:
    submission = pd.read_csv(path, encoding="utf-8-sig")
    submission["forecast_kst_dtm"] = pd.to_datetime(
        submission["forecast_kst_dtm"]
    )
    validate_submission(submission, sample)
    return submission


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} pinn_base_share={PINN_BASE_SHARE} "
        f"tcn_base_share={TCN_BASE_SHARE} weights={BRANCH_WEIGHTS} "
        f"tcn_feature_transform={args.tcn_feature_transform} "
        f"tcn_target_transform={args.tcn_target_transform}",
        flush=True,
    )
    pinn_epochs = epoch_map(args.pinn_oof_training)
    tcn_epochs = epoch_map(args.tcn_oof_training)

    ldaps_train = pd.read_csv("data/train/ldaps_train.csv", encoding="utf-8-sig")
    gfs_train = pd.read_csv("data/train/gfs_train.csv", encoding="utf-8-sig")
    ldaps_test = pd.read_csv("data/test/ldaps_test.csv", encoding="utf-8-sig")
    gfs_test = pd.read_csv("data/test/gfs_test.csv", encoding="utf-8-sig")
    labels = pd.read_csv("data/train/train_labels.csv", encoding="utf-8-sig")
    labels["kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    sample = pd.read_csv("data/sample_submission.csv", encoding="utf-8-sig")
    sample["forecast_kst_dtm"] = pd.to_datetime(sample["forecast_kst_dtm"])
    scada_vestas = pd.read_csv(
        "data/train/scada_vestas_train.csv", encoding="utf-8-sig"
    )
    scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": pd.read_csv(
            "data/train/scada_vestas_train.csv", encoding="utf-8-sig"
        ),
        "kpx_group_3": pd.read_csv(
            "data/train/scada_unison_train.csv", encoding="utf-8-sig"
        ),
    }

    branch_parts = {
        name: [] for name in ("pinn_base", "pinn_joint", "tcn_base", "tcn_joint")
    }
    training_rows = []
    for group_index, group in enumerate(TARGET_COLS):
        print(f"\n=== full group-joint {group} ===", flush=True)
        base_train = get_or_build_group_feature_cache(
            ldaps_train, gfs_train, group, cache_root=args.cache_root
        )
        base_test = get_or_build_group_feature_cache(
            ldaps_test,
            gfs_test,
            group,
            cache_root=args.cache_root,
            cache_tag="test",
        )
        train_features, test_features = load_optimal_grid_full_features(
            base_train, base_test, args.cache_root, group
        )
        weather_cols = optimal_grid_input_columns(group)
        feature_cols = component_feature_columns(weather_cols)
        targets = build_official_aligned_turbine_targets(
            scada_by_group[group], labels, group
        )
        teacher = get_or_build_full_teacher_cache(
            train_features=train_features,
            test_features=test_features,
            targets=targets,
            scada=scada_by_group[group],
            group=group,
            cache_root=args.cache_root,
            rebuild=args.rebuild_full_teacher_cache,
            n_estimators=args.teacher_trees,
            input_feature_cols=weather_cols,
            cache_tag=OPTIMAL_GRID_REPLACE_TAG,
        )
        keys = ["forecast_kst_dtm", "turbine_id"]
        train_teacher = teacher.loc[
            teacher["split"].eq("train_oob"), keys + TEACHER_FEATURE_COLS
        ]
        test_teacher = teacher.loc[
            teacher["split"].eq("test"), keys + TEACHER_FEATURE_COLS
        ]
        target_one = targets[keys + ["turbine_target"]]
        label_one = labels[["kst_dtm", group]].rename(
            columns={
                "kst_dtm": "forecast_kst_dtm",
                group: "official_target",
            }
        )
        train_table = train_features.merge(
            target_one, on=keys, how="left"
        ).merge(label_one, on="forecast_kst_dtm", how="left")
        test_table = test_features.copy()
        test_table["turbine_target"] = np.nan
        test_table["official_target"] = np.nan
        pinn_train = train_table.merge(train_teacher, on=keys, how="inner")
        pinn_test = test_table.merge(test_teacher, on=keys, how="inner")
        for table in (train_table, test_table, pinn_train, pinn_test):
            table["forecast_kst_dtm"] = pd.to_datetime(table["forecast_kst_dtm"])
            table["data_available_kst_dtm"] = pd.to_datetime(
                table["data_available_kst_dtm"]
            )

        tcn_train_table = train_table
        tcn_test_table = test_table
        if args.tcn_feature_transform == "yeo_johnson":
            transformer = FoldYeoJohnsonTransformer().fit(
                train_table, feature_cols["tcn"]
            )
            tcn_train_table = transformer.transform(train_table)
            tcn_test_table = transformer.transform(test_table)
            print(
                f"  TCN Yeo-Johnson transformed="
                f"{len(transformer.lambdas)}/{len(feature_cols['tcn'])}",
                flush=True,
            )

        pinn_artifacts = []
        tcn_artifacts = []
        pinn_base_parts = []
        tcn_base_parts = []
        for turbine_index, turbine in enumerate(GROUP_TURBINE_PREFIXES[group]):
            pinn_train_one = pinn_train[pinn_train["turbine_id"].eq(turbine)].copy()
            pinn_test_one = pinn_test[pinn_test["turbine_id"].eq(turbine)].copy()
            tcn_train_one = tcn_train_table[
                tcn_train_table["turbine_id"].eq(turbine)
            ].copy()
            tcn_test_one = tcn_test_table[
                tcn_test_table["turbine_id"].eq(turbine)
            ].copy()
            for table in (
                pinn_train_one,
                pinn_test_one,
                tcn_train_one,
                tcn_test_one,
            ):
                table.sort_values("forecast_kst_dtm", inplace=True)
                table.reset_index(drop=True, inplace=True)

            offset = group_index * 100 + turbine_index
            pinn_output, pinn_artifact, pinn_diag = train_full_pinn(
                pinn_train_one,
                pinn_test_one,
                feature_cols["pinn"],
                group,
                turbine,
                pinn_epochs[(group, turbine)],
                args,
                device,
                args.seed + 2000 + offset,
            )
            tcn_output, tcn_artifact = train_full_tcn(
                tcn_train_one,
                tcn_test_one,
                feature_cols["tcn"],
                group,
                turbine,
                tcn_epochs[(group, turbine)],
                args,
                device,
                args.seed + 1000 + offset,
            )
            pinn_base_parts.append(pinn_output)
            tcn_base_parts.append(tcn_output)
            pinn_artifacts.append(pinn_artifact)
            tcn_artifacts.append(tcn_artifact)
            training_rows.append(
                {
                    "branch": "base",
                    "group": group,
                    "turbine_id": turbine,
                    "pinn_epochs": pinn_epochs[(group, turbine)],
                    "tcn_epochs": tcn_epochs[(group, turbine)],
                    **pinn_diag,
                }
            )
            print(
                f"  {turbine}: pinn={pinn_epochs[(group, turbine)]} "
                f"tcn={tcn_epochs[(group, turbine)]}",
                flush=True,
            )

        pinn_joint, pinn_history = fine_tune_pinn_group(
            artifacts=pinn_artifacts,
            feature_cols=feature_cols["pinn"],
            group=group,
            pred_year=2025,
            args=args,
            device=device,
        )
        tcn_joint, tcn_history = fine_tune_tcn_group(
            tcn_artifacts, feature_cols["tcn"], group, args, device
        )
        for row in pinn_history:
            training_rows.append({"branch": "pinn_joint", **row})
        training_rows.extend(tcn_history)
        branch_parts["pinn_base"].append(
            pd.concat(pinn_base_parts, ignore_index=True)
        )
        branch_parts["pinn_joint"].append(pinn_joint)
        branch_parts["tcn_base"].append(
            pd.concat(tcn_base_parts, ignore_index=True)
        )
        branch_parts["tcn_joint"].append(tcn_joint)
        del pinn_artifacts, tcn_artifacts
        gc.collect()

    branches = {
        name: pd.concat(parts, ignore_index=True)
        for name, parts in branch_parts.items()
    }
    turbine = branches["pinn_base"].rename(columns={"pred": "pinn_base"})
    for name in ("pinn_joint", "tcn_base", "tcn_joint"):
        turbine = turbine.merge(
            branches[name][
                ["forecast_kst_dtm", "group", "turbine_id", "pred"]
            ].rename(columns={"pred": name}),
            on=["forecast_kst_dtm", "group", "turbine_id"],
            validate="one_to_one",
        )
    turbine = build_turbine_mix(turbine)

    if not args.compact_output:
        turbine_path = Path(args.turbine_output)
        turbine_path.parent.mkdir(parents=True, exist_ok=True)
        turbine.to_csv(turbine_path, index=False, encoding="utf-8-sig")
        pd.DataFrame(training_rows).to_csv(
            args.training_output, index=False, encoding="utf-8-sig"
        )

    pinn_submission = branch_submission(sample, turbine, "pinn_mix")
    tcn_submission = branch_submission(sample, turbine, "tcn_mix")
    validate_submission(pinn_submission, sample)
    validate_submission(tcn_submission, sample)
    tree_submission = load_submission(args.tree_submission, sample)
    output = sample[["forecast_id", "forecast_kst_dtm"]].copy()
    diagnostics = []
    for group in TARGET_COLS:
        capacity = float(GROUP_CAPACITY_KWH[group])
        raw = (
            BRANCH_WEIGHTS["pinn"] * pinn_submission[group].to_numpy(float)
            + BRANCH_WEIGHTS["tree"] * tree_submission[group].to_numpy(float)
            + BRANCH_WEIGHTS["tcn"] * tcn_submission[group].to_numpy(float)
        )
        final = np.clip(raw, FINAL_FLOOR * capacity, capacity)
        output[group] = final
        diagnostics.append(
            {
                "group": group,
                "rows": len(final),
                "pinn_mean": float(pinn_submission[group].mean()),
                "tree_mean": float(tree_submission[group].mean()),
                "tcn_mean": float(tcn_submission[group].mean()),
                "final_mean": float(np.mean(final)),
                "final_min": float(np.min(final)),
                "final_max": float(np.max(final)),
                "final_floor_raised": int(np.sum(raw < FINAL_FLOOR * capacity)),
                "capacity_clipped": int(np.sum(raw > capacity)),
            }
        )

    validate_submission(output, sample)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")
    if not args.compact_output:
        diagnostics_path = Path(args.diagnostics)
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(diagnostics).to_csv(
            diagnostics_path, index=False, encoding="utf-8-sig"
        )
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    print(
        f"saved {output_path}: rows={len(output)} sha256={digest}", flush=True
    )
    print(pd.DataFrame(diagnostics).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

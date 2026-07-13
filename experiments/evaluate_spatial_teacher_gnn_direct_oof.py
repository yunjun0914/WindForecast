from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import _bootstrap  # noqa: F401
from experiments.evaluate_spatial_teacher_gnn_oof import (
    TEACHER_FEATURE_COLS,
    YEARS,
    align_weather,
    build_fold_features,
    build_graph,
    group_aggregation,
    load_labels,
    load_source,
    load_teacher_fold,
    load_turbine_targets,
    node_static_features,
    score_predictions,
    turbine_catalog,
)
from models.spatial_teacher_gnn import SpatialTeacherGNN, TeacherTurbineMLP
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr
from utils.pinn_losses import soft_unit_price
from utils.per_turbine_terrain import (
    TERRAIN_VARIANTS,
    append_gnn_turbine_terrain_features,
)


WEATHER_SIZE = 10
TEACHER_WIND_SLICE = slice(WEATHER_SIZE, WEATHER_SIZE + 4)
TEACHER_POWER_INDEX = WEATHER_SIZE + TEACHER_FEATURE_COLS.index("teacher_power_curve_kwh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ldaps", default="data/train/ldaps_train.csv")
    parser.add_argument("--gfs", default="data/train/gfs_train.csv")
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument("--scada-vestas", default="data/train/scada_vestas_train.csv")
    parser.add_argument("--scada-unison", default="data/train/scada_unison_train.csv")
    parser.add_argument(
        "--teacher-cache", type=Path, default=Path("cache/per_turbine_teacher_v1")
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--stem", default="spatial_teacher_gnn_direct_v2")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--edge-dropout", type=float, default=0.15)
    parser.add_argument("--teacher-dropout", type=float, default=0.20)
    parser.add_argument("--aux-weight", type=float, default=0.20)
    parser.add_argument(
        "--loss-mode", choices=["weighted_l1", "metric_soft"], default="weighted_l1"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument(
        "--terrain-variant", choices=TERRAIN_VARIANTS, default="none"
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
    parser.add_argument(
        "--model-variants",
        default="direct_teacher_mlp,direct_spatial_gnn",
        help="Comma-separated subset of direct_teacher_mlp,direct_spatial_gnn.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(
    variant: str,
    node_size: int,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    turbine_indices: np.ndarray,
    turbine_group_index: np.ndarray,
    aggregation: np.ndarray,
    args: argparse.Namespace,
) -> torch.nn.Module:
    common = {
        "node_size": node_size,
        "turbine_indices": torch.from_numpy(turbine_indices.copy()),
        "turbine_group_index": torch.from_numpy(turbine_group_index.copy()),
        "group_aggregation": torch.from_numpy(aggregation.copy()),
        "hidden_size": args.hidden_size,
        "dropout": args.dropout,
        "direct_output": True,
    }
    if variant == "direct_teacher_mlp":
        return TeacherTurbineMLP(**common)
    if variant == "direct_spatial_gnn":
        return SpatialTeacherGNN(
            **common,
            edge_size=edge_attr.shape[1],
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_layers=args.num_layers,
            edge_dropout=args.edge_dropout,
        )
    raise ValueError(f"Unknown model variant: {variant}")


def score_normalized(
    actual: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> float:
    scores = []
    for group_index in range(len(TARGET_COLS)):
        keep = mask[:, group_index]
        if not keep.any():
            continue
        nmae, ficr = group_nmae_ficr(
            actual[keep, group_index], prediction[keep, group_index], capacity=1.0
        )
        scores.append(0.5 * (1.0 - nmae) + 0.5 * ficr)
    return float(np.mean(scores)) if scores else float("nan")


def make_loader(
    features: np.ndarray,
    teacher_base: np.ndarray,
    group_target: np.ndarray,
    group_mask: np.ndarray,
    turbine_target: np.ndarray,
    turbine_mask: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features[indices]),
        torch.from_numpy(teacher_base[indices]),
        torch.from_numpy(group_target[indices].astype(np.float32)),
        torch.from_numpy(group_mask[indices]),
        torch.from_numpy(turbine_target[indices].astype(np.float32)),
        torch.from_numpy(turbine_mask[indices]),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        pin_memory=pin_memory,
    )


def training_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    turbine_indices: np.ndarray,
    teacher_dropout: float,
    aux_weight: float,
    loss_mode: str,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_rows = 0
    turbine_index_tensor = torch.from_numpy(turbine_indices).to(device)
    for xb, baseb, groupb, group_maskb, turbineb, turbine_maskb in loader:
        xb = xb.to(device, non_blocking=True)
        baseb = baseb.to(device, non_blocking=True)
        groupb = groupb.to(device, non_blocking=True)
        group_maskb = group_maskb.to(device, non_blocking=True)
        turbineb = turbineb.to(device, non_blocking=True)
        turbine_maskb = turbine_maskb.to(device, non_blocking=True)
        if teacher_dropout > 0:
            xb = xb.clone()
            keep = (
                torch.rand(len(xb), len(turbine_indices), 1, device=device)
                >= teacher_dropout
            ).to(xb.dtype)
            turbine_features = xb[:, turbine_index_tensor]
            turbine_features[:, :, TEACHER_WIND_SLICE] *= keep
            xb[:, turbine_index_tensor] = turbine_features
        optimizer.zero_grad(set_to_none=True)
        turbine_pred, group_pred = model(xb, baseb)
        if loss_mode == "weighted_l1":
            group_weights = 0.5 + torch.sqrt(torch.clamp(groupb, 0.0, 1.0))
            group_loss = (
                torch.abs(group_pred - groupb) * group_weights * group_maskb
            ).sum() / group_maskb.sum().clamp_min(1)
        elif loss_mode == "metric_soft":
            group_losses = []
            for group_index in range(groupb.shape[1]):
                valid = group_maskb[:, group_index]
                if not valid.any():
                    continue
                target = groupb[valid, group_index]
                error_rate = torch.abs(group_pred[valid, group_index] - target)
                nmae = error_rate.mean()
                price = soft_unit_price(error_rate)
                ficr_soft = (target * price).sum() / (target * 4.0).sum().clamp_min(1e-6)
                group_losses.append(0.5 * nmae + 0.5 * (1.0 - ficr_soft))
            if not group_losses:
                raise ValueError("Metric loss batch contains no valid group targets")
            group_loss = torch.stack(group_losses).mean()
        else:
            raise ValueError(f"Unknown loss mode: {loss_mode}")
        turbine_loss = (
            torch.abs(turbine_pred - turbineb) * turbine_maskb
        ).sum() / turbine_maskb.sum().clamp_min(1)
        loss = group_loss + aux_weight * turbine_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.detach()) * len(xb)
        total_rows += len(xb)
    return total_loss / max(total_rows, 1)


@torch.no_grad()
def predict_group(
    model: torch.nn.Module,
    features: np.ndarray,
    teacher_base: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    parts = []
    for start in range(0, len(indices), batch_size):
        selection = indices[start : start + batch_size]
        xb = torch.from_numpy(features[selection]).to(device)
        baseb = torch.from_numpy(teacher_base[selection]).to(device)
        _, group_pred = model(xb, baseb)
        parts.append(group_pred.cpu().numpy())
    return np.concatenate(parts)


def inner_epoch_selection(
    variant: str,
    outer_year: int,
    times: pd.DatetimeIndex,
    features: np.ndarray,
    teacher_base: np.ndarray,
    group_target: np.ndarray,
    group_mask: np.ndarray,
    turbine_target: np.ndarray,
    turbine_mask: np.ndarray,
    model_builder,
    turbine_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[int, pd.DataFrame]:
    inner_rows = []
    train_years = [year for year in YEARS if year != outer_year]
    for inner_index, inner_year in enumerate(train_years):
        fit_indices = np.flatnonzero(
            (times.year != outer_year)
            & (times.year != inner_year)
            & group_mask.any(axis=1)
        )
        eval_indices = np.flatnonzero((times.year == inner_year) & group_mask.any(axis=1))
        seed = args.seed + 10000 * outer_year + 100 * inner_year + inner_index
        seed_everything(seed)
        model = model_builder().to(device)
        loader = make_loader(
            features,
            teacher_base,
            group_target,
            group_mask,
            turbine_target,
            turbine_mask,
            fit_indices,
            args.batch_size,
            seed,
            device.type == "cuda",
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        for epoch in range(1, args.max_epochs + 1):
            loss = training_epoch(
                model,
                loader,
                optimizer,
                turbine_indices,
                args.teacher_dropout,
                args.aux_weight,
                args.loss_mode,
                device,
            )
            prediction = predict_group(
                model,
                features,
                teacher_base,
                eval_indices,
                args.eval_batch_size,
                device,
            )
            score = score_normalized(
                group_target[eval_indices], prediction, group_mask[eval_indices]
            )
            inner_rows.append(
                {
                    "variant": variant,
                    "outer_year": outer_year,
                    "inner_year": inner_year,
                    "epoch": epoch,
                    "train_loss": loss,
                    "inner_score": score,
                }
            )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    curves = pd.DataFrame(inner_rows)
    epoch_mean = curves.groupby("epoch")["inner_score"].mean().sort_index()
    smoothed = epoch_mean.rolling(5, min_periods=1).mean()
    selected_epoch = int(smoothed.idxmax())
    return selected_epoch, curves


def final_fit(
    variant: str,
    outer_year: int,
    selected_epoch: int,
    times: pd.DatetimeIndex,
    features: np.ndarray,
    teacher_base: np.ndarray,
    group_target: np.ndarray,
    group_mask: np.ndarray,
    turbine_target: np.ndarray,
    turbine_mask: np.ndarray,
    model_builder,
    turbine_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, object]]:
    train_indices = np.flatnonzero((times.year != outer_year) & group_mask.any(axis=1))
    val_indices = np.flatnonzero(times.year == outer_year)
    seed = args.seed + 100000 * outer_year + (0 if variant.endswith("mlp") else 1)
    seed_everything(seed)
    model = model_builder().to(device)
    loader = make_loader(
        features,
        teacher_base,
        group_target,
        group_mask,
        turbine_target,
        turbine_mask,
        train_indices,
        args.batch_size,
        seed,
        device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    started = time.time()
    final_loss = np.nan
    for epoch in range(1, selected_epoch + 1):
        final_loss = training_epoch(
            model,
            loader,
            optimizer,
            turbine_indices,
            args.teacher_dropout,
            args.aux_weight,
            args.loss_mode,
            device,
        )
    prediction = predict_group(
        model,
        features,
        teacher_base,
        val_indices,
        args.eval_batch_size,
        device,
    )
    stats = {
        "variant": variant,
        "outer_year": outer_year,
        "selected_epoch": selected_epoch,
        "final_train_loss": final_loss,
        "seconds": time.time() - started,
        "n_train_times": len(train_indices),
        "n_val_times": len(val_indices),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction, stats


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    ldaps_data = load_source(args.ldaps, "ldaps")
    gfs_data = load_source(args.gfs, "gfs")
    times, weather, weather_grids, source_index, lead_hour = align_weather(
        ldaps_data, gfs_data
    )
    year_keep = np.isin(times.year, YEARS)
    times = times[year_keep]
    weather = weather[year_keep]
    lead_hour = lead_hour[year_keep]
    turbines = turbine_catalog()
    static, coordinates = node_static_features(weather_grids, source_index, turbines)
    edge_index, edge_attr = build_graph(coordinates, source_index, turbines)
    labels_frame, actual = load_labels(args.labels, times)
    capacities = np.array([GROUP_CAPACITY_KWH[group] for group in TARGET_COLS], dtype=float)
    group_target = actual / capacities[None, :]
    turbine_target = load_turbine_targets(args, labels_frame, times, turbines)
    aggregation, turbine_group_index = group_aggregation(turbines)
    turbine_indices = np.arange(len(weather_grids), len(weather_grids) + len(turbines))
    model_variants = [part.strip() for part in args.model_variants.split(",") if part.strip()]
    invalid_variants = set(model_variants) - {
        "direct_teacher_mlp",
        "direct_spatial_gnn",
    }
    if invalid_variants or not model_variants:
        raise ValueError(f"Invalid model variants: {sorted(invalid_variants)}")

    prediction_parts = []
    inner_parts = []
    training_rows = []
    for outer_year in YEARS:
        print(f"\n=== outer_year={outer_year} ===", flush=True)
        teacher, teacher_available = load_teacher_fold(
            args.teacher_cache, outer_year, times, turbines
        )
        train_time_mask = times.year != outer_year
        features, teacher_base_kwh = build_fold_features(
            times,
            lead_hour,
            weather,
            static,
            teacher,
            teacher_available,
            train_time_mask,
        )
        features = append_gnn_turbine_terrain_features(
            features=features,
            variant=args.terrain_variant,
            turbine_ids=turbines["turbine_id"].tolist(),
            n_weather=len(weather_grids),
            train_time_mask=train_time_mask,
            teacher_available=teacher_available,
            teacher_wd_sin=teacher[:, :, TEACHER_FEATURE_COLS.index("teacher_wd_sin")],
            teacher_wd_cos=teacher[:, :, TEACHER_FEATURE_COLS.index("teacher_wd_cos")],
            turbine_cache=args.terrain_turbine_cache,
            sector_cache=args.terrain_sector_cache,
        )
        features[:, :, TEACHER_POWER_INDEX] = 0.0
        turbine_capacities = turbines["capacity_kwh"].to_numpy(float)
        teacher_base = np.clip(
            np.nan_to_num(teacher_base_kwh) / turbine_capacities[None, :], 0, 1
        ).astype(np.float32)
        group_available = np.zeros((len(times), len(TARGET_COLS)), dtype=bool)
        for group_index, group in enumerate(TARGET_COLS):
            member = turbines["group"].eq(group).to_numpy()
            group_available[:, group_index] = teacher_available[:, member].all(axis=1)
        group_mask = np.isfinite(group_target) & (group_target >= 0.10) & group_available
        turbine_group_target = group_target[:, turbine_group_index]
        turbine_mask = (
            np.isfinite(turbine_target)
            & (turbine_group_target >= 0.10)
            & teacher_available
        )
        clean_group_target = np.nan_to_num(group_target).astype(np.float32)
        clean_turbine_target = np.nan_to_num(turbine_target).astype(np.float32)

        fold_predictions = {}
        for variant in model_variants:
            builder = lambda variant=variant: make_model(
                variant,
                features.shape[-1],
                edge_index,
                edge_attr,
                turbine_indices,
                turbine_group_index,
                aggregation,
                args,
            )
            selected_epoch, inner_curves = inner_epoch_selection(
                variant,
                outer_year,
                times,
                features,
                teacher_base,
                clean_group_target,
                group_mask,
                clean_turbine_target,
                turbine_mask,
                builder,
                turbine_indices,
                args,
                device,
            )
            print(f"{variant} selected_epoch={selected_epoch}", flush=True)
            inner_parts.append(inner_curves)
            prediction, stats = final_fit(
                variant,
                outer_year,
                selected_epoch,
                times,
                features,
                teacher_base,
                clean_group_target,
                group_mask,
                clean_turbine_target,
                turbine_mask,
                builder,
                turbine_indices,
                args,
                device,
            )
            fold_predictions[variant] = prediction
            training_rows.append(stats)

        val_indices = np.flatnonzero(times.year == outer_year)
        teacher_curve = teacher_base[val_indices] @ aggregation
        for group_index, group in enumerate(TARGET_COLS):
            output = {
                "forecast_kst_dtm": times[val_indices],
                "group": group,
                "pred_year": outer_year,
                "actual": actual[val_indices, group_index],
                "teacher_curve": teacher_curve[:, group_index] * capacities[group_index],
            }
            if "direct_teacher_mlp" in fold_predictions:
                output["teacher_mlp"] = (
                    fold_predictions["direct_teacher_mlp"][:, group_index]
                    * capacities[group_index]
                )
            if "direct_spatial_gnn" in fold_predictions:
                output["spatial_gnn"] = (
                    fold_predictions["direct_spatial_gnn"][:, group_index]
                    * capacities[group_index]
                )
            prediction_parts.append(
                pd.DataFrame(output)
            )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    scores, summary = score_predictions(predictions)
    predictions.to_csv(
        args.results_dir / f"{args.stem}_predictions.csv", index=False, encoding="utf-8-sig"
    )
    scores.to_csv(args.results_dir / f"{args.stem}_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(
        args.results_dir / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(inner_parts, ignore_index=True).to_csv(
        args.results_dir / f"{args.stem}_inner_curves.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(training_rows).to_csv(
        args.results_dir / f"{args.stem}_training.csv", index=False, encoding="utf-8-sig"
    )
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print("\n=== training ===")
    print(pd.DataFrame(training_rows).to_string(index=False))


if __name__ == "__main__":
    main()

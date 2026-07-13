from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import _bootstrap  # noqa: F401
from experiments.evaluate_spatial_teacher_gnn_direct_oof import TEACHER_POWER_INDEX, TEACHER_WIND_SLICE
from experiments.evaluate_spatial_teacher_gnn_oof import (
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
    turbine_catalog,
)
from experiments.evaluate_spatiotemporal_hetero_gnn_oof import (
    add_v2_reference,
    build_node_wind,
    make_model,
    score_predictions,
    seed_everything,
)
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Target-centered W3/W6 BiGRU spatial GNN outer-year OOF."
    )
    parser.add_argument("--ldaps", default="data/train/ldaps_train.csv")
    parser.add_argument("--gfs", default="data/train/gfs_train.csv")
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument("--scada-vestas", default="data/train/scada_vestas_train.csv")
    parser.add_argument("--scada-unison", default="data/train/scada_unison_train.csv")
    parser.add_argument(
        "--teacher-cache", type=Path, default=Path("cache/per_turbine_teacher_v1")
    )
    parser.add_argument(
        "--v2-predictions",
        type=Path,
        default=Path("results/spatial_teacher_gnn_direct_v2_predictions.csv"),
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--stem", default="spatiotemporal_gnn_local_w3_w6_v1")
    parser.add_argument("--windows", default="3,6")
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument("--min-selected-epoch", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--gru-hidden-size", type=int, default=48)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--edge-dropout", type=float, default=0.15)
    parser.add_argument("--teacher-dropout", type=float, default=0.20)
    parser.add_argument("--aux-weight", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def window_offsets(window: int) -> tuple[np.ndarray, int]:
    if window == 3:
        return np.asarray([-1, 0, 1]), 1
    if window == 6:
        return np.asarray([-2, -1, 0, 1, 2, 3]), 2
    raise ValueError(f"Unsupported local window: {window}")


def build_local_windows(
    ldaps_path: str,
    times: pd.DatetimeIndex,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    metadata = pd.read_csv(
        ldaps_path,
        encoding="utf-8-sig",
        usecols=["forecast_kst_dtm", "data_available_kst_dtm"],
    ).drop_duplicates()
    metadata["forecast_kst_dtm"] = pd.to_datetime(metadata["forecast_kst_dtm"])
    metadata["issue_time"] = pd.to_datetime(metadata["data_available_kst_dtm"])
    metadata = metadata.set_index("forecast_kst_dtm").reindex(times)
    frame = pd.DataFrame(
        {
            "time_index": np.arange(len(times)),
            "forecast_time": times,
            "issue_time": metadata["issue_time"].to_numpy(),
        }
    )
    offsets, center = window_offsets(window)
    windows = []
    targets = []
    years = []
    for _, table in frame.groupby("issue_time"):
        table = table.sort_values("forecast_time")
        if len(table) != 24:
            continue
        issue_indices = table["time_index"].to_numpy(int)
        issue_times = pd.DatetimeIndex(table["forecast_time"])
        for position in range(24):
            local_positions = np.clip(position + offsets, 0, 23)
            windows.append(issue_indices[local_positions])
            targets.append(issue_indices[position])
            years.append(int(issue_times[position].year))
    if not windows:
        raise ValueError(f"No complete issue windows for W{window}")
    return (
        np.stack(windows),
        np.asarray(targets, dtype=int),
        np.asarray(years, dtype=int),
        center,
    )


class LocalWindowDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        node_wind: np.ndarray,
        group_target: np.ndarray,
        group_mask: np.ndarray,
        turbine_target: np.ndarray,
        turbine_mask: np.ndarray,
        window_index: np.ndarray,
        target_index: np.ndarray,
        samples: np.ndarray,
    ) -> None:
        self.features = features
        self.node_wind = node_wind
        self.group_target = group_target
        self.group_mask = group_mask
        self.turbine_target = turbine_target
        self.turbine_mask = turbine_mask
        self.window_index = window_index
        self.target_index = target_index
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, position: int) -> tuple[torch.Tensor, ...]:
        sample = self.samples[position]
        window = self.window_index[sample]
        target = self.target_index[sample]
        return (
            torch.from_numpy(self.features[window]),
            torch.from_numpy(self.node_wind[window]),
            torch.from_numpy(self.group_target[target].astype(np.float32)),
            torch.from_numpy(self.group_mask[target]),
            torch.from_numpy(self.turbine_target[target].astype(np.float32)),
            torch.from_numpy(self.turbine_mask[target]),
        )


def make_loader(
    features: np.ndarray,
    node_wind: np.ndarray,
    group_target: np.ndarray,
    group_mask: np.ndarray,
    turbine_target: np.ndarray,
    turbine_mask: np.ndarray,
    window_index: np.ndarray,
    target_index: np.ndarray,
    samples: np.ndarray,
    batch_size: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = LocalWindowDataset(
        features,
        node_wind,
        group_target,
        group_mask,
        turbine_target,
        turbine_mask,
        window_index,
        target_index,
        samples,
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
    center: int,
    turbine_indices: np.ndarray,
    teacher_dropout: float,
    aux_weight: float,
    device: torch.device,
) -> float:
    model.train()
    turbine_index = torch.from_numpy(turbine_indices).to(device)
    total_loss = 0.0
    total_rows = 0
    for xb, windb, groupb, group_maskb, turbineb, turbine_maskb in loader:
        xb = xb.to(device, non_blocking=True)
        windb = windb.to(device, non_blocking=True)
        groupb = groupb.to(device, non_blocking=True)
        group_maskb = group_maskb.to(device, non_blocking=True)
        turbineb = turbineb.to(device, non_blocking=True)
        turbine_maskb = turbine_maskb.to(device, non_blocking=True)
        if teacher_dropout > 0:
            xb = xb.clone()
            keep = (
                torch.rand(len(xb), 1, len(turbine_indices), 1, device=device)
                >= teacher_dropout
            ).to(xb.dtype)
            turbine_features = xb[:, :, turbine_index]
            turbine_features[:, :, :, TEACHER_WIND_SLICE] *= keep
            xb[:, :, turbine_index] = turbine_features
        optimizer.zero_grad(set_to_none=True)
        turbine_sequence, group_sequence = model(xb, windb)
        turbine_pred = turbine_sequence[:, center]
        group_pred = group_sequence[:, center]
        group_weights = 0.5 + torch.sqrt(torch.clamp(groupb, 0.0, 1.0))
        group_loss = (
            torch.abs(group_pred - groupb) * group_weights * group_maskb
        ).sum() / group_maskb.sum().clamp_min(1)
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
def predict_samples(
    model: torch.nn.Module,
    features: np.ndarray,
    node_wind: np.ndarray,
    window_index: np.ndarray,
    samples: np.ndarray,
    center: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    parts = []
    for start in range(0, len(samples), batch_size):
        windows = window_index[samples[start : start + batch_size]]
        xb = torch.from_numpy(features[windows]).to(device)
        windb = torch.from_numpy(node_wind[windows]).to(device)
        _, group_sequence = model(xb, windb)
        parts.append(group_sequence[:, center].cpu().numpy())
    return np.concatenate(parts)


def normalized_score(
    actual: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> float:
    scores = []
    for group_index in range(len(TARGET_COLS)):
        valid = mask[:, group_index]
        if not valid.any():
            continue
        nmae, ficr = group_nmae_ficr(
            actual[valid, group_index], prediction[valid, group_index], 1.0
        )
        scores.append(0.5 * (1.0 - nmae) + 0.5 * ficr)
    return float(np.mean(scores)) if scores else float("nan")


def inner_epoch_selection(
    variant: str,
    outer_year: int,
    sample_years: np.ndarray,
    window_index: np.ndarray,
    target_index: np.ndarray,
    center: int,
    features: np.ndarray,
    node_wind: np.ndarray,
    group_target: np.ndarray,
    group_mask: np.ndarray,
    turbine_target: np.ndarray,
    turbine_mask: np.ndarray,
    model_builder,
    turbine_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[int, pd.DataFrame]:
    records = []
    max_epochs = 2 if args.smoke else args.max_epochs
    for inner_index, inner_year in enumerate(year for year in YEARS if year != outer_year):
        fit = np.flatnonzero((sample_years != outer_year) & (sample_years != inner_year))
        evaluate = np.flatnonzero(sample_years == inner_year)
        seed = args.seed + outer_year * 10000 + inner_year * 100 + inner_index
        seed_everything(seed)
        model = model_builder().to(device)
        loader = make_loader(
            features,
            node_wind,
            group_target,
            group_mask,
            turbine_target,
            turbine_mask,
            window_index,
            target_index,
            fit,
            args.batch_size,
            seed,
            device.type == "cuda",
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        for epoch in range(1, max_epochs + 1):
            loss = training_epoch(
                model,
                loader,
                optimizer,
                center,
                turbine_indices,
                args.teacher_dropout,
                args.aux_weight,
                device,
            )
            prediction = predict_samples(
                model,
                features,
                node_wind,
                window_index,
                evaluate,
                center,
                args.eval_batch_size,
                device,
            )
            targets = target_index[evaluate]
            score = normalized_score(
                group_target[targets], prediction, group_mask[targets]
            )
            records.append(
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
    curves = pd.DataFrame(records)
    smoothed = (
        curves.groupby("epoch")["inner_score"].mean().sort_index().rolling(3, min_periods=1).mean()
    )
    selected = int(smoothed.idxmax())
    if not args.smoke:
        selected = max(args.min_selected_epoch, selected)
    return selected, curves


def final_fit(
    variant: str,
    outer_year: int,
    sample_years: np.ndarray,
    window_index: np.ndarray,
    target_index: np.ndarray,
    center: int,
    selected_epoch: int,
    features: np.ndarray,
    node_wind: np.ndarray,
    group_target: np.ndarray,
    group_mask: np.ndarray,
    turbine_target: np.ndarray,
    turbine_mask: np.ndarray,
    model_builder,
    turbine_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    train = np.flatnonzero(sample_years != outer_year)
    validation = np.flatnonzero(sample_years == outer_year)
    seed = args.seed + outer_year * 100000 + int(variant.rsplit("w", 1)[1])
    seed_everything(seed)
    model = model_builder().to(device)
    loader = make_loader(
        features,
        node_wind,
        group_target,
        group_mask,
        turbine_target,
        turbine_mask,
        window_index,
        target_index,
        train,
        args.batch_size,
        seed,
        device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    started = time.time()
    final_loss = np.nan
    for _ in range(selected_epoch):
        final_loss = training_epoch(
            model,
            loader,
            optimizer,
            center,
            turbine_indices,
            args.teacher_dropout,
            args.aux_weight,
            device,
        )
    prediction = predict_samples(
        model,
        features,
        node_wind,
        window_index,
        validation,
        center,
        args.eval_batch_size,
        device,
    )
    stats = {
        "variant": variant,
        "outer_year": outer_year,
        "selected_epoch": selected_epoch,
        "final_train_loss": final_loss,
        "seconds": time.time() - started,
        "n_train_samples": len(train),
        "n_val_samples": len(validation),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction, target_index[validation], stats


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    windows = [int(value.strip()) for value in args.windows.split(",") if value.strip()]
    ldaps_data = load_source(args.ldaps, "ldaps")
    gfs_data = load_source(args.gfs, "gfs")
    times, weather, weather_grids, source_index, lead_hour = align_weather(
        ldaps_data, gfs_data
    )
    year_keep = np.isin(times.year, YEARS)
    times, weather, lead_hour = times[year_keep], weather[year_keep], lead_hour[year_keep]
    window_data = {
        window: build_local_windows(args.ldaps, times, window) for window in windows
    }
    turbines = turbine_catalog()
    static, coordinates = node_static_features(weather_grids, source_index, turbines)
    edge_index, edge_attr = build_graph(coordinates, source_index, turbines)
    labels_frame, actual = load_labels(args.labels, times)
    capacities = np.asarray([GROUP_CAPACITY_KWH[group] for group in TARGET_COLS])
    group_target = actual / capacities[None, :]
    turbine_target = load_turbine_targets(args, labels_frame, times, turbines)
    aggregation, turbine_group_index = group_aggregation(turbines)
    turbine_indices = np.arange(len(weather_grids), len(weather_grids) + len(turbines))
    node_type = np.concatenate([source_index, np.full(len(turbines), 2, dtype=int)])

    prediction_parts = []
    inner_parts = []
    training_rows = []
    outer_years = [2022] if args.smoke else list(YEARS)
    for outer_year in outer_years:
        print(f"\n=== outer_year={outer_year} ===", flush=True)
        teacher, teacher_available = load_teacher_fold(
            args.teacher_cache, outer_year, times, turbines
        )
        train_time_mask = times.year != outer_year
        features, _ = build_fold_features(
            times,
            lead_hour,
            weather,
            static,
            teacher,
            teacher_available,
            train_time_mask,
        )
        features[:, :, TEACHER_POWER_INDEX] = 0.0
        node_wind = build_node_wind(weather, teacher, teacher_available)
        group_available = np.zeros((len(times), len(TARGET_COLS)), dtype=bool)
        for group_index, group in enumerate(TARGET_COLS):
            members = turbines["group"].eq(group).to_numpy()
            group_available[:, group_index] = teacher_available[:, members].all(axis=1)
        group_mask = np.isfinite(group_target) & (group_target >= 0.10) & group_available
        turbine_group_target = group_target[:, turbine_group_index]
        turbine_mask = (
            np.isfinite(turbine_target)
            & (turbine_group_target >= 0.10)
            & teacher_available
        )
        clean_group_target = np.nan_to_num(group_target).astype(np.float32)
        clean_turbine_target = np.nan_to_num(turbine_target).astype(np.float32)

        for window in windows:
            variant = f"static_bigru_w{window}"
            window_index, target_index, sample_years, center = window_data[window]
            builder = lambda: make_model(
                "static_bigru",
                features.shape[-1],
                edge_index,
                edge_attr,
                node_type,
                turbine_indices,
                aggregation,
                args,
            )
            selected_epoch, inner_curves = inner_epoch_selection(
                variant,
                outer_year,
                sample_years,
                window_index,
                target_index,
                center,
                features,
                node_wind,
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
            prediction, validation_targets, stats = final_fit(
                variant,
                outer_year,
                sample_years,
                window_index,
                target_index,
                center,
                selected_epoch,
                features,
                node_wind,
                clean_group_target,
                group_mask,
                clean_turbine_target,
                turbine_mask,
                builder,
                turbine_indices,
                args,
                device,
            )
            training_rows.append(stats)
            for group_index, group in enumerate(TARGET_COLS):
                prediction_parts.append(
                    pd.DataFrame(
                        {
                            "forecast_kst_dtm": times[validation_targets],
                            "group": group,
                            "pred_year": outer_year,
                            "actual": actual[validation_targets, group_index],
                            "prediction": prediction[:, group_index] * capacities[group_index],
                            "variant": variant,
                        }
                    )
                )
    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions = add_v2_reference(predictions, args.v2_predictions)
    scores, summary = score_predictions(predictions)
    predictions.to_csv(
        args.results_dir / f"{args.stem}_predictions.csv", index=False, encoding="utf-8-sig"
    )
    scores.to_csv(args.results_dir / f"{args.stem}_scores.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.results_dir / f"{args.stem}_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(inner_parts, ignore_index=True).to_csv(
        args.results_dir / f"{args.stem}_inner_curves.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(training_rows).to_csv(
        args.results_dir / f"{args.stem}_training.csv", index=False, encoding="utf-8-sig"
    )
    print("\n=== local-window BiGRU GNN summary ===")
    print(summary.to_string(index=False))
    print("\n=== training ===")
    print(pd.DataFrame(training_rows).to_string(index=False))


if __name__ == "__main__":
    main()

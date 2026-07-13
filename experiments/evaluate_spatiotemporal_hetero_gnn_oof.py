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
from experiments.evaluate_spatial_teacher_gnn_direct_oof import (
    TEACHER_POWER_INDEX,
    TEACHER_WIND_SLICE,
)
from experiments.evaluate_spatial_teacher_gnn_oof import (
    TEACHER_FEATURE_COLS,
    WEATHER_FEATURES,
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
from models.spatiotemporal_hetero_gnn import SpatioTemporalHeteroGNN
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr


VARIANTS = ["static_bigru", "dynamic_bigru"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="24h static/dynamic spatio-temporal heterogeneous GNN OOF."
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
    parser.add_argument("--stem", default="spatiotemporal_hetero_gnn_bigru_v1")
    parser.add_argument("--model-variants", default=",".join(VARIANTS))
    parser.add_argument("--max-epochs", type=int, default=35)
    parser.add_argument("--min-selected-epoch", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_issue_index(ldaps_path: str, times: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    metadata = pd.read_csv(
        ldaps_path,
        encoding="utf-8-sig",
        usecols=["forecast_kst_dtm", "data_available_kst_dtm"],
    ).drop_duplicates()
    metadata["forecast_kst_dtm"] = pd.to_datetime(metadata["forecast_kst_dtm"])
    metadata["issue_time"] = pd.to_datetime(metadata["data_available_kst_dtm"])
    metadata = metadata.set_index("forecast_kst_dtm").reindex(times)
    if metadata["issue_time"].isna().any():
        raise ValueError("Missing issue time for aligned weather")
    frame = pd.DataFrame(
        {
            "time_index": np.arange(len(times)),
            "forecast_time": times,
            "issue_time": metadata["issue_time"].to_numpy(),
        }
    )
    sequences = []
    years = []
    for _, table in frame.groupby("issue_time"):
        table = table.sort_values("forecast_time")
        unique_years = table["forecast_time"].dt.year.unique()
        if len(table) != 24 or len(unique_years) != 1:
            continue
        sequences.append(table["time_index"].to_numpy(int))
        years.append(int(unique_years[0]))
    if not sequences:
        raise ValueError("No complete single-year 24h issue batches")
    return np.stack(sequences), np.asarray(years, dtype=int)


def build_node_wind(
    weather: np.ndarray,
    teacher: np.ndarray,
    teacher_available: np.ndarray,
) -> np.ndarray:
    n_times, n_weather = weather.shape[:2]
    n_turbines = teacher.shape[1]
    output = np.zeros((n_times, n_weather + n_turbines, 3), dtype=np.float32)
    u = weather[:, :, WEATHER_FEATURES.index("u_hub")]
    v = weather[:, :, WEATHER_FEATURES.index("v_hub")]
    speed = np.hypot(u, v)
    output[:, :n_weather, 0] = u / np.maximum(speed, 1e-4)
    output[:, :n_weather, 1] = v / np.maximum(speed, 1e-4)
    output[:, :n_weather, 2] = np.clip(speed / 20.0, 0.0, 2.0)

    ws = teacher[:, :, TEACHER_FEATURE_COLS.index("teacher_ws_mean")]
    wd_sin = teacher[:, :, TEACHER_FEATURE_COLS.index("teacher_wd_sin")]
    wd_cos = teacher[:, :, TEACHER_FEATURE_COLS.index("teacher_wd_cos")]
    output[:, n_weather:, 0] = np.where(teacher_available, wd_cos, 0.0)
    output[:, n_weather:, 1] = np.where(teacher_available, wd_sin, 0.0)
    output[:, n_weather:, 2] = np.where(
        teacher_available, np.clip(ws / 20.0, 0.0, 2.0), 0.0
    )
    return np.nan_to_num(output)


def make_model(
    variant: str,
    node_size: int,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    node_type: np.ndarray,
    turbine_indices: np.ndarray,
    aggregation: np.ndarray,
    args: argparse.Namespace,
) -> SpatioTemporalHeteroGNN:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown ST-HGNN variant: {variant}")
    return SpatioTemporalHeteroGNN(
        node_size=node_size,
        edge_size=edge_attr.shape[1],
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_type=torch.from_numpy(node_type.copy()),
        turbine_indices=torch.from_numpy(turbine_indices.copy()),
        group_aggregation=torch.from_numpy(aggregation.copy()),
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        gru_hidden_size=args.gru_hidden_size,
        dropout=args.dropout,
        edge_dropout=args.edge_dropout,
        dynamic_edges=variant == "dynamic_bigru",
    )


def make_loader(
    features: np.ndarray,
    node_wind: np.ndarray,
    group_target: np.ndarray,
    group_mask: np.ndarray,
    turbine_target: np.ndarray,
    turbine_mask: np.ndarray,
    sequence_index: np.ndarray,
    issue_indices: np.ndarray,
    batch_size: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    time_index = sequence_index[issue_indices]
    dataset = TensorDataset(
        torch.from_numpy(features[time_index]),
        torch.from_numpy(node_wind[time_index]),
        torch.from_numpy(group_target[time_index].astype(np.float32)),
        torch.from_numpy(group_mask[time_index]),
        torch.from_numpy(turbine_target[time_index].astype(np.float32)),
        torch.from_numpy(turbine_mask[time_index]),
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
    device: torch.device,
) -> float:
    model.train()
    turbine_index = torch.from_numpy(turbine_indices).to(device)
    total_loss = 0.0
    total_issues = 0
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
        turbine_pred, group_pred = model(xb, windb)
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
        total_issues += len(xb)
    return total_loss / max(total_issues, 1)


@torch.no_grad()
def predict_issues(
    model: torch.nn.Module,
    features: np.ndarray,
    node_wind: np.ndarray,
    sequence_index: np.ndarray,
    issue_indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    parts = []
    for start in range(0, len(issue_indices), batch_size):
        selected = sequence_index[issue_indices[start : start + batch_size]]
        xb = torch.from_numpy(features[selected]).to(device)
        windb = torch.from_numpy(node_wind[selected]).to(device)
        _, group_pred = model(xb, windb)
        parts.append(group_pred.cpu().numpy())
    return np.concatenate(parts)


def score_issue_predictions(
    target: np.ndarray, prediction: np.ndarray, mask: np.ndarray
) -> float:
    scores = []
    for group_index in range(len(TARGET_COLS)):
        valid = mask[:, :, group_index]
        if not valid.any():
            continue
        nmae, ficr = group_nmae_ficr(
            target[:, :, group_index][valid],
            prediction[:, :, group_index][valid],
            1.0,
        )
        scores.append(0.5 * (1.0 - nmae) + 0.5 * ficr)
    return float(np.mean(scores)) if scores else float("nan")


def inner_epoch_selection(
    variant: str,
    outer_year: int,
    issue_years: np.ndarray,
    sequence_index: np.ndarray,
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
    train_years = [year for year in YEARS if year != outer_year]
    max_epochs = 2 if args.smoke else args.max_epochs
    for inner_index, inner_year in enumerate(train_years):
        fit = np.flatnonzero((issue_years != outer_year) & (issue_years != inner_year))
        evaluate = np.flatnonzero(issue_years == inner_year)
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
            sequence_index,
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
                turbine_indices,
                args.teacher_dropout,
                args.aux_weight,
                device,
            )
            prediction = predict_issues(
                model,
                features,
                node_wind,
                sequence_index,
                evaluate,
                args.eval_batch_size,
                device,
            )
            target = group_target[sequence_index[evaluate]]
            mask = group_mask[sequence_index[evaluate]]
            score = score_issue_predictions(target, prediction, mask)
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
    mean_curve = curves.groupby("epoch")["inner_score"].mean().sort_index()
    smoothed = mean_curve.rolling(3, min_periods=1).mean()
    selected = int(smoothed.idxmax())
    if not args.smoke:
        selected = max(args.min_selected_epoch, selected)
    return selected, curves


def final_fit(
    variant: str,
    outer_year: int,
    selected_epoch: int,
    issue_years: np.ndarray,
    sequence_index: np.ndarray,
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
) -> tuple[np.ndarray, dict[str, object]]:
    train = np.flatnonzero(issue_years != outer_year)
    validation = np.flatnonzero(issue_years == outer_year)
    seed = args.seed + outer_year * 100000 + VARIANTS.index(variant)
    seed_everything(seed)
    model = model_builder().to(device)
    loader = make_loader(
        features,
        node_wind,
        group_target,
        group_mask,
        turbine_target,
        turbine_mask,
        sequence_index,
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
            turbine_indices,
            args.teacher_dropout,
            args.aux_weight,
            device,
        )
    prediction = predict_issues(
        model,
        features,
        node_wind,
        sequence_index,
        validation,
        args.eval_batch_size,
        device,
    )
    stats = {
        "variant": variant,
        "outer_year": outer_year,
        "selected_epoch": selected_epoch,
        "final_train_loss": final_loss,
        "seconds": time.time() - started,
        "n_train_issues": len(train),
        "n_val_issues": len(validation),
    }
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return prediction, stats


def score_predictions(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for (variant, group, year), fold in table.groupby(["variant", "group", "pred_year"]):
        capacity = GROUP_CAPACITY_KWH[group]
        valid = fold["actual"].notna() & fold["actual"].ge(0.10 * capacity)
        if not valid.any():
            continue
        nmae, ficr = group_nmae_ficr(
            fold.loc[valid, "actual"], fold.loc[valid, "prediction"], capacity
        )
        records.append(
            {
                "variant": variant,
                "group": group,
                "pred_year": int(year),
                "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
                "nmae": nmae,
                "ficr": ficr,
            }
        )
    scores = pd.DataFrame(records)
    yearly = scores.groupby(["variant", "pred_year"], as_index=False)[
        ["score", "nmae", "ficr"]
    ].mean()
    summary = (
        yearly.groupby("variant", as_index=False)
        .agg(
            mean_score=("score", "mean"),
            mean_nmae=("nmae", "mean"),
            mean_ficr=("ficr", "mean"),
            worst_year=("score", "min"),
            std_year=("score", "std"),
        )
        .sort_values("mean_score", ascending=False)
    )
    return scores, summary


def add_v2_reference(predictions: pd.DataFrame, path: Path) -> pd.DataFrame:
    if not path.exists():
        return predictions
    reference = pd.read_csv(path, encoding="utf-8-sig")
    reference["forecast_kst_dtm"] = pd.to_datetime(reference["forecast_kst_dtm"])
    keys = predictions[["forecast_kst_dtm", "group", "pred_year"]].drop_duplicates()
    reference = reference.merge(keys, on=["forecast_kst_dtm", "group", "pred_year"], how="inner")
    output = reference[
        ["forecast_kst_dtm", "group", "pred_year", "actual", "spatial_gnn"]
    ].rename(columns={"spatial_gnn": "prediction"})
    output["variant"] = "spatial_gnn_v2_reference"
    return pd.concat([predictions, output[predictions.columns]], ignore_index=True)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    ldaps_data = load_source(args.ldaps, "ldaps")
    gfs_data = load_source(args.gfs, "gfs")
    times, weather, weather_grids, source_index, lead_hour = align_weather(
        ldaps_data, gfs_data
    )
    year_keep = np.isin(times.year, YEARS)
    times, weather, lead_hour = times[year_keep], weather[year_keep], lead_hour[year_keep]
    sequence_index, issue_years = build_issue_index(args.ldaps, times)
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
    model_variants = [part.strip() for part in args.model_variants.split(",") if part.strip()]
    invalid = set(model_variants) - set(VARIANTS)
    if invalid or not model_variants:
        raise ValueError(f"Invalid model variants: {sorted(invalid)}")

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
        fold_predictions = {}
        for variant in model_variants:
            builder = lambda variant=variant: make_model(
                variant,
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
                issue_years,
                sequence_index,
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
            prediction, stats = final_fit(
                variant,
                outer_year,
                selected_epoch,
                issue_years,
                sequence_index,
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
            fold_predictions[variant] = prediction
            training_rows.append(stats)

        validation_issues = np.flatnonzero(issue_years == outer_year)
        validation_times = times.to_numpy()[sequence_index[validation_issues]].reshape(-1)
        validation_actual = actual[sequence_index[validation_issues]]
        for group_index, group in enumerate(TARGET_COLS):
            for variant, prediction in fold_predictions.items():
                prediction_parts.append(
                    pd.DataFrame(
                        {
                            "forecast_kst_dtm": validation_times,
                            "group": group,
                            "pred_year": outer_year,
                            "actual": validation_actual[:, :, group_index].reshape(-1),
                            "prediction": (
                                prediction[:, :, group_index].reshape(-1)
                                * capacities[group_index]
                            ),
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
    print("\n=== ST-HGNN BiGRU summary ===")
    print(summary.to_string(index=False))
    print("\n=== training ===")
    print(pd.DataFrame(training_rows).to_string(index=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import copy
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import _bootstrap  # noqa: F401
from models.spatial_teacher_gnn import SpatialTeacherGNN, TeacherTurbineMLP
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr
from utils.per_turbine_scada import (
    build_official_aligned_turbine_targets,
    turbine_capacity_kwh,
)
from utils.per_turbine_teacher import TEACHER_FEATURE_COLS
from utils.power_curve import GROUP_MANUFACTURER, GROUP_TURBINE_PREFIXES
from utils.site_metadata import _latlon_to_xy_km, load_turbine_metadata


YEARS = (2022, 2023, 2024)
WEATHER_FEATURES = [
    "u10",
    "v10",
    "u_hub",
    "v_hub",
    "ws10",
    "ws_hub",
    "temperature",
    "humidity",
    "pressure",
    "gust",
]
SOURCE_COLUMNS = {
    "ldaps": {
        "u10": "heightAboveGround_10_10u",
        "v10": "heightAboveGround_10_10v",
        "u_hub": "heightAboveGround_50_50MUmax",
        "v_hub": "heightAboveGround_50_50MVmax",
        "temperature": "heightAboveGround_2_t",
        "humidity": "heightAboveGround_2_r",
        "pressure": "surface_0_sp",
    },
    "gfs": {
        "u10": "heightAboveGround_10_10u",
        "v10": "heightAboveGround_10_10v",
        "u_hub": "heightAboveGround_100_100u",
        "v_hub": "heightAboveGround_100_100v",
        "temperature": "heightAboveGround_2_2t",
        "humidity": "heightAboveGround_2_2r",
        "pressure": "surface_0_sp",
        "gust": "surface_0_gust",
    },
}
TEACHER_TAG = "optimal_grid_replace_local16_v1"


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
    parser.add_argument("--stem", default="spatial_teacher_gnn_v1")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--residual-amplitude", type=float, default=0.25)
    parser.add_argument("--aux-weight", type=float, default=0.20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260712)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_source(
    path: str, source: str
) -> tuple[pd.DatetimeIndex, np.ndarray, pd.DataFrame, np.ndarray]:
    mapping = SOURCE_COLUMNS[source]
    raw_columns = sorted(set(mapping.values()))
    usecols = [
        "forecast_kst_dtm",
        "data_available_kst_dtm",
        "grid_id",
        "latitude",
        "longitude",
        *raw_columns,
    ]
    table = pd.read_csv(path, encoding="utf-8-sig", usecols=usecols)
    table["forecast_kst_dtm"] = pd.to_datetime(table["forecast_kst_dtm"])
    table["data_available_kst_dtm"] = pd.to_datetime(table["data_available_kst_dtm"])
    if table.duplicated(["forecast_kst_dtm", "grid_id"]).any():
        raise ValueError(f"Duplicate valid/grid rows in {source}")
    grids = (
        table[["grid_id", "latitude", "longitude"]]
        .drop_duplicates("grid_id")
        .sort_values("grid_id")
        .reset_index(drop=True)
    )
    times = pd.DatetimeIndex(sorted(table["forecast_kst_dtm"].unique()))
    availability = table.groupby("forecast_kst_dtm")["data_available_kst_dtm"].agg(
        ["min", "nunique"]
    ).reindex(times)
    if not availability["nunique"].eq(1).all():
        raise ValueError(f"Multiple forecast runs for one valid time in {source}")
    lead_hour = (
        times - pd.DatetimeIndex(availability["min"])
    ).total_seconds().to_numpy(float) / 3600.0
    grid_ids = grids["grid_id"].tolist()
    raw_values = {}
    for name, column in mapping.items():
        pivot = table.pivot(index="forecast_kst_dtm", columns="grid_id", values=column)
        pivot = pivot.reindex(index=times, columns=grid_ids).ffill().bfill()
        if pivot.isna().any().any():
            raise ValueError(f"Unfilled {source} weather values: {column}")
        raw_values[name] = pivot.to_numpy(np.float32)
    raw_values["ws10"] = np.hypot(raw_values["u10"], raw_values["v10"])
    raw_values["ws_hub"] = np.hypot(raw_values["u_hub"], raw_values["v_hub"])
    if "gust" not in raw_values:
        raw_values["gust"] = raw_values["ws_hub"]
    values = np.stack([raw_values[name] for name in WEATHER_FEATURES], axis=-1)
    return times, values, grids, lead_hour


def align_weather(
    ldaps_data: tuple[pd.DatetimeIndex, np.ndarray, pd.DataFrame, np.ndarray],
    gfs_data: tuple[pd.DatetimeIndex, np.ndarray, pd.DataFrame, np.ndarray],
) -> tuple[pd.DatetimeIndex, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray]:
    ldaps_times, ldaps_values, ldaps_grids, ldaps_lead = ldaps_data
    gfs_times, gfs_values, gfs_grids, gfs_lead = gfs_data
    times = ldaps_times.intersection(gfs_times)
    ldaps_index = ldaps_times.get_indexer(times)
    gfs_index = gfs_times.get_indexer(times)
    values = np.concatenate([ldaps_values[ldaps_index], gfs_values[gfs_index]], axis=1)
    grids = pd.concat(
        [ldaps_grids.assign(source="ldaps"), gfs_grids.assign(source="gfs")],
        ignore_index=True,
    )
    source_index = np.concatenate(
        [np.zeros(len(ldaps_grids), dtype=int), np.ones(len(gfs_grids), dtype=int)]
    )
    aligned_ldaps_lead = ldaps_lead[ldaps_index]
    aligned_gfs_lead = gfs_lead[gfs_index]
    if not np.allclose(aligned_ldaps_lead, aligned_gfs_lead):
        raise ValueError("LDAPS and GFS lead hours differ")
    return times, values, grids, source_index, aligned_ldaps_lead


def turbine_catalog() -> pd.DataFrame:
    metadata = load_turbine_metadata().set_index("turbine_id")
    rows = []
    for group_index, group in enumerate(TARGET_COLS):
        for turbine in GROUP_TURBINE_PREFIXES[group]:
            row = metadata.loc[turbine]
            rows.append(
                {
                    "turbine_id": turbine,
                    "group": group,
                    "group_index": group_index,
                    "manufacturer": GROUP_MANUFACTURER[group],
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "capacity_kwh": turbine_capacity_kwh(group),
                    "hub_height_m": float(row["Hub Height(m)"]),
                    "rotor_diameter_m": float(row["Rotor Diameter(m)"]),
                }
            )
    return pd.DataFrame(rows)


def node_static_features(
    weather_grids: pd.DataFrame, source_index: np.ndarray, turbines: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    all_lat = np.concatenate([weather_grids["latitude"], turbines["latitude"]]).astype(float)
    all_lon = np.concatenate([weather_grids["longitude"], turbines["longitude"]]).astype(float)
    ref_lat = float(turbines["latitude"].mean())
    ref_lon = float(turbines["longitude"].mean())
    x_km, y_km = _latlon_to_xy_km(all_lat, all_lon, ref_lat, ref_lon)
    coordinates = np.column_stack([x_km, y_km]).astype(np.float32)
    coordinate_scale = np.maximum(np.std(coordinates, axis=0), 1.0)
    coordinate_features = coordinates / coordinate_scale

    n_weather = len(weather_grids)
    n_turbines = len(turbines)
    static = np.zeros((n_weather + n_turbines, 13), dtype=np.float32)
    static[:, :2] = coordinate_features
    static[np.arange(n_weather), 2 + source_index] = 1.0
    static[n_weather:, 4] = 1.0
    for turbine_index, row in turbines.iterrows():
        node_index = n_weather + turbine_index
        static[node_index, 5 + int(row["group_index"])] = 1.0
        static[node_index, 8 + (0 if row["manufacturer"] == "vestas" else 1)] = 1.0
        static[node_index, 10] = float(row["capacity_kwh"]) / float(
            turbines["capacity_kwh"].max()
        )
        static[node_index, 11] = float(row["hub_height_m"]) / 150.0
        static[node_index, 12] = float(row["rotor_diameter_m"]) / 150.0
    return static, coordinates


def build_graph(
    coordinates: np.ndarray,
    source_index: np.ndarray,
    turbines: pd.DataFrame,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_weather = len(source_index)
    n_nodes = len(coordinates)
    edges: list[tuple[int, int, int]] = []

    def add_knn(indices: np.ndarray, k: int, relation: int) -> None:
        points = coordinates[indices]
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
        for row, node in enumerate(indices):
            neighbors = np.argsort(distances[row])[1 : k + 1]
            for neighbor in neighbors:
                edges.append((int(node), int(indices[neighbor]), relation))

    ldaps_nodes = np.flatnonzero(source_index == 0)
    gfs_nodes = np.flatnonzero(source_index == 1)
    add_knn(ldaps_nodes, 4, 0)
    add_knn(gfs_nodes, 3, 1)

    for turbine_index, _ in turbines.iterrows():
        turbine_node = n_weather + turbine_index
        for source_nodes, k, relation in [(ldaps_nodes, 4, 2), (gfs_nodes, 3, 3)]:
            distance = np.linalg.norm(coordinates[source_nodes] - coordinates[turbine_node], axis=1)
            for weather_node in source_nodes[np.argsort(distance)[:k]]:
                edges.append((int(weather_node), turbine_node, relation))
                edges.append((turbine_node, int(weather_node), relation))

    for _, group_part in turbines.groupby("group"):
        turbine_nodes = (n_weather + group_part.index.to_numpy(int)).tolist()
        for source in turbine_nodes:
            for target in turbine_nodes:
                if source != target:
                    edges.append((source, target, 4))

    edges = list(dict.fromkeys(edges))
    relation_scales = {}
    for relation in range(5):
        distances = [
            np.linalg.norm(coordinates[target] - coordinates[source])
            for source, target, edge_relation in edges
            if edge_relation == relation
        ]
        relation_scales[relation] = max(float(np.median(distances)), 0.1)

    edge_index = np.array([[source, target] for source, target, _ in edges], dtype=np.int64).T
    attributes = []
    for source, target, relation in edges:
        delta = coordinates[target] - coordinates[source]
        distance = float(np.linalg.norm(delta))
        scale = relation_scales[relation]
        angle = math.atan2(float(delta[1]), float(delta[0]))
        relation_one_hot = np.zeros(5, dtype=np.float32)
        relation_one_hot[relation] = 1.0
        attributes.append(
            [
                float(delta[0]) / scale,
                float(delta[1]) / scale,
                distance / scale,
                math.sin(angle),
                math.cos(angle),
                *relation_one_hot,
            ]
        )
    if edge_index.shape[1] == 0 or n_nodes != len(coordinates):
        raise ValueError("Empty or malformed graph")
    return torch.from_numpy(edge_index), torch.tensor(attributes, dtype=torch.float32)


def load_labels(path: str, times: pd.DatetimeIndex) -> tuple[pd.DataFrame, np.ndarray]:
    labels = pd.read_csv(path, encoding="utf-8-sig")
    labels["forecast_kst_dtm"] = pd.to_datetime(labels["kst_dtm"])
    labels = labels.set_index("forecast_kst_dtm").reindex(times)
    labels.index.name = "forecast_kst_dtm"
    actual = labels[TARGET_COLS].to_numpy(float)
    return labels.reset_index()[["forecast_kst_dtm", *TARGET_COLS]], actual


def load_turbine_targets(
    args: argparse.Namespace,
    labels_frame: pd.DataFrame,
    times: pd.DatetimeIndex,
    turbines: pd.DataFrame,
) -> np.ndarray:
    labels_for_builder = labels_frame.rename(columns={"forecast_kst_dtm": "kst_dtm"})
    scada = {
        "vestas": pd.read_csv(args.scada_vestas, encoding="utf-8-sig"),
        "unison": pd.read_csv(args.scada_unison, encoding="utf-8-sig"),
    }
    parts = []
    for group in TARGET_COLS:
        manufacturer = GROUP_MANUFACTURER[group]
        targets = build_official_aligned_turbine_targets(
            scada[manufacturer], labels_for_builder, group
        )
        parts.append(targets[["forecast_kst_dtm", "turbine_id", "turbine_target"]])
    target_table = pd.concat(parts, ignore_index=True).set_index(
        ["forecast_kst_dtm", "turbine_id"]
    )
    index = pd.MultiIndex.from_product(
        [times, turbines["turbine_id"]], names=["forecast_kst_dtm", "turbine_id"]
    )
    values = target_table.reindex(index)["turbine_target"].to_numpy(float)
    values = values.reshape(len(times), len(turbines))
    capacities = turbines["capacity_kwh"].to_numpy(float)
    return values / capacities[None, :]


def load_teacher_fold(
    cache_root: Path,
    outer_year: int,
    times: pd.DatetimeIndex,
    turbines: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((len(times), len(turbines), len(TEACHER_FEATURE_COLS)), np.nan)
    available = np.zeros((len(times), len(turbines)), dtype=bool)
    time_positions = pd.Series(np.arange(len(times)), index=times)
    turbine_positions = pd.Series(np.arange(len(turbines)), index=turbines["turbine_id"])
    for group in TARGET_COLS:
        path = cache_root / f"{group}_pred{outer_year}_{TEACHER_TAG}.pkl"
        if not path.exists():
            if group == "kpx_group_3" and outer_year == 2022:
                continue
            raise FileNotFoundError(f"Required teacher cache is missing: {path}")
        table = pd.read_pickle(path)
        table["forecast_kst_dtm"] = pd.to_datetime(table["forecast_kst_dtm"])
        missing = [column for column in TEACHER_FEATURE_COLS if column not in table.columns]
        if missing:
            raise ValueError(f"Missing teacher columns in {path}: {missing}")
        table = table.loc[table["forecast_kst_dtm"].isin(times)].copy()
        time_index = table["forecast_kst_dtm"].map(time_positions)
        turbine_index = table["turbine_id"].map(turbine_positions)
        valid = time_index.notna() & turbine_index.notna()
        row_index = time_index.loc[valid].astype(int).to_numpy()
        col_index = turbine_index.loc[valid].astype(int).to_numpy()
        values[row_index, col_index] = table.loc[valid, TEACHER_FEATURE_COLS].to_numpy(float)
        available[row_index, col_index] = np.isfinite(values[row_index, col_index]).all(axis=1)
    return values, available


def group_aggregation(turbines: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.zeros((len(turbines), len(TARGET_COLS)), dtype=np.float32)
    group_indices = turbines["group_index"].to_numpy(int).copy()
    for turbine_index, row in turbines.iterrows():
        group = str(row["group"])
        matrix[turbine_index, int(row["group_index"])] = float(row["capacity_kwh"]) / float(
            GROUP_CAPACITY_KWH[group]
        )
    return matrix, group_indices


def build_fold_features(
    times: pd.DatetimeIndex,
    lead_hour: np.ndarray,
    weather: np.ndarray,
    static: np.ndarray,
    teacher: np.ndarray,
    teacher_available: np.ndarray,
    train_time_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weather_mean = np.mean(weather[train_time_mask], axis=(0, 1), keepdims=True)
    weather_std = np.std(weather[train_time_mask], axis=(0, 1), keepdims=True)
    weather_scaled = (weather - weather_mean) / np.maximum(weather_std, 1e-4)

    teacher_scaled = np.zeros_like(teacher, dtype=np.float32)
    for column in range(teacher.shape[-1]):
        fit_mask = train_time_mask[:, None] & teacher_available
        fit_values = teacher[:, :, column][fit_mask]
        mean = float(np.mean(fit_values))
        std = max(float(np.std(fit_values)), 1e-4)
        teacher_scaled[:, :, column] = np.where(
            np.isfinite(teacher[:, :, column]), (teacher[:, :, column] - mean) / std, 0.0
        )

    n_times = len(times)
    n_weather = weather.shape[1]
    n_turbines = teacher.shape[1]
    context = np.column_stack(
        [
            lead_hour / 35.0,
            np.sin(2 * np.pi * (times.month.to_numpy() - 1) / 12.0),
            np.cos(2 * np.pi * (times.month.to_numpy() - 1) / 12.0),
        ]
    ).astype(np.float32)
    node_size = len(WEATHER_FEATURES) + len(TEACHER_FEATURE_COLS) + 1 + static.shape[1] + 3
    features = np.zeros((n_times, n_weather + n_turbines, node_size), dtype=np.float32)
    offset = 0
    features[:, :n_weather, offset : offset + len(WEATHER_FEATURES)] = weather_scaled
    offset += len(WEATHER_FEATURES)
    features[:, n_weather:, offset : offset + len(TEACHER_FEATURE_COLS)] = teacher_scaled
    offset += len(TEACHER_FEATURE_COLS)
    features[:, n_weather:, offset] = teacher_available.astype(np.float32)
    offset += 1
    features[:, :, offset : offset + static.shape[1]] = static[None, :, :]
    offset += static.shape[1]
    features[:, :, offset : offset + 3] = context[:, None, :]

    teacher_base = np.nan_to_num(teacher[:, :, TEACHER_FEATURE_COLS.index("teacher_power_curve_kwh")])
    return features, teacher_base.astype(np.float32)


def train_model(
    model: torch.nn.Module,
    features: np.ndarray,
    teacher_base: np.ndarray,
    group_target: np.ndarray,
    group_mask: np.ndarray,
    turbine_target: np.ndarray,
    turbine_mask: np.ndarray,
    train_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> list[float]:
    dataset = TensorDataset(
        torch.from_numpy(features[train_indices]),
        torch.from_numpy(teacher_base[train_indices]),
        torch.from_numpy(group_target[train_indices].astype(np.float32)),
        torch.from_numpy(group_mask[train_indices]),
        torch.from_numpy(turbine_target[train_indices].astype(np.float32)),
        torch.from_numpy(turbine_mask[train_indices]),
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    history = []
    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        total_rows = 0
        for xb, baseb, groupb, group_maskb, turbineb, turbine_maskb in loader:
            xb = xb.to(device, non_blocking=True)
            baseb = baseb.to(device, non_blocking=True)
            groupb = groupb.to(device, non_blocking=True)
            group_maskb = group_maskb.to(device, non_blocking=True)
            turbineb = turbineb.to(device, non_blocking=True)
            turbine_maskb = turbine_maskb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            turbine_pred, group_pred = model(xb, baseb)
            group_weights = 0.5 + torch.sqrt(torch.clamp(groupb, 0.0, 1.0))
            group_error = torch.abs(group_pred - groupb) * group_weights
            group_loss = (group_error * group_maskb).sum() / group_maskb.sum().clamp_min(1)
            turbine_loss = (
                torch.abs(turbine_pred - turbineb) * turbine_maskb
            ).sum() / turbine_maskb.sum().clamp_min(1)
            residual_penalty = torch.mean(torch.square(turbine_pred - baseb))
            loss = group_loss + args.aux_weight * turbine_loss + 0.01 * residual_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(xb)
            total_rows += len(xb)
        history.append(total_loss / max(total_rows, 1))
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"epoch={epoch:03d} loss={history[-1]:.6f}", flush=True)
    return history


@torch.no_grad()
def predict_model(
    model: torch.nn.Module,
    features: np.ndarray,
    teacher_base: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    turbine_parts = []
    group_parts = []
    for start in range(0, len(indices), batch_size):
        selection = indices[start : start + batch_size]
        xb = torch.from_numpy(features[selection]).to(device)
        baseb = torch.from_numpy(teacher_base[selection]).to(device)
        turbine_pred, group_pred = model(xb, baseb)
        turbine_parts.append(turbine_pred.cpu().numpy())
        group_parts.append(group_pred.cpu().numpy())
    return np.concatenate(turbine_parts), np.concatenate(group_parts)


def score_predictions(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for variant in ["teacher_curve", "teacher_mlp", "spatial_gnn"]:
        if variant not in table.columns:
            continue
        for (group, year), fold in table.groupby(["group", "pred_year"]):
            capacity = GROUP_CAPACITY_KWH[group]
            valid = np.isfinite(fold["actual"]) & fold["actual"].ge(0.10 * capacity)
            if not valid.any():
                continue
            nmae, ficr = group_nmae_ficr(
                fold.loc[valid, "actual"], fold.loc[valid, variant], capacity
            )
            rows.append(
                {
                    "variant": variant,
                    "group": group,
                    "pred_year": int(year),
                    "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
                    "nmae": nmae,
                    "ficr": ficr,
                }
            )
    scores = pd.DataFrame(rows)
    summary_rows = []
    for variant, part in scores.groupby("variant"):
        years = part.groupby("pred_year", as_index=False)[["score", "nmae", "ficr"]].mean()
        summary_rows.append(
            {
                "variant": variant,
                "mean_score": float(years["score"].mean()),
                "mean_nmae": float(years["nmae"].mean()),
                "mean_ficr": float(years["ficr"].mean()),
                "worst_fold": float(years["score"].min()),
                "std_score": float(years["score"].std(ddof=0)),
            }
        )
    return scores, pd.DataFrame(summary_rows).sort_values("mean_score", ascending=False)


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
    year_mask = np.isin(times.year, YEARS)
    times = times[year_mask]
    weather = weather[year_mask]
    lead_hour = lead_hour[year_mask]
    turbines = turbine_catalog()
    static, coordinates = node_static_features(weather_grids, source_index, turbines)
    edge_index, edge_attr = build_graph(coordinates, source_index, turbines)
    labels_frame, actual = load_labels(args.labels, times)
    capacities = np.array([GROUP_CAPACITY_KWH[group] for group in TARGET_COLS], dtype=float)
    group_target = actual / capacities[None, :]
    turbine_target = load_turbine_targets(args, labels_frame, times, turbines)
    aggregation, turbine_group_index = group_aggregation(turbines)
    turbine_indices = np.arange(len(weather_grids), len(weather_grids) + len(turbines))

    prediction_parts = []
    training_rows = []
    for outer_year in YEARS:
        print(f"\n=== outer_year={outer_year} ===", flush=True)
        teacher, teacher_available = load_teacher_fold(
            args.teacher_cache, outer_year, times, turbines
        )
        train_time_mask = times.year != outer_year
        val_time_mask = times.year == outer_year
        features, teacher_base_kwh = build_fold_features(
            times,
            lead_hour,
            weather,
            static,
            teacher,
            teacher_available,
            train_time_mask,
        )
        turbine_capacities = turbines["capacity_kwh"].to_numpy(float)
        teacher_base = teacher_base_kwh / turbine_capacities[None, :]
        teacher_base = np.clip(teacher_base, 0, 1).astype(np.float32)
        group_available = np.zeros((len(times), len(TARGET_COLS)), dtype=bool)
        for group_index, group in enumerate(TARGET_COLS):
            member = turbines["group"].eq(group).to_numpy()
            group_available[:, group_index] = teacher_available[:, member].all(axis=1)
        group_mask = (
            np.isfinite(group_target)
            & (group_target >= 0.10)
            & group_available
        )
        turbine_group_target = group_target[:, turbine_group_index]
        turbine_mask = (
            np.isfinite(turbine_target)
            & (turbine_group_target >= 0.10)
            & teacher_available
        )
        clean_group_target = np.nan_to_num(group_target).astype(np.float32)
        clean_turbine_target = np.nan_to_num(turbine_target).astype(np.float32)
        train_indices = np.flatnonzero(train_time_mask & group_mask.any(axis=1))
        val_indices = np.flatnonzero(val_time_mask)

        model_kwargs = {
            "node_size": features.shape[-1],
            "turbine_indices": torch.from_numpy(turbine_indices),
            "turbine_group_index": torch.from_numpy(turbine_group_index),
            "group_aggregation": torch.from_numpy(aggregation),
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "residual_amplitude": args.residual_amplitude,
        }
        models = {
            "teacher_mlp": TeacherTurbineMLP(**model_kwargs),
            "spatial_gnn": SpatialTeacherGNN(
                **model_kwargs,
                edge_size=edge_attr.shape[1],
                edge_index=edge_index,
                edge_attr=edge_attr,
                num_layers=args.num_layers,
            ),
        }
        fold_predictions = {}
        for model_index, (variant, model) in enumerate(models.items()):
            seed_everything(args.seed + 100 * outer_year + model_index)
            model = model.to(device)
            started = time.time()
            history = train_model(
                model,
                features,
                teacher_base,
                clean_group_target,
                group_mask,
                clean_turbine_target,
                turbine_mask,
                train_indices,
                args,
                device,
            )
            turbine_pred, group_pred = predict_model(
                model,
                features,
                teacher_base,
                val_indices,
                args.eval_batch_size,
                device,
            )
            fold_predictions[variant] = group_pred
            training_rows.append(
                {
                    "outer_year": outer_year,
                    "variant": variant,
                    "epochs": args.epochs,
                    "final_loss": history[-1],
                    "min_train_loss": min(history),
                    "seconds": time.time() - started,
                    "n_train_times": len(train_indices),
                    "n_val_times": len(val_indices),
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        teacher_curve = teacher_base[val_indices] @ aggregation
        for group_index, group in enumerate(TARGET_COLS):
            prediction_parts.append(
                pd.DataFrame(
                    {
                        "forecast_kst_dtm": times[val_indices],
                        "group": group,
                        "pred_year": outer_year,
                        "actual": actual[val_indices, group_index],
                        "teacher_curve": teacher_curve[:, group_index]
                        * GROUP_CAPACITY_KWH[group],
                        "teacher_mlp": fold_predictions["teacher_mlp"][:, group_index]
                        * GROUP_CAPACITY_KWH[group],
                        "spatial_gnn": fold_predictions["spatial_gnn"][:, group_index]
                        * GROUP_CAPACITY_KWH[group],
                    }
                )
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
    pd.DataFrame(training_rows).to_csv(
        args.results_dir / f"{args.stem}_training.csv", index=False, encoding="utf-8-sig"
    )
    print("\n=== summary ===")
    print(summary.to_string(index=False))
    print("\n=== training ===")
    print(pd.DataFrame(training_rows).to_string(index=False))


if __name__ == "__main__":
    main()

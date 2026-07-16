from __future__ import annotations

import argparse
import copy
import gc
import json
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

try:
    import _bootstrap  # noqa: F401
except ModuleNotFoundError:
    from experiments import _bootstrap  # type: ignore[no-redef]  # noqa: F401

from models.source_expert_tcn import SourceExpertTCN
from utils.metrics import (
    GROUP_CAPACITY_KWH,
    TARGET_COLS,
    group_nmae_ficr,
    pooled_oof_summary,
)
from utils.source_expert_dataset import (
    GFS_CORE_SPEC,
    GFS_10M_CORE_SPEC,
    GFS_SURFACE_REGIME_SPEC,
    GFS_SURFACE_PRESSURE_SPEC,
    GFS_THERMO_SYNOPTIC_SPEC,
    GFS_VERTICAL_THERMO_SPEC,
    GFS_VERTICAL_WIND_SPEC,
    GEFS_NEAR_SPREAD_VECTORS,
    GEFS_UPPER_SPREAD_VECTORS,
    LDAPS_CORE_SPEC,
    LDAPS_5M_CORE_SPEC,
    LDAPS_MSLP_SPEC,
    LDAPS_SURFACE_REGIME_SPEC,
    LDAPS_SURFACE_PRESSURE_SPEC,
    LDAPS_THERMO_PBL_SPEC,
    GEFSIssueTensor,
    SourceIssueTensor,
    apply_gefs_publication_fallback,
    build_gefs_mean_core_tensor,
    build_gefs_mean700_core_tensor,
    build_gefs_near_spread_core_tensor,
    build_gefs_spread_core_tensor,
    build_gefs_upper_spread_core_tensor,
    build_grid_source_core_tensor,
    build_ldaps_blh_ratio_tensor,
    build_ldaps_pressure_tendency_tensor,
    fit_source_channel_scaler,
    gefs_publication_audit,
    load_gefs_core_frames,
    ldaps_blh_ratio_required_columns,
    ldaps_pressure_tendency_required_columns,
    select_gefs_issues,
    source_required_columns,
    transform_source_channels,
)
from utils.source_expert_loss import group_balanced_pure_six_loss


CORE_SOURCE_NAMES = ("ldaps_core", "gfs_core", "gefs_mean_core")
SOURCE_NAMES = (
    *CORE_SOURCE_NAMES,
    "gefs_spread_core",
    "gefs_mean700_core",
    "gefs_near_spread_core",
    "gefs_upper_spread_core",
    "gfs_10m_core",
    "gfs_vertical_wind_core",
    "gfs_thermo_synoptic_core",
    "gfs_vertical_thermo_core",
    "gfs_surface_regime_core",
    "ldaps_5m_core",
    "ldaps_blh_ratio_core",
    "ldaps_pressure_tendency_core",
    "ldaps_mslp_core",
    "ldaps_thermo_pbl_core",
    "ldaps_surface_regime_core",
    "ldaps_core_sp",
    "gfs_core_sp",
)
PREDICTION_FILES = {
    "ldaps_core": "ldaps_core_oof_predictions.csv",
    "gfs_core": "gfs_core_oof_predictions.csv",
    "gefs_mean_core": "gefs_mean_core_oof_predictions.csv",
    "gefs_spread_core": "gefs_spread_core_oof_predictions.csv",
    "gefs_mean700_core": "gefs_mean700_core_oof_predictions.csv",
    "gefs_near_spread_core": "gefs_near_spread_core_oof_predictions.csv",
    "gefs_upper_spread_core": "gefs_upper_spread_core_oof_predictions.csv",
    "gfs_10m_core": "gfs_10m_core_oof_predictions.csv",
    "gfs_vertical_wind_core": "gfs_vertical_wind_core_oof_predictions.csv",
    "gfs_thermo_synoptic_core": "gfs_thermo_synoptic_core_oof_predictions.csv",
    "gfs_vertical_thermo_core": "gfs_vertical_thermo_core_oof_predictions.csv",
    "gfs_surface_regime_core": "gfs_surface_regime_core_oof_predictions.csv",
    "ldaps_5m_core": "ldaps_5m_core_oof_predictions.csv",
    "ldaps_blh_ratio_core": "ldaps_blh_ratio_core_oof_predictions.csv",
    "ldaps_pressure_tendency_core": "ldaps_pressure_tendency_core_oof_predictions.csv",
    "ldaps_mslp_core": "ldaps_mslp_core_oof_predictions.csv",
    "ldaps_thermo_pbl_core": "ldaps_thermo_pbl_core_oof_predictions.csv",
    "ldaps_surface_regime_core": "ldaps_surface_regime_core_oof_predictions.csv",
    "ldaps_core_sp": "ldaps_core_sp_oof_predictions.csv",
    "gfs_core_sp": "gfs_core_sp_oof_predictions.csv",
}


@dataclass(frozen=True)
class ExperimentConfig:
    years: tuple[int, ...] = (2022, 2023, 2024)
    seed: int = 42
    epochs: int = 120
    patience: int = 18
    batch_size: int = 32
    eval_batch_size: int = 128
    temporal_hidden_size: int = 64
    num_temporal_blocks: int = 3
    kernel_size: int = 3
    dropout: float = 0.10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    min_delta: float = 1e-5
    min_output_ratio: float = 0.10
    temperature_start: float = 0.10
    temperature_end: float = 0.01


@dataclass(frozen=True)
class SourceBundle:
    name: str
    components: tuple[SourceIssueTensor, ...]

    @property
    def primary(self) -> SourceIssueTensor:
        return self.components[0]

    def validate(self) -> None:
        if not self.components:
            raise ValueError(f"{self.name}: source bundle has no components")
        first = self.primary
        first.validate()
        if first.targets is None:
            raise ValueError(f"{self.name}: training targets are missing")
        for component in self.components[1:]:
            component.validate()
            if not np.array_equal(first.issue_times, component.issue_times):
                raise ValueError(f"{self.name}: component issue times differ")
            if not np.array_equal(first.forecast_times, component.forecast_times):
                raise ValueError(f"{self.name}: component forecast times differ")


@dataclass(frozen=True)
class FoldArrays:
    train_components: tuple[np.ndarray, ...]
    validation_components: tuple[np.ndarray, ...]
    train_time_features: np.ndarray
    validation_time_features: np.ndarray
    train_targets: np.ndarray
    train_valid: np.ndarray
    validation_targets: np.ndarray
    validation_times: np.ndarray
    validation_leads: np.ndarray
    mean_train_targets: np.ndarray
    train_issue_count: int
    validation_issue_count: int
    cross_year_issues_removed: int
    fallback_train_issues_removed: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate approved LDAPS/GFS/GEFS core source experts."
    )
    parser.add_argument("--sources", default=",".join(CORE_SOURCE_NAMES))
    parser.add_argument("--ldaps-train", default="data/train/ldaps_train.csv")
    parser.add_argument("--gfs-train", default="data/train/gfs_train.csv")
    parser.add_argument("--labels", default="data/train/train_labels.csv")
    parser.add_argument("--gefs-root", default="data/external/gefs")
    parser.add_argument(
        "--output-dir", default="windforecast_runs/source_experts_v1"
    )
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_head() -> str | None:
    repository_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        cwd=repository_root,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def read_source_csv(path: str, columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", usecols=list(columns))


def read_labels(path: str) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        usecols=["kst_dtm", *TARGET_COLS],
    )


def load_source_bundle(
    source: str,
    args: argparse.Namespace,
    labels: pd.DataFrame,
) -> SourceBundle:
    if source == "ldaps_core":
        tensor = build_grid_source_core_tensor(
            read_source_csv(
                args.ldaps_train, source_required_columns(LDAPS_CORE_SPEC)
            ),
            LDAPS_CORE_SPEC,
            labels=labels,
        )
        bundle = SourceBundle(source, (tensor,))
    elif source == "ldaps_5m_core":
        tensor = build_grid_source_core_tensor(
            read_source_csv(
                args.ldaps_train,
                source_required_columns(LDAPS_5M_CORE_SPEC),
            ),
            LDAPS_5M_CORE_SPEC,
            labels=labels,
        )
        bundle = SourceBundle(source, (tensor,))
    elif source == "ldaps_blh_ratio_core":
        tensor = build_ldaps_blh_ratio_tensor(
            read_source_csv(
                args.ldaps_train,
                ldaps_blh_ratio_required_columns(),
            ),
            labels=labels,
        )
        bundle = SourceBundle(source, (tensor,))
    elif source == "ldaps_pressure_tendency_core":
        tensor = build_ldaps_pressure_tendency_tensor(
            read_source_csv(
                args.ldaps_train,
                ldaps_pressure_tendency_required_columns(),
            ),
            labels=labels,
        )
        bundle = SourceBundle(source, (tensor,))
    elif source == "ldaps_mslp_core":
        tensor = build_grid_source_core_tensor(
            read_source_csv(
                args.ldaps_train,
                source_required_columns(LDAPS_MSLP_SPEC),
            ),
            LDAPS_MSLP_SPEC,
            labels=labels,
        )
        bundle = SourceBundle(source, (tensor,))
    elif source == "ldaps_thermo_pbl_core":
        tensor = build_grid_source_core_tensor(
            read_source_csv(
                args.ldaps_train,
                source_required_columns(LDAPS_THERMO_PBL_SPEC),
            ),
            LDAPS_THERMO_PBL_SPEC,
            labels=labels,
        )
        bundle = SourceBundle(source, (tensor,))
    elif source == "ldaps_surface_regime_core":
        tensor = build_grid_source_core_tensor(
            read_source_csv(
                args.ldaps_train,
                source_required_columns(LDAPS_SURFACE_REGIME_SPEC),
            ),
            LDAPS_SURFACE_REGIME_SPEC,
            labels=labels,
        )
        bundle = SourceBundle(source, (tensor,))
    elif source == "ldaps_core_sp":
        tensor = build_grid_source_core_tensor(
            read_source_csv(
                args.ldaps_train,
                source_required_columns(LDAPS_SURFACE_PRESSURE_SPEC),
            ),
            LDAPS_SURFACE_PRESSURE_SPEC,
            labels=labels,
        )
        bundle = SourceBundle(source, (tensor,))
    elif source == "gfs_core":
        tensor = build_grid_source_core_tensor(
            read_source_csv(args.gfs_train, source_required_columns(GFS_CORE_SPEC)),
            GFS_CORE_SPEC,
            labels=labels,
        )
        bundle = SourceBundle(source, (tensor,))
    elif source == "gfs_10m_core":
        tensor = build_grid_source_core_tensor(
            read_source_csv(
                args.gfs_train,
                source_required_columns(GFS_10M_CORE_SPEC),
            ),
            GFS_10M_CORE_SPEC,
            labels=labels,
        )
        bundle = SourceBundle(source, (tensor,))
    elif source == "gfs_vertical_wind_core":
        tensor = build_grid_source_core_tensor(
            read_source_csv(
                args.gfs_train,
                source_required_columns(GFS_VERTICAL_WIND_SPEC),
            ),
            GFS_VERTICAL_WIND_SPEC,
            labels=labels,
        )
        bundle = SourceBundle(source, (tensor,))
    elif source == "gfs_thermo_synoptic_core":
        tensor = build_grid_source_core_tensor(
            read_source_csv(
                args.gfs_train,
                source_required_columns(GFS_THERMO_SYNOPTIC_SPEC),
            ),
            GFS_THERMO_SYNOPTIC_SPEC,
            labels=labels,
        )
        bundle = SourceBundle(source, (tensor,))
    elif source == "gfs_vertical_thermo_core":
        tensor = build_grid_source_core_tensor(
            read_source_csv(
                args.gfs_train,
                source_required_columns(GFS_VERTICAL_THERMO_SPEC),
            ),
            GFS_VERTICAL_THERMO_SPEC,
            labels=labels,
        )
        bundle = SourceBundle(source, (tensor,))
    elif source == "gfs_surface_regime_core":
        tensor = build_grid_source_core_tensor(
            read_source_csv(
                args.gfs_train,
                source_required_columns(GFS_SURFACE_REGIME_SPEC),
            ),
            GFS_SURFACE_REGIME_SPEC,
            labels=labels,
        )
        bundle = SourceBundle(source, (tensor,))
    elif source == "gfs_core_sp":
        tensor = build_grid_source_core_tensor(
            read_source_csv(
                args.gfs_train,
                source_required_columns(GFS_SURFACE_PRESSURE_SPEC),
            ),
            GFS_SURFACE_PRESSURE_SPEC,
            labels=labels,
        )
        bundle = SourceBundle(source, (tensor,))
    elif source in (
        "gefs_mean_core",
        "gefs_spread_core",
        "gefs_mean700_core",
        "gefs_near_spread_core",
        "gefs_upper_spread_core",
    ):
        include_spread = source == "gefs_spread_core"
        include_700 = source == "gefs_mean700_core"
        spread_vectors = {
            "gefs_near_spread_core": GEFS_NEAR_SPREAD_VECTORS,
            "gefs_upper_spread_core": GEFS_UPPER_SPREAD_VECTORS,
        }.get(source, ())
        include_gust_spread = source == "gefs_near_spread_core"
        pressure, gust = load_gefs_core_frames(
            args.gefs_root,
            include_spread=include_spread,
            include_700=include_700,
            spread_vectors=spread_vectors,
            include_gust_spread=include_gust_spread,
        )
        builders = {
            "gefs_mean_core": build_gefs_mean_core_tensor,
            "gefs_spread_core": build_gefs_spread_core_tensor,
            "gefs_mean700_core": build_gefs_mean700_core_tensor,
            "gefs_near_spread_core": build_gefs_near_spread_core_tensor,
            "gefs_upper_spread_core": build_gefs_upper_spread_core_tensor,
        }
        builder = builders[source]
        gefs_all = builder(pressure, gust, labels=labels)
        publication = gefs_publication_audit(args.gefs_root, kind="geavg")
        if include_spread or spread_vectors or include_gust_spread:
            spread_publication = gefs_publication_audit(
                args.gefs_root,
                kind="gespr",
            )
            spread_safe = spread_publication.set_index("data_available_kst_dtm")["safe"]
            publication = publication.copy()
            publication["safe"] = publication["safe"] & publication[
                "data_available_kst_dtm"
            ].map(spread_safe).fillna(False)
        gefs_all = apply_gefs_publication_fallback(gefs_all, publication)
        issue_times = (
            pd.read_csv(
                args.ldaps_train,
                encoding="utf-8-sig",
                usecols=["data_available_kst_dtm"],
            )["data_available_kst_dtm"]
            .pipe(pd.to_datetime)
            .drop_duplicates()
            .sort_values()
            .to_numpy(dtype="datetime64[ns]")
        )
        selected: GEFSIssueTensor = select_gefs_issues(gefs_all, issue_times)
        bundle = SourceBundle(source, (selected.pressure, selected.gust))
    else:
        raise ValueError(f"Unknown source: {source}")
    bundle.validate()
    return bundle


def build_fold_masks(
    years: np.ndarray,
    fallback_flags: np.ndarray | None,
    pred_year: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    years = np.asarray(years)
    if years.ndim != 2:
        raise ValueError("years must be [issue,time]")
    cross_year = np.any(years != years[:, :1], axis=1)
    issue_year = years[:, 0]
    train_mask = (issue_year != int(pred_year)) & ~cross_year
    validation_mask = (issue_year == int(pred_year)) & ~cross_year
    if fallback_flags is None:
        fallback_flags = np.zeros(len(years), dtype=bool)
    fallback_flags = np.asarray(fallback_flags, dtype=bool)
    if fallback_flags.shape != (len(years),):
        raise ValueError("fallback flags differ from issue axis")
    fallback_train_removed = train_mask & fallback_flags
    train_mask &= ~fallback_flags
    return train_mask, validation_mask, cross_year, fallback_train_removed


def prepare_fold_arrays(
    bundle: SourceBundle,
    pred_year: int,
    config: ExperimentConfig,
) -> FoldArrays:
    primary = bundle.primary
    train_mask, validation_mask, cross_year, fallback_removed = build_fold_masks(
        primary.years,
        primary.fallback_flags,
        pred_year,
    )
    if not train_mask.any() or not validation_mask.any():
        raise ValueError(f"{bundle.name}: empty train/validation fold for {pred_year}")

    capacities = np.asarray(
        [GROUP_CAPACITY_KWH[group] for group in TARGET_COLS], dtype=np.float32
    )
    targets = primary.targets
    if targets is None:
        raise ValueError(f"{bundle.name}: targets are missing")
    normalized_targets = targets / capacities.reshape(1, 1, -1)
    train_targets = normalized_targets[train_mask]
    train_valid = np.isfinite(train_targets) & (
        train_targets >= float(config.min_output_ratio)
    )
    train_keep = train_valid.any(axis=(1, 2))
    train_targets = train_targets[train_keep]
    train_valid = train_valid[train_keep]
    if not train_keep.any():
        raise ValueError(f"{bundle.name}: fold {pred_year} has no scored train targets")
    mean_train_targets = np.asarray(
        [
            train_targets[..., group_index][train_valid[..., group_index]].mean()
            for group_index in range(len(TARGET_COLS))
        ],
        dtype=np.float32,
    )
    if not np.isfinite(mean_train_targets).all():
        raise ValueError(f"{bundle.name}: fold {pred_year} has an unobserved group")

    train_components = []
    validation_components = []
    for component in bundle.components:
        scaler = fit_source_channel_scaler(component, issue_mask=train_mask)
        transformed = transform_source_channels(component, scaler)
        train_components.append(transformed[train_mask][train_keep])
        validation_components.append(transformed[validation_mask])

    return FoldArrays(
        train_components=tuple(train_components),
        validation_components=tuple(validation_components),
        train_time_features=primary.time_features[train_mask][train_keep].astype(
            np.float32
        ),
        validation_time_features=primary.time_features[validation_mask].astype(
            np.float32
        ),
        train_targets=np.nan_to_num(train_targets, nan=0.0).astype(np.float32),
        train_valid=train_valid.astype(np.float32),
        validation_targets=normalized_targets[validation_mask].astype(np.float32),
        validation_times=primary.forecast_times[validation_mask],
        validation_leads=np.broadcast_to(
            primary.leads.reshape(1, -1),
            (int(validation_mask.sum()), len(primary.leads)),
        ).copy(),
        mean_train_targets=mean_train_targets,
        train_issue_count=int(train_keep.sum()),
        validation_issue_count=int(validation_mask.sum()),
        cross_year_issues_removed=int(cross_year.sum()),
        fallback_train_issues_removed=int(fallback_removed.sum()),
    )


def model_configuration(
    bundle: SourceBundle,
    config: ExperimentConfig,
) -> SourceExpertTCN:
    if bundle.name.startswith("gefs_"):
        hidden_sizes = [64, 16]
        embedding_sizes = [64, 16]
    else:
        hidden_sizes = [64]
        embedding_sizes = [64]
    return SourceExpertTCN(
        component_channels=[
            len(component.channel_names) for component in bundle.components
        ],
        spatial_masks=[
            torch.from_numpy(component.spatial_mask) for component in bundle.components
        ],
        component_hidden_sizes=hidden_sizes,
        component_embedding_sizes=embedding_sizes,
        time_feature_size=bundle.primary.time_features.shape[-1],
        temporal_hidden_size=config.temporal_hidden_size,
        num_temporal_blocks=config.num_temporal_blocks,
        kernel_size=config.kernel_size,
        dropout=config.dropout,
        output_size=len(TARGET_COLS),
    )


def make_loader(
    arrays: FoldArrays,
    batch_size: int,
    seed: int,
    pin_memory: bool,
) -> DataLoader:
    tensors = [torch.from_numpy(component) for component in arrays.train_components]
    tensors.extend(
        [
            torch.from_numpy(arrays.train_time_features),
            torch.from_numpy(arrays.train_targets),
            torch.from_numpy(arrays.train_valid),
        ]
    )
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=pin_memory,
        generator=torch.Generator().manual_seed(seed),
        num_workers=0,
    )


def predict_normalized(
    model: SourceExpertTCN,
    components: tuple[np.ndarray, ...],
    time_features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(time_features), batch_size):
            stop = start + batch_size
            component_batch = [
                torch.from_numpy(component[start:stop]).to(device)
                for component in components
            ]
            time_batch = torch.from_numpy(time_features[start:stop]).to(device)
            parts.append(model(component_batch, time_batch).cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


def score_predictions(
    targets_normalized: np.ndarray,
    predictions_normalized: np.ndarray,
) -> tuple[float, list[dict[str, float | int | str]]]:
    rows = []
    for group_index, group in enumerate(TARGET_COLS):
        actual = targets_normalized[..., group_index].reshape(-1)
        prediction = predictions_normalized[..., group_index].reshape(-1)
        valid = (
            np.isfinite(actual)
            & np.isfinite(prediction)
            & (actual >= 0.10)
        )
        if not valid.any():
            continue
        capacity = GROUP_CAPACITY_KWH[group]
        nmae, ficr = group_nmae_ficr(
            actual[valid] * capacity,
            prediction[valid] * capacity,
            capacity,
        )
        rows.append(
            {
                "group": group,
                "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
                "nmae": nmae,
                "ficr": ficr,
                "n_rows": int(valid.sum()),
            }
        )
    if not rows:
        raise ValueError("Validation fold has no scored targets")
    return float(np.mean([float(row["score"]) for row in rows])), rows


def train_fold(
    bundle: SourceBundle,
    arrays: FoldArrays,
    pred_year: int,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, float | int], list[dict[str, float | int]]]:
    set_seed(config.seed)
    model = model_configuration(bundle, config).to(device)
    loader = make_loader(
        arrays,
        batch_size=config.batch_size,
        seed=config.seed,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    means = torch.from_numpy(arrays.mean_train_targets).to(device)
    best_score = -np.inf
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    history = []
    temperature_ratio = config.temperature_end / config.temperature_start

    for epoch in range(1, config.epochs + 1):
        progress = (epoch - 1) / max(config.epochs - 1, 1)
        temperature = config.temperature_start * temperature_ratio**progress
        model.train()
        losses = []
        for batch in loader:
            component_batch = [
                value.to(device, non_blocking=True)
                for value in batch[: len(bundle.components)]
            ]
            time_batch, target_batch, valid_batch = [
                value.to(device, non_blocking=True)
                for value in batch[len(bundle.components) :]
            ]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(component_batch, time_batch)
            loss = group_balanced_pure_six_loss(
                target_batch,
                prediction,
                valid_batch,
                means,
                temperature,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        validation_prediction = predict_normalized(
            model,
            arrays.validation_components,
            arrays.validation_time_features,
            device,
            config.eval_batch_size,
        )
        validation_score, _ = score_predictions(
            arrays.validation_targets,
            validation_prediction,
        )
        history.append(
            {
                "pred_year": pred_year,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "temperature": temperature,
                "validation_score": validation_score,
            }
        )
        if validation_score > best_score + config.min_delta:
            best_score = validation_score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch == 1 or epoch % 10 == 0 or bad_epochs == 0:
            print(
                f"  {bundle.name} pred={pred_year} epoch={epoch:03d} "
                f"loss={np.mean(losses):.6f} score={validation_score:.6f} "
                f"best={best_score:.6f}@{best_epoch}",
                flush=True,
            )
        if bad_epochs >= config.patience:
            break

    if best_state is None:
        raise RuntimeError(f"{bundle.name}: no checkpoint selected for {pred_year}")
    model.load_state_dict(best_state)
    prediction = predict_normalized(
        model,
        arrays.validation_components,
        arrays.validation_time_features,
        device,
        config.eval_batch_size,
    )
    stats = {
        "best_epoch": best_epoch,
        "epochs_trained": len(history),
        "best_validation_score": best_score,
        "n_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "receptive_field": model.receptive_field,
    }
    return prediction, stats, history


def prediction_frame(
    source: str,
    pred_year: int,
    arrays: FoldArrays,
    predictions: np.ndarray,
) -> pd.DataFrame:
    parts = []
    for group_index, group in enumerate(TARGET_COLS):
        actual = arrays.validation_targets[..., group_index]
        prediction = predictions[..., group_index]
        finite = np.isfinite(actual) & np.isfinite(prediction)
        if not finite.any():
            continue
        capacity = GROUP_CAPACITY_KWH[group]
        parts.append(
            pd.DataFrame(
                {
                    "forecast_kst_dtm": pd.to_datetime(
                        arrays.validation_times[finite]
                    ),
                    "pred_year": pred_year,
                    "lead": arrays.validation_leads[finite].astype(int),
                    "variant": source,
                    "group": group,
                    "official_target": actual[finite] * capacity,
                    "pred": prediction[finite] * capacity,
                }
            )
        )
    if not parts:
        raise ValueError(f"{source}: fold {pred_year} has no OOF predictions")
    return pd.concat(parts, ignore_index=True)


def diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lead_bins = ((12, 17), (18, 23), (24, 29), (30, 35))
    target_bins = ((0.10, 0.25), (0.25, 0.50), (0.50, 0.75), (0.75, 1.01))
    for (source, group), frame in predictions.groupby(["variant", "group"]):
        capacity = GROUP_CAPACITY_KWH[group]
        for lower, upper in lead_bins:
            subset = frame.loc[frame["lead"].between(lower, upper)]
            nmae, ficr = group_nmae_ficr(
                subset["official_target"], subset["pred"], capacity
            )
            rows.append(
                {
                    "diagnostic": "lead_bin",
                    "variant": source,
                    "group": group,
                    "bin": f"{lower}-{upper}h",
                    "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
                    "nmae": nmae,
                    "ficr": ficr,
                    "n_rows": len(subset),
                    "prediction_std_ratio": np.nan,
                }
            )
        ratio = frame["official_target"] / capacity
        for lower, upper in target_bins:
            subset = frame.loc[(ratio >= lower) & (ratio < upper)]
            if subset.empty:
                continue
            nmae, ficr = group_nmae_ficr(
                subset["official_target"], subset["pred"], capacity
            )
            rows.append(
                {
                    "diagnostic": "target_bin",
                    "variant": source,
                    "group": group,
                    "bin": f"{lower:.2f}-{min(upper, 1.0):.2f}",
                    "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
                    "nmae": nmae,
                    "ficr": ficr,
                    "n_rows": len(subset),
                    "prediction_std_ratio": np.nan,
                }
            )
        rows.append(
            {
                "diagnostic": "prediction_variance",
                "variant": source,
                "group": group,
                "bin": "all",
                "score": np.nan,
                "nmae": np.nan,
                "ficr": np.nan,
                "n_rows": len(frame),
                "prediction_std_ratio": float(frame["pred"].std(ddof=0) / capacity),
            }
        )
    return pd.DataFrame(rows)


def residual_correlations(predictions: pd.DataFrame) -> pd.DataFrame:
    work = predictions.copy()
    work["capacity"] = work["group"].map(GROUP_CAPACITY_KWH)
    work["residual_rate"] = (
        work["pred"] - work["official_target"]
    ) / work["capacity"]
    pivot = work.pivot_table(
        index=["forecast_kst_dtm", "group"],
        columns="variant",
        values="residual_rate",
        aggfunc="first",
    )
    correlation = pivot.corr()
    rows = []
    for left in correlation.index:
        for right in correlation.columns:
            if left >= right:
                continue
            rows.append(
                {
                    "left_source": left,
                    "right_source": right,
                    "residual_correlation": correlation.loc[left, right],
                    "common_rows": int(pivot[[left, right]].dropna().shape[0]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    sources = tuple(part.strip() for part in args.sources.split(",") if part.strip())
    unknown = [source for source in sources if source not in SOURCE_NAMES]
    if unknown or not sources:
        raise ValueError(f"Unknown or empty source selection: {unknown}")
    config = ExperimentConfig()
    if args.smoke_test:
        config = ExperimentConfig(years=(2023,), epochs=2, patience=2)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = read_labels(args.labels)
    print(
        f"device={device} sources={sources} years={config.years} seed={config.seed} "
        f"epochs={config.epochs} patience={config.patience}",
        flush=True,
    )

    all_prediction_parts = []
    fold_rows = []
    history_rows = []
    source_manifest = {}
    for source in sources:
        print(f"\n=== Loading {source} ===", flush=True)
        bundle = load_source_bundle(source, args, labels)
        cross_year = np.any(
            bundle.primary.years != bundle.primary.years[:, :1], axis=1
        )
        source_manifest[source] = {
            "components": [component.source for component in bundle.components],
            "component_shapes": [
                list(component.values.shape) for component in bundle.components
            ],
            "channels": [
                list(component.channel_names) for component in bundle.components
            ],
            "cross_year_issue_times": [
                str(pd.Timestamp(value))
                for value in bundle.primary.issue_times[cross_year]
            ],
            "fallback_issue_times": [
                str(pd.Timestamp(value))
                for value in bundle.primary.issue_times[
                    bundle.primary.fallback_flags.astype(bool)
                ]
            ],
        }
        source_parts = []
        for pred_year in config.years:
            print(f"\n--- {source}: outer validation {pred_year} ---", flush=True)
            arrays = prepare_fold_arrays(bundle, pred_year, config)
            print(
                f"train_issues={arrays.train_issue_count} "
                f"validation_issues={arrays.validation_issue_count} "
                f"cross_year_removed={arrays.cross_year_issues_removed} "
                f"fallback_train_removed={arrays.fallback_train_issues_removed}",
                flush=True,
            )
            prediction, stats, history = train_fold(
                bundle,
                arrays,
                pred_year,
                config,
                device,
            )
            _, group_rows = score_predictions(arrays.validation_targets, prediction)
            for row in group_rows:
                fold_rows.append(
                    {
                        "variant": source,
                        "pred_year": pred_year,
                        **row,
                        **stats,
                        "train_issues": arrays.train_issue_count,
                        "validation_issues": arrays.validation_issue_count,
                        "cross_year_issues_removed": arrays.cross_year_issues_removed,
                        "fallback_train_issues_removed": arrays.fallback_train_issues_removed,
                    }
                )
            source_parts.append(
                prediction_frame(source, pred_year, arrays, prediction)
            )
            history_rows.extend(
                {"variant": source, **row} for row in history
            )
            del arrays
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        source_predictions = pd.concat(source_parts, ignore_index=True)
        source_predictions.to_csv(
            output_dir / PREDICTION_FILES[source],
            index=False,
            encoding="utf-8-sig",
        )
        all_prediction_parts.append(source_predictions)
        del bundle
        gc.collect()

    predictions = pd.concat(all_prediction_parts, ignore_index=True)
    summary, group_scores = pooled_oof_summary(predictions)
    summary = summary.sort_values("mean_score", ascending=False).reset_index(drop=True)
    fold_scores = pd.DataFrame(fold_rows)
    diagnostics_frame = diagnostics(predictions)
    correlations = residual_correlations(predictions)

    outputs = {
        "source_expert_core_fold_scores.csv": fold_scores,
        "source_expert_core_summary.csv": summary,
        "source_expert_core_group_scores.csv": group_scores,
        "source_expert_core_diagnostics.csv": diagnostics_frame,
        "source_expert_core_residual_correlations.csv": correlations,
        "source_expert_core_training_history.csv": pd.DataFrame(history_rows),
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, encoding="utf-8-sig")

    manifest = {
        "git_head": git_head(),
        "phase": "source_expert_core_oof",
        "sources": list(sources),
        "config": asdict(config),
        "loss": "generation-weighted pure <=6% smooth reward; no L1/NMAE term",
        "checkpoint_metric": "held-out-year actual hard Score",
        "cross_year_policy": "drop entire issue from outer train and validation",
        "gefs_fallback_policy": "exclude from train loss; retain in validation",
        "test_prediction_created": False,
        "submission_created": False,
        "source_contracts": source_manifest,
        "outputs": [
            *(PREDICTION_FILES[source] for source in sources),
            *outputs.keys(),
        ],
    }
    with (output_dir / "source_expert_core_manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)

    print("\n=== Pooled strict outer-year OOF ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"Outputs: {output_dir}", flush=True)


if __name__ == "__main__":
    main()

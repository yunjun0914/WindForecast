from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import _bootstrap  # noqa: F401
from models.issue_block_tcn import IssueBlockTCN
from utils.issue_block_dataset import make_per_turbine_issue_blocks
from utils.metrics import (
    GROUP_CAPACITY_KWH,
    TARGET_COLS,
    group_nmae_ficr,
    pooled_oof_summary,
)
from utils.per_turbine_features import get_or_build_group_feature_cache
from utils.per_turbine_optimal_grid import (
    OPTIMAL_GRID_FEATURES,
    add_optimal_grid_issue_context,
    optimal_grid_input_columns,
)
from utils.per_turbine_optimal_grid_builder import (
    WindCandidateMatrix,
    build_wind_candidate_matrix,
    select_optimal_grid_features,
)
from utils.per_turbine_scada import build_turbine_scada_hourly, turbine_capacity_kwh
from utils.per_turbine_sequence import SequenceStandardScaler
from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.preprocessing import TIME_KEY_COLS


GRID_CACHE_VERSION = "two_stage_scada_cubic_grid_v1"
STAGE_CACHE_VERSION = "two_stage_scada_cubic_tcn_v1"


@dataclass
class TurbinePanel:
    features: np.ndarray
    wind: np.ndarray
    power: np.ndarray
    official: np.ndarray
    forecast_times: np.ndarray
    issue_times: np.ndarray
    years: np.ndarray
    turbines: tuple[str, ...]
    feature_cols: tuple[str, ...]


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Strict outer-year two-stage TCN OOF: rich weather -> per-turbine "
            "SCADA cubic-equivalent wind -> wind-only TCN + output-side DOY "
            "bias -> measured turbine power -> official group sum."
        )
    )
    parser.add_argument("--groups", default=",".join(TARGET_COLS))
    parser.add_argument("--years", default="2022,2023,2024")
    parser.add_argument("--wind-epochs", type=int, default=100)
    parser.add_argument("--wind-patience", type=int, default=12)
    parser.add_argument("--wind-hidden-size", type=int, default=128)
    parser.add_argument("--wind-num-layers", type=int, default=3)
    parser.add_argument("--wind-kernel-size", type=int, default=3)
    parser.add_argument("--wind-dropout", type=float, default=0.10)
    parser.add_argument("--wind-lr", type=float, default=1e-3)
    parser.add_argument("--wind-weight-decay", type=float, default=1e-4)
    parser.add_argument("--wind-min-delta", type=float, default=1e-5)
    parser.add_argument("--power-epochs", type=int, default=100)
    parser.add_argument("--power-patience", type=int, default=15)
    parser.add_argument("--power-hidden-size", type=int, default=32)
    parser.add_argument("--power-num-layers", type=int, default=2)
    parser.add_argument("--power-kernel-size", type=int, default=3)
    parser.add_argument("--power-dropout", type=float, default=0.05)
    parser.add_argument("--power-lr", type=float, default=1e-3)
    parser.add_argument("--power-weight-decay", type=float, default=1e-4)
    parser.add_argument("--power-min-delta", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--power-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--within-year-folds", type=int, default=3)
    parser.add_argument("--epoch-val-fraction", type=float, default=0.20)
    parser.add_argument("--target-min-output-ratio", type=float, default=0.10)
    parser.add_argument("--turbine-loss-weight", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--cache-root", type=Path, default=Path("cache"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--rebuild-feature-cache", action="store_true")
    parser.add_argument("--rebuild-grid-cache", action="store_true")
    parser.add_argument("--rebuild-stage-cache", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--stem", default="two_stage_scada_cubic_tcn_oof_v1"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def add_turbine_one_hot(
    features: pd.DataFrame, turbines: tuple[str, ...]
) -> tuple[pd.DataFrame, list[str]]:
    out = features.copy()
    columns = []
    for turbine in turbines:
        column = f"turbine_is_{turbine}"
        out[column] = out["turbine_id"].eq(turbine).astype(np.float32)
        columns.append(column)
    return out, columns


def build_panel(
    features: pd.DataFrame,
    scada_hourly: pd.DataFrame,
    labels: pd.DataFrame,
    group: str,
    feature_cols: list[str],
) -> TurbinePanel:
    turbines = tuple(GROUP_TURBINE_PREFIXES[group])
    features, identity_cols = add_turbine_one_hot(features, turbines)
    all_feature_cols = [*feature_cols, *identity_cols]
    target_cols = [
        "forecast_kst_dtm",
        "turbine_id",
        "scada_ws_cubic",
        "scada_power_kwh",
    ]
    target = scada_hourly[target_cols].copy()
    target["forecast_kst_dtm"] = pd.to_datetime(target["forecast_kst_dtm"])
    label = labels[["kst_dtm", group]].rename(
        columns={"kst_dtm": "forecast_kst_dtm", group: "official_target"}
    )
    label["forecast_kst_dtm"] = pd.to_datetime(label["forecast_kst_dtm"])
    table = features.merge(
        target,
        on=["forecast_kst_dtm", "turbine_id"],
        how="left",
        validate="many_to_one",
    ).merge(
        label,
        on="forecast_kst_dtm",
        how="left",
        validate="many_to_one",
    )

    feature_parts = []
    wind_parts = []
    power_parts = []
    reference = None
    for turbine in turbines:
        turbine_table = table.loc[table["turbine_id"].eq(turbine)].copy()
        wind_blocks = make_per_turbine_issue_blocks(
            turbine_table,
            all_feature_cols,
            target_col="scada_ws_cubic",
            official_col="official_target",
        )
        power_blocks = make_per_turbine_issue_blocks(
            turbine_table,
            all_feature_cols,
            target_col="scada_power_kwh",
            official_col="official_target",
        )
        if reference is None:
            reference = wind_blocks
        else:
            if not np.array_equal(reference.issue_times, wind_blocks.issue_times):
                raise ValueError(f"Issue mismatch inside {group}: {turbine}")
            if not np.array_equal(reference.forecast_times, wind_blocks.forecast_times):
                raise ValueError(f"Forecast mismatch inside {group}: {turbine}")
            if not np.array_equal(reference.years, wind_blocks.years):
                raise ValueError(f"Year mismatch inside {group}: {turbine}")
        if not np.array_equal(wind_blocks.issue_times, power_blocks.issue_times):
            raise ValueError(f"Wind/power issue mismatch for {group}/{turbine}")
        feature_parts.append(wind_blocks.features)
        wind_parts.append(wind_blocks.targets)
        power_parts.append(power_blocks.targets)

    if reference is None:
        raise ValueError(f"No issue panel for {group}")
    official = reference.official_targets
    return TurbinePanel(
        features=np.stack(feature_parts, axis=1).astype(np.float32),
        wind=np.stack(wind_parts, axis=1).astype(np.float32),
        power=np.stack(power_parts, axis=1).astype(np.float32),
        official=official.astype(np.float32),
        forecast_times=reference.forecast_times,
        issue_times=reference.issue_times,
        years=reference.years,
        turbines=turbines,
        feature_cols=tuple(all_feature_cols),
    )


def assert_panel_alignment(reference: TurbinePanel, candidate: TurbinePanel) -> None:
    if reference.turbines != candidate.turbines:
        raise ValueError("Turbine order changed between feature panels")
    if not np.array_equal(reference.issue_times, candidate.issue_times):
        raise ValueError("Issue order changed between feature panels")
    if not np.array_equal(reference.forecast_times, candidate.forecast_times):
        raise ValueError("Forecast order changed between feature panels")
    if not np.array_equal(reference.years, candidate.years):
        raise ValueError("Year order changed between feature panels")


def forecast_time_hash(forecast_times: np.ndarray) -> str:
    values = np.sort(np.asarray(forecast_times, dtype="datetime64[ns]").astype(np.int64))
    return hashlib.sha1(values.tobytes()).hexdigest()[:12]


def get_or_build_cubic_grid(
    candidates: WindCandidateMatrix,
    scada_hourly: pd.DataFrame,
    group: str,
    fit_forecast_times: np.ndarray,
    cache_label: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_times = np.unique(
        np.asarray(fit_forecast_times, dtype="datetime64[ns]").reshape(-1)
    )
    digest = forecast_time_hash(fit_times)
    cache_dir = args.cache_root / GRID_CACHE_VERSION
    feature_path = cache_dir / f"{group}_{cache_label}_{digest}_features.pkl"
    selection_path = cache_dir / f"{group}_{cache_label}_{digest}_selection.csv"
    if (
        feature_path.exists()
        and selection_path.exists()
        and not args.rebuild_grid_cache
    ):
        return pd.read_pickle(feature_path), pd.read_csv(selection_path)

    masked_targets = scada_hourly[
        ["forecast_kst_dtm", "turbine_id", "scada_ws_cubic"]
    ].copy()
    masked_targets["forecast_kst_dtm"] = pd.to_datetime(
        masked_targets["forecast_kst_dtm"]
    )
    allowed = masked_targets["forecast_kst_dtm"].isin(pd.DatetimeIndex(fit_times))
    masked_targets.loc[~allowed, "scada_ws_cubic"] = np.nan
    train_years = sorted(pd.DatetimeIndex(fit_times).year.unique().astype(int).tolist())
    optimal, selections = select_optimal_grid_features(
        candidates,
        masked_targets,
        group,
        train_years,
        wind_target="scada_ws_cubic",
    )
    selections["fit_time_hash"] = digest
    selections["fit_hours"] = int(len(fit_times))
    selections["cache_label"] = cache_label
    cache_dir.mkdir(parents=True, exist_ok=True)
    optimal.to_pickle(feature_path)
    selections.to_csv(selection_path, index=False, encoding="utf-8-sig")
    return optimal, selections


def build_cubic_feature_panel(
    base_features: pd.DataFrame,
    candidates: WindCandidateMatrix,
    scada_hourly: pd.DataFrame,
    labels: pd.DataFrame,
    group: str,
    fit_forecast_times: np.ndarray,
    cache_label: str,
    args: argparse.Namespace,
) -> tuple[TurbinePanel, pd.DataFrame]:
    optimal, selections = get_or_build_cubic_grid(
        candidates,
        scada_hourly,
        group,
        fit_forecast_times,
        cache_label,
        args,
    )
    keys = [*TIME_KEY_COLS, "turbine_id"]
    features = base_features.merge(
        optimal[keys + OPTIMAL_GRID_FEATURES],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    coverage = float(features["optgrid_ws_raw"].notna().mean())
    if coverage < 0.95:
        raise ValueError(f"Low cubic-grid coverage for {group}/{cache_label}: {coverage}")
    features = add_optimal_grid_issue_context(features)
    feature_cols = optimal_grid_input_columns(group, include_issue_context=True)
    return build_panel(features, scada_hourly, labels, group, feature_cols), selections


def remove_forecast_overlap(
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    forecast_times: np.ndarray,
) -> np.ndarray:
    train_indices = np.asarray(train_indices, dtype=int)
    val_indices = np.asarray(val_indices, dtype=int)
    if len(train_indices) == 0 or len(val_indices) == 0:
        return train_indices
    heldout_times = np.unique(forecast_times[val_indices].reshape(-1))
    overlap = np.isin(forecast_times[train_indices], heldout_times).any(axis=1)
    return train_indices[~overlap]


def make_crossfit_folds(
    issue_indices: np.ndarray,
    years: np.ndarray,
    issue_times: np.ndarray,
    within_year_folds: int,
) -> list[tuple[str, np.ndarray]]:
    issue_indices = np.asarray(issue_indices, dtype=int)
    present_years = sorted(np.unique(years[issue_indices]).astype(int).tolist())
    if len(present_years) >= 2:
        return [
            (f"year{year}", issue_indices[years[issue_indices] == year])
            for year in present_years
        ]
    ordered = issue_indices[np.argsort(issue_times[issue_indices])]
    chunks = [chunk for chunk in np.array_split(ordered, within_year_folds) if len(chunk)]
    return [(f"chunk{index + 1}", chunk) for index, chunk in enumerate(chunks)]


def make_epoch_split(
    train_indices: np.ndarray,
    years: np.ndarray,
    issue_times: np.ndarray,
    forecast_times: np.ndarray,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_indices = np.asarray(train_indices, dtype=int)
    present_years = sorted(np.unique(years[train_indices]).astype(int).tolist())
    if len(present_years) >= 2:
        val_year = present_years[-1]
        val_indices = train_indices[years[train_indices] == val_year]
        fit_indices = train_indices[years[train_indices] != val_year]
    else:
        ordered = train_indices[np.argsort(issue_times[train_indices])]
        n_val = max(1, int(round(len(ordered) * val_fraction)))
        n_val = min(n_val, max(1, len(ordered) // 2))
        val_indices = ordered[-n_val:]
        fit_indices = ordered[:-n_val]
    fit_indices = remove_forecast_overlap(fit_indices, val_indices, forecast_times)
    if len(fit_indices) < 10 or len(val_indices) < 5:
        raise ValueError(
            f"Insufficient epoch split: train={len(fit_indices)} val={len(val_indices)}"
        )
    return fit_indices, val_indices


def initialize_residual_head(model: IssueBlockTCN) -> None:
    final = model.heads[0][-1]
    if not isinstance(final, nn.Linear):
        raise TypeError("Unexpected IssueBlockTCN head")
    nn.init.zeros_(final.weight)
    nn.init.zeros_(final.bias)


def new_wind_model(
    input_size: int, args: argparse.Namespace, device: torch.device
) -> IssueBlockTCN:
    model = IssueBlockTCN(
        input_size=input_size,
        output_size=1,
        hidden_size=args.wind_hidden_size,
        num_layers=args.wind_num_layers,
        kernel_size=args.wind_kernel_size,
        dropout=args.wind_dropout,
        full_context=True,
    ).to(device)
    initialize_residual_head(model)
    return model


def wind_prediction(
    model: IssueBlockTCN, features: torch.Tensor, base_wind: torch.Tensor
) -> torch.Tensor:
    residual = model(features)[..., 0]
    return torch.clamp(base_wind + residual, min=0.0, max=40.0)


def wind_cubic_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    difference = torch.pow(prediction / scale, 3) - torch.pow(target / scale, 3)
    return (torch.square(difference) * observed).sum() / torch.clamp(
        observed.sum(), min=1.0
    )


def flatten_wind_data(
    panel: TurbinePanel, issue_indices: np.ndarray, base_index: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    features = panel.features[issue_indices]
    target = panel.wind[issue_indices]
    base = features[..., base_index]
    observed = np.isfinite(target)
    n_issue, n_turbine, n_time, n_feature = features.shape
    return (
        features.reshape(n_issue * n_turbine, n_time, n_feature),
        np.nan_to_num(target, nan=0.0).reshape(n_issue * n_turbine, n_time),
        base.reshape(n_issue * n_turbine, n_time),
        observed.reshape(n_issue * n_turbine, n_time).astype(np.float32),
    )


def wind_loader(
    features: np.ndarray,
    target: np.ndarray,
    base: np.ndarray,
    observed: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    keep = observed.any(axis=1)
    dataset = TensorDataset(
        torch.from_numpy(features[keep].astype(np.float32)),
        torch.from_numpy(target[keep].astype(np.float32)),
        torch.from_numpy(base[keep].astype(np.float32)),
        torch.from_numpy(observed[keep].astype(np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )


def train_wind_epochs(
    model: IssueBlockTCN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    scale: float,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    for _ in range(epochs):
        model.train()
        for xb, yb, bb, mb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            bb = bb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction = wind_prediction(model, xb, bb)
            loss = wind_cubic_mse(prediction, yb, mb, scale)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()


def predict_wind_flat(
    model: IssueBlockTCN,
    features: np.ndarray,
    base: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            xb = torch.from_numpy(features[start : start + batch_size]).to(device)
            bb = torch.from_numpy(base[start : start + batch_size]).to(device)
            parts.append(wind_prediction(model, xb, bb).cpu().numpy())
    return np.concatenate(parts, axis=0).astype(np.float32)


def wind_metrics(
    actual: np.ndarray, prediction: np.ndarray, scale: float
) -> dict[str, float]:
    valid = np.isfinite(actual) & np.isfinite(prediction)
    actual = actual[valid].astype(float)
    prediction = prediction[valid].astype(float)
    cubic_difference = (prediction / scale) ** 3 - (actual / scale) ** 3
    return {
        "wind_mae": float(np.mean(np.abs(prediction - actual))),
        "wind_rmse": float(np.sqrt(np.mean(np.square(prediction - actual)))),
        "wind_cubic_mse": float(np.mean(np.square(cubic_difference))),
        "wind_bias": float(np.mean(prediction - actual)),
        "n_wind": int(len(actual)),
    }


def select_and_refit_wind(
    panel: TurbinePanel,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    base_index = panel.feature_cols.index("optgrid_ws_calibrated")
    epoch_train, epoch_val = make_epoch_split(
        train_indices,
        panel.years,
        panel.issue_times,
        panel.forecast_times,
        args.epoch_val_fraction,
    )
    observed_train = panel.wind[train_indices]
    valid_train = observed_train[np.isfinite(observed_train)]
    if len(valid_train) < 500:
        raise ValueError(f"Too few wind targets: {len(valid_train)}")
    scale = float(np.quantile(valid_train, 0.99))
    scale = max(scale, 5.0)

    select_scaler = SequenceStandardScaler()
    select_scaler.fit(panel.features[epoch_train])
    select_panel = copy.copy(panel)
    select_panel.features = select_scaler.transform(panel.features)
    x_train, y_train, b_train, m_train = flatten_wind_data(
        select_panel, epoch_train, base_index
    )
    x_val, _, b_val, _ = flatten_wind_data(select_panel, epoch_val, base_index)
    loader = wind_loader(
        x_train,
        y_train,
        b_train,
        m_train,
        args.batch_size,
        device,
    )
    set_seed(seed)
    model = new_wind_model(panel.features.shape[-1], args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.wind_lr, weight_decay=args.wind_weight_decay
    )
    best_epoch = 0
    best_loss = np.inf
    best_state = None
    bad_epochs = 0
    for epoch in range(1, args.wind_epochs + 1):
        train_wind_epochs(model, loader, optimizer, 1, scale, args, device)
        prediction = predict_wind_flat(
            model, x_val, b_val, args.eval_batch_size, device
        )
        actual = panel.wind[epoch_val].reshape(prediction.shape)
        valid = np.isfinite(actual)
        cubic_difference = (prediction[valid] / scale) ** 3 - (
            actual[valid] / scale
        ) ** 3
        val_loss = float(np.mean(np.square(cubic_difference)))
        if val_loss < best_loss - args.wind_min_delta:
            best_epoch = epoch
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.wind_patience:
            break
    if best_state is None or best_epoch <= 0:
        raise RuntimeError("Wind epoch selection failed")
    epoch_mae = float(np.mean(np.abs(prediction[valid] - actual[valid])))
    del model, optimizer, loader, select_panel
    release_cuda()

    final_scaler = SequenceStandardScaler()
    final_scaler.fit(panel.features[train_indices])
    final_panel = copy.copy(panel)
    final_panel.features = final_scaler.transform(panel.features)
    x_train, y_train, b_train, m_train = flatten_wind_data(
        final_panel, train_indices, base_index
    )
    x_predict, _, b_predict, _ = flatten_wind_data(
        final_panel, predict_indices, base_index
    )
    loader = wind_loader(
        x_train,
        y_train,
        b_train,
        m_train,
        args.batch_size,
        device,
    )
    set_seed(seed + 1)
    final_model = new_wind_model(panel.features.shape[-1], args, device)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=args.wind_lr,
        weight_decay=args.wind_weight_decay,
    )
    train_wind_epochs(
        final_model,
        loader,
        final_optimizer,
        best_epoch,
        scale,
        args,
        device,
    )
    flat_prediction = predict_wind_flat(
        final_model, x_predict, b_predict, args.eval_batch_size, device
    )
    prediction = flat_prediction.reshape(
        len(predict_indices), len(panel.turbines), 24
    )
    stats = {
        "stage": "wind",
        "best_epoch": int(best_epoch),
        "epoch_val_cubic_mse": float(best_loss),
        "epoch_val_wind_mae_last": epoch_mae,
        "wind_scale_p99": scale,
        "n_epoch_train_issues": int(len(epoch_train)),
        "n_epoch_val_issues": int(len(epoch_val)),
        "n_refit_issues": int(len(train_indices)),
        "n_predict_issues": int(len(predict_indices)),
        "n_parameters": int(sum(p.numel() for p in final_model.parameters())),
    }
    del final_model, final_optimizer, loader, final_panel
    release_cuda()
    return prediction.astype(np.float32), stats


class PowerTCNWithDoyBias(nn.Module):
    def __init__(self, n_turbines: int, args: argparse.Namespace) -> None:
        super().__init__()
        self.core = IssueBlockTCN(
            input_size=1,
            output_size=1,
            hidden_size=args.power_hidden_size,
            num_layers=args.power_num_layers,
            kernel_size=args.power_kernel_size,
            dropout=args.power_dropout,
            full_context=True,
        )
        self.doy_coefficients = nn.Parameter(torch.zeros(n_turbines, 3))

    def initialize_base(self, mean_normalized_power: float) -> None:
        final = self.core.heads[0][-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Unexpected IssueBlockTCN head")
        probability = float(np.clip(mean_normalized_power, 0.02, 0.98))
        logit = math.log(probability / (1.0 - probability))
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, logit)

    def forward(
        self, normalized_wind: torch.Tensor, doy_design: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if normalized_wind.ndim != 3:
            raise ValueError(
                f"Expected wind [batch,turbine,time], got {normalized_wind.shape}"
            )
        batch, n_turbines, n_time = normalized_wind.shape
        core_input = normalized_wind.reshape(batch * n_turbines, n_time, 1)
        raw = self.core(core_input)[..., 0].reshape(batch, n_turbines, n_time)
        base = torch.sigmoid(raw)
        bias = torch.einsum("btc,nc->bnt", doy_design, self.doy_coefficients)
        prediction = torch.clamp(base + bias, min=0.0, max=1.0)
        return prediction, base, bias


def make_doy_design(forecast_times: np.ndarray) -> np.ndarray:
    flat = pd.DatetimeIndex(forecast_times.reshape(-1))
    angle = 2.0 * math.pi * (flat.dayofyear.to_numpy(float) - 1.0) / 365.25
    design = np.stack(
        [np.ones(len(flat)), np.sin(angle), np.cos(angle)], axis=-1
    ).astype(np.float32)
    return design.reshape(*forecast_times.shape, 3)


def normalize_wind(
    wind: np.ndarray, train_indices: np.ndarray
) -> tuple[np.ndarray, float, float]:
    train_values = wind[train_indices]
    mean = float(np.mean(train_values))
    std = float(np.std(train_values))
    std = max(std, 1e-3)
    return ((wind - mean) / std).astype(np.float32), mean, std


def power_loss(
    prediction_norm: torch.Tensor,
    target_norm: torch.Tensor,
    turbine_observed: torch.Tensor,
    official_norm: torch.Tensor,
    scored: torch.Tensor,
    turbine_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n_turbines = prediction_norm.shape[1]
    turbine_mask = turbine_observed * scored[:, None, :]
    turbine_error = torch.abs(prediction_norm - target_norm) * turbine_mask
    turbine_denominator = torch.clamp(turbine_mask.sum(), min=1.0)
    turbine_mae = turbine_error.sum() / turbine_denominator

    group_prediction = prediction_norm.mean(dim=1)
    group_error = torch.abs(group_prediction - official_norm) * scored
    group_mae = group_error.sum() / torch.clamp(scored.sum(), min=1.0)
    total = turbine_weight * turbine_mae + (1.0 - turbine_weight) * group_mae
    if n_turbines <= 0:
        raise ValueError("No turbines in power loss")
    return total, turbine_mae, group_mae


def power_loader(
    wind: np.ndarray,
    doy: np.ndarray,
    power_norm: np.ndarray,
    observed: np.ndarray,
    official_norm: np.ndarray,
    scored: np.ndarray,
    issue_indices: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(wind[issue_indices].astype(np.float32)),
        torch.from_numpy(doy[issue_indices].astype(np.float32)),
        torch.from_numpy(power_norm[issue_indices].astype(np.float32)),
        torch.from_numpy(observed[issue_indices].astype(np.float32)),
        torch.from_numpy(official_norm[issue_indices].astype(np.float32)),
        torch.from_numpy(scored[issue_indices].astype(np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )


def train_power_epochs(
    model: PowerTCNWithDoyBias,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    for _ in range(epochs):
        model.train()
        for wb, db, yb, mb, ob, sb in loader:
            wb = wb.to(device, non_blocking=True)
            db = db.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            ob = ob.to(device, non_blocking=True)
            sb = sb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction, _, _ = model(wb, db)
            loss, _, _ = power_loss(
                prediction,
                yb,
                mb,
                ob,
                sb,
                args.turbine_loss_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()


def predict_power(
    model: PowerTCNWithDoyBias,
    wind: np.ndarray,
    doy: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    prediction_parts = []
    base_parts = []
    bias_parts = []
    with torch.no_grad():
        for start in range(0, len(wind), batch_size):
            wb = torch.from_numpy(wind[start : start + batch_size]).to(device)
            db = torch.from_numpy(doy[start : start + batch_size]).to(device)
            prediction, base, bias = model(wb, db)
            prediction_parts.append(prediction.cpu().numpy())
            base_parts.append(base.cpu().numpy())
            bias_parts.append(bias.cpu().numpy())
    return (
        np.concatenate(prediction_parts).astype(np.float32),
        np.concatenate(base_parts).astype(np.float32),
        np.concatenate(bias_parts).astype(np.float32),
    )


def evaluate_power_loss_numpy(
    prediction: np.ndarray,
    power_norm: np.ndarray,
    observed: np.ndarray,
    official_norm: np.ndarray,
    scored: np.ndarray,
    turbine_weight: float,
) -> tuple[float, float, float]:
    turbine_mask = observed * scored[:, None, :]
    turbine_mae = float(
        (np.abs(prediction - power_norm) * turbine_mask).sum()
        / max(float(turbine_mask.sum()), 1.0)
    )
    group_mae = float(
        (np.abs(prediction.mean(axis=1) - official_norm) * scored).sum()
        / max(float(scored.sum()), 1.0)
    )
    total = turbine_weight * turbine_mae + (1.0 - turbine_weight) * group_mae
    return total, turbine_mae, group_mae


def select_and_refit_power(
    panel: TurbinePanel,
    predicted_wind: np.ndarray,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    group: str,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    turbine_capacity = float(turbine_capacity_kwh(group))
    group_capacity = float(GROUP_CAPACITY_KWH[group])
    normalized_wind, wind_mean, wind_std = normalize_wind(
        predicted_wind, train_indices
    )
    doy = make_doy_design(panel.forecast_times)
    power_norm = np.clip(panel.power / turbine_capacity, 0.0, 1.0)
    observed = np.isfinite(panel.power).astype(np.float32)
    power_norm = np.nan_to_num(power_norm, nan=0.0).astype(np.float32)
    official_norm = np.nan_to_num(panel.official / group_capacity, nan=0.0).astype(
        np.float32
    )
    scored = (
        np.isfinite(panel.official)
        & (panel.official >= group_capacity * args.target_min_output_ratio)
    ).astype(np.float32)
    epoch_train, epoch_val = make_epoch_split(
        train_indices,
        panel.years,
        panel.issue_times,
        panel.forecast_times,
        args.epoch_val_fraction,
    )
    mean_power = float(
        power_norm[epoch_train][observed[epoch_train].astype(bool)].mean()
    )
    loader = power_loader(
        normalized_wind,
        doy,
        power_norm,
        observed,
        official_norm,
        scored,
        epoch_train,
        args.power_batch_size,
        device,
    )
    set_seed(seed)
    model = PowerTCNWithDoyBias(len(panel.turbines), args).to(device)
    model.initialize_base(mean_power)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.power_lr, weight_decay=args.power_weight_decay
    )
    best_epoch = 0
    best_loss = np.inf
    best_turbine_mae = np.nan
    best_group_mae = np.nan
    best_state = None
    bad_epochs = 0
    for epoch in range(1, args.power_epochs + 1):
        train_power_epochs(model, loader, optimizer, 1, args, device)
        val_prediction, _, _ = predict_power(
            model,
            normalized_wind[epoch_val],
            doy[epoch_val],
            args.eval_batch_size,
            device,
        )
        val_loss, val_turbine, val_group = evaluate_power_loss_numpy(
            val_prediction,
            power_norm[epoch_val],
            observed[epoch_val],
            official_norm[epoch_val],
            scored[epoch_val],
            args.turbine_loss_weight,
        )
        if val_loss < best_loss - args.power_min_delta:
            best_epoch = epoch
            best_loss = val_loss
            best_turbine_mae = val_turbine
            best_group_mae = val_group
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.power_patience:
            break
    if best_state is None or best_epoch <= 0:
        raise RuntimeError("Power epoch selection failed")
    del model, optimizer, loader
    release_cuda()

    mean_power = float(power_norm[train_indices][observed[train_indices].astype(bool)].mean())
    final_loader = power_loader(
        normalized_wind,
        doy,
        power_norm,
        observed,
        official_norm,
        scored,
        train_indices,
        args.power_batch_size,
        device,
    )
    set_seed(seed + 1)
    final_model = PowerTCNWithDoyBias(len(panel.turbines), args).to(device)
    final_model.initialize_base(mean_power)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=args.power_lr,
        weight_decay=args.power_weight_decay,
    )
    train_power_epochs(
        final_model,
        final_loader,
        final_optimizer,
        best_epoch,
        args,
        device,
    )
    prediction, base, bias = predict_power(
        final_model,
        normalized_wind[predict_indices],
        doy[predict_indices],
        args.eval_batch_size,
        device,
    )
    coefficients = final_model.doy_coefficients.detach().cpu().numpy()
    stats: dict[str, object] = {
        "stage": "power",
        "best_epoch": int(best_epoch),
        "epoch_val_loss": float(best_loss),
        "epoch_val_turbine_nmae": float(best_turbine_mae),
        "epoch_val_group_nmae": float(best_group_mae),
        "predicted_wind_mean": wind_mean,
        "predicted_wind_std": wind_std,
        "n_epoch_train_issues": int(len(epoch_train)),
        "n_epoch_val_issues": int(len(epoch_val)),
        "n_refit_issues": int(len(train_indices)),
        "n_predict_issues": int(len(predict_indices)),
        "n_parameters": int(sum(p.numel() for p in final_model.parameters())),
    }
    for turbine_index, turbine in enumerate(panel.turbines):
        stats[f"{turbine}_bias_intercept"] = float(coefficients[turbine_index, 0])
        stats[f"{turbine}_bias_sin_doy"] = float(coefficients[turbine_index, 1])
        stats[f"{turbine}_bias_cos_doy"] = float(coefficients[turbine_index, 2])
    del final_model, final_optimizer, final_loader
    release_cuda()
    return prediction, base, bias, stats


def available_target_years(
    labels: pd.DataFrame, group: str, requested_years: list[int]
) -> list[int]:
    years = pd.to_datetime(labels["kst_dtm"]).dt.year
    return [
        year
        for year in requested_years
        if int(labels.loc[years.eq(year), group].notna().sum()) >= 200
    ]


def fold_score_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, group, pred_year), part in predictions.groupby(
        ["variant", "group", "pred_year"], sort=False
    ):
        nmae, ficr = group_nmae_ficr(
            part["official_target"], part["pred"], GROUP_CAPACITY_KWH[group]
        )
        rows.append(
            {
                "variant": variant,
                "group": group,
                "pred_year": int(pred_year),
                "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
                "nmae": nmae,
                "ficr": ficr,
                "bias_kwh": float(np.mean(part["pred"] - part["official_target"])),
                "n_rows": int(len(part)),
            }
        )
    return pd.DataFrame(rows)


def wind_score_rows(turbine_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in ["optgrid_ws_calibrated", "predicted_scada_ws"]:
        for scope_keys, part in [(("pooled",), turbine_predictions)]:
            del scope_keys
            metrics = wind_metrics(
                part["actual_scada_ws_cubic"].to_numpy(float),
                part[model].to_numpy(float),
                float(part["wind_scale_p99"].median()),
            )
            rows.append({"scope": "pooled", "model": model, **metrics})
        for (group, pred_year), part in turbine_predictions.groupby(
            ["group", "pred_year"], sort=True
        ):
            metrics = wind_metrics(
                part["actual_scada_ws_cubic"].to_numpy(float),
                part[model].to_numpy(float),
                float(part["wind_scale_p99"].median()),
            )
            rows.append(
                {
                    "scope": f"{group}_year{pred_year}",
                    "model": model,
                    **metrics,
                }
            )
        for (group, turbine), part in turbine_predictions.groupby(
            ["group", "turbine_id"], sort=True
        ):
            metrics = wind_metrics(
                part["actual_scada_ws_cubic"].to_numpy(float),
                part[model].to_numpy(float),
                float(part["wind_scale_p99"].median()),
            )
            rows.append(
                {
                    "scope": f"{group}_{turbine}",
                    "model": model,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.within_year_folds < 2:
        raise ValueError("--within-year-folds must be at least 2")
    if not 0.05 <= args.epoch_val_fraction <= 0.50:
        raise ValueError("--epoch-val-fraction must be in [0.05, 0.50]")
    if not 0.0 <= args.turbine_loss_weight <= 1.0:
        raise ValueError("--turbine-loss-weight must be in [0, 1]")
    groups = parse_csv(args.groups)
    requested_years = [int(value) for value in parse_csv(args.years)]
    args.results_dir.mkdir(parents=True, exist_ok=True)
    stage_cache_dir = args.cache_root / STAGE_CACHE_VERSION
    stage_cache_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} groups={groups} years={requested_years} "
        f"wind=h{args.wind_hidden_size}/L{args.wind_num_layers} cubic-MSE "
        f"power=h{args.power_hidden_size}/L{args.power_num_layers} normalized-MAE",
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
    candidates = build_wind_candidate_matrix(ldaps, gfs)
    print(
        f"candidate_rows={len(candidates.keys)} candidates={len(candidates.names)}",
        flush=True,
    )

    group_prediction_parts = []
    turbine_prediction_parts = []
    training_rows = []
    selection_parts = []
    variant = "two_stage_scada_cubic_tcn"
    completed_outer_folds = 0

    for group_index, group in enumerate(groups):
        target_years = available_target_years(labels, group, requested_years)
        print(f"\n{group}: official target years={target_years}", flush=True)
        base_features = get_or_build_group_feature_cache(
            ldaps,
            gfs,
            group,
            cache_root=args.cache_root,
            rebuild=args.rebuild_feature_cache,
        )
        scada_hourly = build_turbine_scada_hourly(scada_by_group[group], group)
        reference_feature = optimal_grid_input_columns(group)[0]
        reference = build_panel(
            base_features,
            scada_hourly,
            labels,
            group,
            [reference_feature],
        )

        for pred_year in target_years:
            train_years = [year for year in target_years if year != pred_year]
            outer_train_indices = np.flatnonzero(np.isin(reference.years, train_years))
            outer_val_indices = np.flatnonzero(reference.years == pred_year)
            if len(outer_train_indices) < 20 or len(outer_val_indices) < 20:
                raise ValueError(
                    f"Insufficient outer issues {group} pred{pred_year}: "
                    f"train={len(outer_train_indices)} val={len(outer_val_indices)}"
                )
            print(
                f"\n{group} pred_year={pred_year} train_years={train_years} "
                f"train_issues={len(outer_train_indices)} val_issues={len(outer_val_indices)}",
                flush=True,
            )

            crossfit_wind = np.full(reference.wind.shape, np.nan, dtype=np.float32)
            crossfit_folds = make_crossfit_folds(
                outer_train_indices,
                reference.years,
                reference.issue_times,
                args.within_year_folds,
            )
            for fold_index, (fold_name, heldout_indices) in enumerate(crossfit_folds):
                wind_train_indices = np.setdiff1d(
                    outer_train_indices, heldout_indices, assume_unique=False
                )
                wind_train_indices = remove_forecast_overlap(
                    wind_train_indices, heldout_indices, reference.forecast_times
                )
                fit_times = reference.forecast_times[wind_train_indices]
                cache_label = f"pred{pred_year}_oof_{fold_name}"
                panel, selections = build_cubic_feature_panel(
                    base_features,
                    candidates,
                    scada_hourly,
                    labels,
                    group,
                    fit_times,
                    cache_label,
                    args,
                )
                assert_panel_alignment(reference, panel)
                prediction, stats = select_and_refit_wind(
                    panel,
                    wind_train_indices,
                    heldout_indices,
                    args,
                    device,
                    args.seed
                    + group_index * 100000
                    + pred_year * 100
                    + fold_index * 10,
                )
                crossfit_wind[heldout_indices] = prediction
                stats.update(
                    {
                        "group": group,
                        "pred_year": pred_year,
                        "role": f"stage1_oof_{fold_name}",
                        "train_years": ",".join(map(str, train_years)),
                    }
                )
                training_rows.append(stats)
                selections["pred_year"] = pred_year
                selections["role"] = f"stage1_oof_{fold_name}"
                selection_parts.append(selections)
                actual = reference.wind[heldout_indices]
                metrics = wind_metrics(
                    actual,
                    prediction,
                    float(stats["wind_scale_p99"]),
                )
                print(
                    f"  stage1 OOF {fold_name}: train={len(wind_train_indices)} "
                    f"heldout={len(heldout_indices)} epoch={stats['best_epoch']} "
                    f"wind_MAE={metrics['wind_mae']:.4f} "
                    f"cubic_MSE={metrics['wind_cubic_mse']:.6f}",
                    flush=True,
                )

            if not np.isfinite(crossfit_wind[outer_train_indices]).all():
                missing = int(
                    np.count_nonzero(~np.isfinite(crossfit_wind[outer_train_indices]))
                )
                raise ValueError(f"Incomplete stage1 cross-fit wind: missing={missing}")

            final_fit_times = reference.forecast_times[outer_train_indices]
            final_panel, selections = build_cubic_feature_panel(
                base_features,
                candidates,
                scada_hourly,
                labels,
                group,
                final_fit_times,
                f"pred{pred_year}_outer",
                args,
            )
            assert_panel_alignment(reference, final_panel)
            outer_wind, wind_stats = select_and_refit_wind(
                final_panel,
                outer_train_indices,
                outer_val_indices,
                args,
                device,
                args.seed + group_index * 100000 + pred_year * 100 + 90,
            )
            wind_stats.update(
                {
                    "group": group,
                    "pred_year": pred_year,
                    "role": "stage1_outer",
                    "train_years": ",".join(map(str, train_years)),
                }
            )
            training_rows.append(wind_stats)
            selections["pred_year"] = pred_year
            selections["role"] = "stage1_outer"
            selection_parts.append(selections)
            predicted_wind = np.full(reference.wind.shape, np.nan, dtype=np.float32)
            predicted_wind[outer_train_indices] = crossfit_wind[outer_train_indices]
            predicted_wind[outer_val_indices] = outer_wind

            power_prediction, power_base, power_bias, power_stats = select_and_refit_power(
                reference,
                predicted_wind,
                outer_train_indices,
                outer_val_indices,
                group,
                args,
                device,
                args.seed + group_index * 100000 + pred_year * 100 + 95,
            )
            power_stats.update(
                {
                    "group": group,
                    "pred_year": pred_year,
                    "role": "stage2_outer",
                    "train_years": ",".join(map(str, train_years)),
                }
            )
            training_rows.append(power_stats)

            turbine_capacity = float(turbine_capacity_kwh(group))
            predicted_power_kwh = power_prediction * turbine_capacity
            base_power_kwh = power_base * turbine_capacity
            bias_power_kwh = power_bias * turbine_capacity
            group_prediction = predicted_power_kwh.sum(axis=1)
            group_prediction = np.clip(
                group_prediction, 0.0, GROUP_CAPACITY_KWH[group]
            )
            official = reference.official[outer_val_indices]
            group_part = pd.DataFrame(
                {
                    "forecast_kst_dtm": reference.forecast_times[
                        outer_val_indices
                    ].reshape(-1),
                    "issue_kst_dtm": np.repeat(
                        reference.issue_times[outer_val_indices], 24
                    ),
                    "group": group,
                    "pred_year": pred_year,
                    "official_target": official.reshape(-1),
                    "pred": group_prediction.reshape(-1),
                    "variant": variant,
                }
            ).dropna(subset=["official_target"])
            if group_part.duplicated(["forecast_kst_dtm", "group"]).any():
                raise ValueError(f"Duplicate outer predictions for {group} pred{pred_year}")
            group_prediction_parts.append(group_part)

            base_index = final_panel.feature_cols.index("optgrid_ws_calibrated")
            optgrid_outer = final_panel.features[outer_val_indices, :, :, base_index]
            for turbine_index, turbine in enumerate(reference.turbines):
                turbine_part = pd.DataFrame(
                    {
                        "forecast_kst_dtm": reference.forecast_times[
                            outer_val_indices
                        ].reshape(-1),
                        "issue_kst_dtm": np.repeat(
                            reference.issue_times[outer_val_indices], 24
                        ),
                        "group": group,
                        "turbine_id": turbine,
                        "pred_year": pred_year,
                        "actual_scada_ws_cubic": reference.wind[
                            outer_val_indices, turbine_index
                        ].reshape(-1),
                        "optgrid_ws_calibrated": optgrid_outer[
                            :, turbine_index
                        ].reshape(-1),
                        "predicted_scada_ws": outer_wind[:, turbine_index].reshape(-1),
                        "actual_scada_power_kwh": reference.power[
                            outer_val_indices, turbine_index
                        ].reshape(-1),
                        "predicted_power_kwh": predicted_power_kwh[
                            :, turbine_index
                        ].reshape(-1),
                        "base_power_kwh": base_power_kwh[:, turbine_index].reshape(-1),
                        "doy_bias_kwh": bias_power_kwh[:, turbine_index].reshape(-1),
                        "wind_scale_p99": float(wind_stats["wind_scale_p99"]),
                    }
                )
                turbine_prediction_parts.append(turbine_part)

            nmae, ficr = group_nmae_ficr(
                group_part["official_target"],
                group_part["pred"],
                GROUP_CAPACITY_KWH[group],
            )
            outer_wind_metrics = wind_metrics(
                reference.wind[outer_val_indices],
                outer_wind,
                float(wind_stats["wind_scale_p99"]),
            )
            print(
                f"  stage1 OUTER: epoch={wind_stats['best_epoch']} "
                f"wind_MAE={outer_wind_metrics['wind_mae']:.4f}; "
                f"stage2 epoch={power_stats['best_epoch']} "
                f"group score={0.5 * (1.0 - nmae) + 0.5 * ficr:.6f} "
                f"nMAE={nmae:.6f} FiCR={ficr:.6f}",
                flush=True,
            )

            completed_outer_folds += 1
            if args.smoke_test:
                print("smoke test complete after one outer fold", flush=True)
                break

        del base_features, scada_hourly, reference
        release_cuda()
        if args.smoke_test and completed_outer_folds:
            break

    predictions = pd.concat(group_prediction_parts, ignore_index=True)
    turbine_predictions = pd.concat(turbine_prediction_parts, ignore_index=True)
    folds = fold_score_rows(predictions)
    wind_scores = wind_score_rows(turbine_predictions)
    prefix = args.results_dir / args.stem
    predictions.to_csv(f"{prefix}_predictions.csv", index=False, encoding="utf-8-sig")
    turbine_predictions.to_csv(
        f"{prefix}_turbine_predictions.csv", index=False, encoding="utf-8-sig"
    )
    folds.to_csv(f"{prefix}_fold_scores.csv", index=False, encoding="utf-8-sig")
    wind_scores.to_csv(
        f"{prefix}_wind_scores.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(training_rows).to_csv(
        f"{prefix}_training.csv", index=False, encoding="utf-8-sig"
    )
    if selection_parts:
        pd.concat(selection_parts, ignore_index=True).to_csv(
            f"{prefix}_optimal_grid_selection.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if set(predictions["group"].unique()) == set(TARGET_COLS):
        summary, pooled = pooled_oof_summary(predictions)
        summary.to_csv(f"{prefix}_summary.csv", index=False, encoding="utf-8-sig")
        pooled.to_csv(
            f"{prefix}_pooled_group_scores.csv", index=False, encoding="utf-8-sig"
        )
        print("\n=== pooled official OOF ===", flush=True)
        print(summary.to_string(index=False), flush=True)
        print("\n", pooled.to_string(index=False), sep="", flush=True)
    print("\n=== outer-year diagnostics ===", flush=True)
    print(folds.to_string(index=False), flush=True)
    print("\n=== pooled wind diagnostics ===", flush=True)
    print(
        wind_scores.loc[wind_scores["scope"].eq("pooled")].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()

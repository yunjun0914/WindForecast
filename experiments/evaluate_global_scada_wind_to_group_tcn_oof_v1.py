from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import _bootstrap  # noqa: F401
from experiments import evaluate_two_stage_scada_cubic_tcn_oof_v3 as wind_patch
from models.issue_block_tcn import IssueBlockTCN
from utils.decision_reward import smooth_ficr_reward
from utils.metrics import (
    GROUP_CAPACITY_KWH,
    TARGET_COLS,
    group_nmae_ficr,
    pooled_oof_summary,
)
from utils.per_turbine_features import get_or_build_group_feature_cache
from utils.per_turbine_optimal_grid import optimal_grid_input_columns
from utils.per_turbine_optimal_grid_builder import build_wind_candidate_matrix
from utils.per_turbine_scada import build_turbine_scada_hourly
from utils.per_turbine_sequence import SequenceStandardScaler


base = wind_patch.base

STAGE1_CACHE_VERSION = "global_scada_wind_stage1_cubic_mae_v1"
RUNNER_VERSION = "global_17wind_to_3group_pure_band_v1"
BAND_TEMPERATURE_START = 0.10
BAND_TEMPERATURE_END = 0.01
BAND_MIN_EPOCHS = 20


def band_temperature(epoch: int, maximum_epochs: int) -> float:
    progress = (epoch - 1) / max(maximum_epochs - 1, 1)
    ratio = BAND_TEMPERATURE_END / BAND_TEMPERATURE_START
    return float(BAND_TEMPERATURE_START * ratio**progress)


def assert_time_alignment(
    reference: base.TurbinePanel,
    candidate: base.TurbinePanel,
    group: str,
) -> None:
    if not np.array_equal(reference.issue_times, candidate.issue_times):
        raise ValueError(f"Issue order mismatch across groups: {group}")
    if not np.array_equal(reference.forecast_times, candidate.forecast_times):
        raise ValueError(f"Forecast order mismatch across groups: {group}")
    if not np.array_equal(reference.years, candidate.years):
        raise ValueError(f"Year order mismatch across groups: {group}")


def _json_default(value):
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON encode {type(value)!r}")


def stage1_cache_paths(
    group: str,
    pred_year: int,
    panel: base.TurbinePanel,
    args,
) -> tuple[Path, Path, Path]:
    observed = np.isfinite(panel.wind).any(axis=(1, 2))
    payload = {
        "version": STAGE1_CACHE_VERSION,
        "group": group,
        "pred_year": int(pred_year),
        "years": sorted(np.unique(panel.years).astype(int).tolist()),
        "wind_epochs": int(args.wind_epochs),
        "wind_patience": int(args.wind_patience),
        "wind_hidden_size": int(args.wind_hidden_size),
        "wind_num_layers": int(args.wind_num_layers),
        "wind_kernel_size": int(args.wind_kernel_size),
        "wind_dropout": float(args.wind_dropout),
        "wind_lr": float(args.wind_lr),
        "wind_weight_decay": float(args.wind_weight_decay),
        "wind_min_delta": float(args.wind_min_delta),
        "batch_size": int(args.batch_size),
        "within_year_folds": int(args.within_year_folds),
        "epoch_val_fraction": float(args.epoch_val_fraction),
        "seed": int(args.seed),
        "forecast_hash": hashlib.sha1(
            np.asarray(panel.forecast_times, dtype="datetime64[ns]")
            .astype(np.int64)
            .tobytes()
        ).hexdigest(),
        "observed_hash": hashlib.sha1(observed.tobytes()).hexdigest(),
    }
    digest = hashlib.sha1(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]
    cache_dir = args.cache_root / STAGE1_CACHE_VERSION
    prefix = cache_dir / f"{group}_pred{pred_year}_{digest}"
    return (
        Path(f"{prefix}.npz"),
        Path(f"{prefix}.json"),
        Path(f"{prefix}_selection.csv"),
    )


def generate_stage1_wind(
    base_features: pd.DataFrame,
    candidates,
    scada_hourly: pd.DataFrame,
    labels: pd.DataFrame,
    group: str,
    group_index: int,
    pred_year: int,
    requested_years: list[int],
    reference: base.TurbinePanel,
    args,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float, list[dict[str, object]], list[pd.DataFrame]]:
    outer_train_indices = np.flatnonzero(
        np.isin(reference.years, [year for year in requested_years if year != pred_year])
    )
    outer_val_indices = np.flatnonzero(reference.years == pred_year)
    observed_issue = np.isfinite(reference.wind).any(axis=(1, 2))
    wind_fit_indices = outer_train_indices[observed_issue[outer_train_indices]]
    unlabeled_train_indices = outer_train_indices[~observed_issue[outer_train_indices]]
    if len(wind_fit_indices) < 20 or len(outer_val_indices) < 20:
        raise ValueError(
            f"Insufficient stage1 issues {group} pred{pred_year}: "
            f"fit={len(wind_fit_indices)} val={len(outer_val_indices)}"
        )

    npz_path, json_path, selection_path = stage1_cache_paths(
        group, pred_year, reference, args
    )
    if (
        npz_path.exists()
        and json_path.exists()
        and not args.rebuild_stage_cache
    ):
        with np.load(npz_path, allow_pickle=False) as cached:
            predicted_wind = cached["predicted_wind"].astype(np.float32)
            outer_optgrid = cached["outer_optgrid"].astype(np.float32)
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
        training_rows = metadata.get("training_rows", [])
        for row in training_rows:
            row["stage1_cache_hit"] = True
        selection_parts = (
            [pd.read_csv(selection_path)] if selection_path.exists() else []
        )
        relevant = np.union1d(outer_train_indices, outer_val_indices)
        if predicted_wind.shape != reference.wind.shape:
            raise ValueError(f"Cached stage1 shape mismatch: {npz_path}")
        if not np.isfinite(predicted_wind[relevant]).all():
            raise ValueError(f"Cached stage1 contains missing predictions: {npz_path}")
        print(
            f"  {group} stage1 cache hit: fit={len(wind_fit_indices)} "
            f"unlabeled_train={len(unlabeled_train_indices)}",
            flush=True,
        )
        return (
            predicted_wind,
            outer_optgrid,
            float(metadata["wind_scale_p99"]),
            training_rows,
            selection_parts,
        )

    crossfit_wind = np.full(reference.wind.shape, np.nan, dtype=np.float32)
    training_rows: list[dict[str, object]] = []
    selection_parts: list[pd.DataFrame] = []
    crossfit_folds = base.make_crossfit_folds(
        wind_fit_indices,
        reference.years,
        reference.issue_times,
        args.within_year_folds,
    )
    for fold_index, (fold_name, heldout_indices) in enumerate(crossfit_folds):
        wind_train_indices = np.setdiff1d(
            wind_fit_indices, heldout_indices, assume_unique=False
        )
        wind_train_indices = base.remove_forecast_overlap(
            wind_train_indices, heldout_indices, reference.forecast_times
        )
        fit_times = reference.forecast_times[wind_train_indices]
        cache_label = f"pred{pred_year}_oof_{fold_name}"
        panel, selections = base.build_cubic_feature_panel(
            base_features,
            candidates,
            scada_hourly,
            labels,
            group,
            fit_times,
            cache_label,
            args,
        )
        base.assert_panel_alignment(reference, panel)
        prediction, stats = base.select_and_refit_wind(
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
                "train_years": ",".join(
                    map(
                        str,
                        sorted(
                            np.unique(reference.years[wind_train_indices])
                            .astype(int)
                            .tolist()
                        ),
                    )
                ),
                "stage1_cache_hit": False,
            }
        )
        training_rows.append(stats)
        selections["group"] = group
        selections["pred_year"] = pred_year
        selections["role"] = f"stage1_oof_{fold_name}"
        selection_parts.append(selections)
        metrics = base.wind_metrics(
            reference.wind[heldout_indices],
            prediction,
            float(stats["wind_scale_p99"]),
        )
        print(
            f"  {group} stage1 OOF {fold_name}: "
            f"train={len(wind_train_indices)} heldout={len(heldout_indices)} "
            f"epoch={stats['best_epoch']} wind_MAE={metrics['wind_mae']:.4f}",
            flush=True,
        )

    if not np.isfinite(crossfit_wind[wind_fit_indices]).all():
        missing = int(
            np.count_nonzero(~np.isfinite(crossfit_wind[wind_fit_indices]))
        )
        raise ValueError(f"Incomplete stage1 cross-fit wind: {group} missing={missing}")

    final_fit_times = reference.forecast_times[wind_fit_indices]
    final_panel, selections = base.build_cubic_feature_panel(
        base_features,
        candidates,
        scada_hourly,
        labels,
        group,
        final_fit_times,
        f"pred{pred_year}_outer",
        args,
    )
    base.assert_panel_alignment(reference, final_panel)
    final_predict_indices = np.union1d(unlabeled_train_indices, outer_val_indices)
    final_prediction, wind_stats = base.select_and_refit_wind(
        final_panel,
        wind_fit_indices,
        final_predict_indices,
        args,
        device,
        args.seed + group_index * 100000 + pred_year * 100 + 90,
    )
    predicted_wind = np.full(reference.wind.shape, np.nan, dtype=np.float32)
    predicted_wind[wind_fit_indices] = crossfit_wind[wind_fit_indices]
    predicted_wind[final_predict_indices] = final_prediction
    relevant = np.union1d(outer_train_indices, outer_val_indices)
    if not np.isfinite(predicted_wind[relevant]).all():
        missing = int(np.count_nonzero(~np.isfinite(predicted_wind[relevant])))
        raise ValueError(f"Incomplete joint stage1 input: {group} missing={missing}")

    wind_stats.update(
        {
            "group": group,
            "pred_year": pred_year,
            "role": "stage1_outer_and_unlabeled_inference",
            "train_years": ",".join(
                map(
                    str,
                    sorted(
                        np.unique(reference.years[wind_fit_indices])
                        .astype(int)
                        .tolist()
                    ),
                )
            ),
            "n_unlabeled_train_inference_issues": int(
                len(unlabeled_train_indices)
            ),
            "stage1_cache_hit": False,
        }
    )
    training_rows.append(wind_stats)
    selections["group"] = group
    selections["pred_year"] = pred_year
    selections["role"] = "stage1_outer"
    selection_parts.append(selections)

    base_index = final_panel.feature_cols.index("optgrid_ws_calibrated")
    outer_optgrid = final_panel.features[
        outer_val_indices, :, :, base_index
    ].astype(np.float32)
    outer_prediction = predicted_wind[outer_val_indices]
    outer_metrics = base.wind_metrics(
        reference.wind[outer_val_indices],
        outer_prediction,
        float(wind_stats["wind_scale_p99"]),
    )
    print(
        f"  {group} stage1 OUTER: fit={len(wind_fit_indices)} "
        f"unlabeled_train={len(unlabeled_train_indices)} "
        f"epoch={wind_stats['best_epoch']} wind_MAE={outer_metrics['wind_mae']:.4f}",
        flush=True,
    )

    npz_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_npz = Path(f"{npz_path}.tmp.npz")
    np.savez_compressed(
        tmp_npz,
        predicted_wind=predicted_wind,
        outer_optgrid=outer_optgrid,
    )
    tmp_npz.replace(npz_path)
    metadata = {
        "runner_version": RUNNER_VERSION,
        "group": group,
        "pred_year": int(pred_year),
        "wind_scale_p99": float(wind_stats["wind_scale_p99"]),
        "training_rows": training_rows,
    }
    json_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    if selection_parts:
        pd.concat(selection_parts, ignore_index=True).to_csv(
            selection_path, index=False, encoding="utf-8-sig"
        )
    return (
        predicted_wind,
        outer_optgrid,
        float(wind_stats["wind_scale_p99"]),
        training_rows,
        selection_parts,
    )


def make_joint_epoch_split(
    train_indices: np.ndarray,
    years: np.ndarray,
    issue_times: np.ndarray,
    forecast_times: np.ndarray,
    scored: np.ndarray,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    train_indices = np.asarray(train_indices, dtype=int)
    g3_has_target = scored[:, :, 2].any(axis=1)
    g3_years = sorted(
        np.unique(years[train_indices[g3_has_target[train_indices]]])
        .astype(int)
        .tolist()
    )
    if not g3_years:
        raise ValueError("No group3 targets in joint TCN2 outer training split")
    latest_g3_year = g3_years[-1]
    latest_g3_indices = train_indices[
        (years[train_indices] == latest_g3_year)
        & g3_has_target[train_indices]
    ]
    if len(g3_years) >= 2:
        val_indices = latest_g3_indices
    else:
        ordered = latest_g3_indices[np.argsort(issue_times[latest_g3_indices])]
        n_val = max(1, int(round(len(ordered) * val_fraction)))
        n_val = min(n_val, max(1, len(ordered) // 2))
        val_indices = ordered[-n_val:]
    fit_indices = np.setdiff1d(train_indices, val_indices, assume_unique=False)
    fit_indices = base.remove_forecast_overlap(
        fit_indices, val_indices, forecast_times
    )
    if len(fit_indices) < 20 or len(val_indices) < 5:
        raise ValueError(
            f"Insufficient joint epoch split: train={len(fit_indices)} "
            f"val={len(val_indices)}"
        )
    for group_index, group in enumerate(TARGET_COLS):
        if not scored[fit_indices, :, group_index].any():
            raise ValueError(f"No {group} target in joint inner training split")
        if not scored[val_indices, :, group_index].any():
            raise ValueError(f"No {group} target in joint inner validation split")
    return fit_indices, val_indices


def make_joint_loader(
    features: np.ndarray,
    target: np.ndarray,
    scored: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(features[indices].astype(np.float32)),
        torch.from_numpy(target[indices].astype(np.float32)),
        torch.from_numpy(scored[indices].astype(np.float32)),
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def bounded_group_prediction(
    model: IssueBlockTCN,
    features: torch.Tensor,
    output_floor: float,
) -> torch.Tensor:
    return output_floor + (1.0 - output_floor) * torch.sigmoid(model(features))


def initialize_group_heads(
    model: IssueBlockTCN,
    mean_targets: np.ndarray,
    output_floor: float,
) -> None:
    for group_index, mean_target in enumerate(mean_targets):
        final = model.heads[group_index][-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("Unexpected IssueBlockTCN group head")
        normalized = (float(mean_target) - output_floor) / (1.0 - output_floor)
        normalized = float(np.clip(normalized, 0.02, 0.98))
        bias = math.log(normalized / (1.0 - normalized))
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, bias)


def new_joint_model(
    args,
    device: torch.device,
    mean_targets: np.ndarray,
    output_floor: float,
) -> IssueBlockTCN:
    model = IssueBlockTCN(
        input_size=17,
        output_size=3,
        hidden_size=args.power_hidden_size,
        num_layers=args.power_num_layers,
        kernel_size=args.power_kernel_size,
        dropout=args.power_dropout,
        full_context=True,
    ).to(device)
    initialize_group_heads(model, mean_targets, output_floor)
    return model


def split_target_statistics(
    target: np.ndarray,
    scored: np.ndarray,
    indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    means = []
    valid_fractions = []
    for group_index, group in enumerate(TARGET_COLS):
        group_mask = scored[indices, :, group_index].astype(bool)
        if not group_mask.any():
            raise ValueError(f"No scored rows for {group}")
        means.append(
            float(target[indices, :, group_index][group_mask].mean())
        )
        valid_fractions.append(float(group_mask.mean()))
    return np.asarray(means, dtype=np.float32), np.asarray(
        valid_fractions, dtype=np.float32
    )


def train_joint_epoch(
    model: IssueBlockTCN,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    mean_targets: np.ndarray,
    valid_fractions: np.ndarray,
    temperature: float,
    output_floor: float,
    args,
    device: torch.device,
) -> float:
    model.train()
    losses: list[float] = []
    mean_targets_t = torch.as_tensor(mean_targets, device=device)
    valid_fractions_t = torch.as_tensor(valid_fractions, device=device)
    for xb, yb, mb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        prediction = bounded_group_prediction(model, xb, output_floor)
        group_losses = []
        batch_rows = float(mb.shape[0] * mb.shape[1])
        for group_index in range(len(TARGET_COLS)):
            reward = smooth_ficr_reward(
                yb[..., group_index],
                prediction[..., group_index],
                float(mean_targets_t[group_index].item()),
                temperature,
            )
            denominator = batch_rows * valid_fractions_t[group_index]
            group_losses.append(
                -(reward * mb[..., group_index]).sum()
                / torch.clamp(denominator, min=1.0)
            )
        loss = torch.stack(group_losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def predict_joint(
    model: IssueBlockTCN,
    features: np.ndarray,
    batch_size: int,
    output_floor: float,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            xb = torch.from_numpy(features[start : start + batch_size]).to(device)
            parts.append(
                bounded_group_prediction(model, xb, output_floor).cpu().numpy()
            )
    return np.concatenate(parts, axis=0).astype(np.float32)


def joint_hard_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    scored: np.ndarray,
    indices: np.ndarray,
) -> dict[str, object]:
    group_nmaes = []
    group_ficrs = []
    result: dict[str, object] = {}
    for group_index, group in enumerate(TARGET_COLS):
        mask = scored[indices, :, group_index].astype(bool)
        capacity = float(GROUP_CAPACITY_KWH[group])
        actual = target[indices, :, group_index][mask] * capacity
        forecast = prediction[:, :, group_index][mask] * capacity
        nmae, ficr = group_nmae_ficr(actual, forecast, capacity)
        group_nmaes.append(float(nmae))
        group_ficrs.append(float(ficr))
        result[f"{group}_nmae"] = float(nmae)
        result[f"{group}_ficr"] = float(ficr)
    mean_nmae = float(np.mean(group_nmaes))
    mean_ficr = float(np.mean(group_ficrs))
    result.update(
        {
            "mean_nmae": mean_nmae,
            "mean_ficr": mean_ficr,
            "mean_score": 0.5 * (1.0 - mean_nmae) + 0.5 * mean_ficr,
        }
    )
    return result


def select_and_refit_joint_tcn2(
    wind_features: np.ndarray,
    official_kwh: np.ndarray,
    years: np.ndarray,
    issue_times: np.ndarray,
    forecast_times: np.ndarray,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    args,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    capacities = np.asarray(
        [GROUP_CAPACITY_KWH[group] for group in TARGET_COLS], dtype=np.float32
    )
    target = official_kwh / capacities.reshape(1, 1, -1)
    scored = np.isfinite(official_kwh) & (
        official_kwh
        >= capacities.reshape(1, 1, -1) * args.target_min_output_ratio
    )
    target = np.nan_to_num(target, nan=0.0).astype(np.float32)
    scored = scored.astype(np.float32)
    output_floor = float(args.target_min_output_ratio)

    epoch_train, epoch_val = make_joint_epoch_split(
        train_indices,
        years,
        issue_times,
        forecast_times,
        scored,
        args.epoch_val_fraction,
    )
    select_scaler = SequenceStandardScaler()
    select_scaler.fit(wind_features[epoch_train])
    select_features = select_scaler.transform(wind_features)
    mean_targets, valid_fractions = split_target_statistics(
        target, scored, epoch_train
    )
    loader = make_joint_loader(
        select_features,
        target,
        scored,
        epoch_train,
        args.power_batch_size,
        device,
        seed,
    )
    base.set_seed(seed)
    model = new_joint_model(args, device, mean_targets, output_floor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.power_lr, weight_decay=args.power_weight_decay
    )

    best_epoch = 0
    best_ficr = -np.inf
    best_metrics: dict[str, object] = {}
    best_temperature = np.nan
    best_train_loss = np.nan
    bad_epochs = 0
    epochs_trained = 0
    for epoch in range(1, args.power_epochs + 1):
        temperature = band_temperature(epoch, args.power_epochs)
        train_loss = train_joint_epoch(
            model,
            loader,
            optimizer,
            mean_targets,
            valid_fractions,
            temperature,
            output_floor,
            args,
            device,
        )
        val_prediction = predict_joint(
            model,
            select_features[epoch_val],
            args.eval_batch_size,
            output_floor,
            device,
        )
        metrics = joint_hard_metrics(
            val_prediction, target, scored, epoch_val
        )
        epochs_trained = epoch
        mean_ficr = float(metrics["mean_ficr"])
        if mean_ficr > best_ficr:
            best_epoch = epoch
            best_ficr = mean_ficr
            best_metrics = metrics
            best_temperature = temperature
            best_train_loss = train_loss
            bad_epochs = 0
        elif epoch >= BAND_MIN_EPOCHS:
            bad_epochs += 1
        if epoch >= BAND_MIN_EPOCHS and bad_epochs >= args.power_patience:
            break
    if best_epoch <= 0 or not np.isfinite(best_ficr):
        raise RuntimeError("Joint TCN2 epoch selection failed")
    del model, optimizer, loader
    base.release_cuda()

    final_scaler = SequenceStandardScaler()
    final_scaler.fit(wind_features[train_indices])
    final_features = final_scaler.transform(wind_features)
    full_means, full_valid_fractions = split_target_statistics(
        target, scored, train_indices
    )
    final_loader = make_joint_loader(
        final_features,
        target,
        scored,
        train_indices,
        args.power_batch_size,
        device,
        seed + 1,
    )
    base.set_seed(seed + 1)
    final_model = new_joint_model(args, device, full_means, output_floor)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=args.power_lr,
        weight_decay=args.power_weight_decay,
    )
    final_train_loss = np.nan
    for epoch in range(1, best_epoch + 1):
        final_train_loss = train_joint_epoch(
            final_model,
            final_loader,
            final_optimizer,
            full_means,
            full_valid_fractions,
            band_temperature(epoch, args.power_epochs),
            output_floor,
            args,
            device,
        )
    prediction = predict_joint(
        final_model,
        final_features[predict_indices],
        args.eval_batch_size,
        output_floor,
        device,
    )
    stats: dict[str, object] = {
        "stage": "joint_group_power",
        "role": "stage2_outer",
        "architecture": "IssueBlockTCN_17_predicted_winds_to_3_groups",
        "loss": "equal_group_pure_band_ficr",
        "loss_formula": "mean_g[-sum(mask*y/mean_y*(.75*sigmoid((.08-e)/T)+.25*sigmoid((.06-e)/T)))/(batch_rows*q_g)]",
        "checkpoint_metric": "inner_validation_equal_group_mean_hard_ficr",
        "output_transform": f"{output_floor:.2f}+(1-{output_floor:.2f})*sigmoid(raw)",
        "head_initialization": "zero_weight_group_mean_logit",
        "best_epoch": int(best_epoch),
        "epochs_trained": int(epochs_trained),
        "best_epoch_temperature": float(best_temperature),
        "best_epoch_train_loss": float(best_train_loss),
        "final_train_loss": float(final_train_loss),
        "n_epoch_train_issues": int(len(epoch_train)),
        "n_epoch_val_issues": int(len(epoch_val)),
        "n_refit_issues": int(len(train_indices)),
        "n_predict_issues": int(len(predict_indices)),
        "n_parameters": int(sum(p.numel() for p in final_model.parameters())),
        "band_temperature_start": BAND_TEMPERATURE_START,
        "band_temperature_end": BAND_TEMPERATURE_END,
        "band_min_epochs": BAND_MIN_EPOCHS,
    }
    for key, value in best_metrics.items():
        stats[f"epoch_val_{key}"] = value
    for group_index, group in enumerate(TARGET_COLS):
        stats[f"{group}_mean_target_inner"] = float(mean_targets[group_index])
        stats[f"{group}_valid_fraction_inner"] = float(
            valid_fractions[group_index]
        )
        stats[f"{group}_mean_target_refit"] = float(full_means[group_index])
        stats[f"{group}_valid_fraction_refit"] = float(
            full_valid_fractions[group_index]
        )
    del final_model, final_optimizer, final_loader
    base.release_cuda()
    return prediction, stats


def main() -> None:
    args = base.parse_args()
    groups = base.parse_csv(args.groups)
    requested_years = [int(value) for value in base.parse_csv(args.years)]
    if groups != TARGET_COLS:
        raise ValueError(
            "This joint runner requires --groups in canonical order: "
            + ",".join(TARGET_COLS)
        )
    if len(requested_years) < 3:
        raise ValueError("Joint OOF requires all three requested train years")
    if args.within_year_folds < 2:
        raise ValueError("--within-year-folds must be at least 2")
    if not 0.05 <= args.epoch_val_fraction <= 0.50:
        raise ValueError("--epoch-val-fraction must be in [0.05, 0.50]")
    if args.power_epochs < BAND_MIN_EPOCHS:
        raise ValueError(f"--power-epochs must be at least {BAND_MIN_EPOCHS}")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} years={requested_years} "
        f"TCN1=cubic-MAE h{args.wind_hidden_size}/L{args.wind_num_layers}; "
        f"TCN2=17wind->3group h{args.power_hidden_size}/L{args.power_num_layers} "
        "equal-group pure-band no-DOY",
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
    raw_scada_by_group = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }
    scada_hourly_by_group = {
        group: build_turbine_scada_hourly(raw_scada_by_group[group], group)
        for group in TARGET_COLS
    }
    candidates = build_wind_candidate_matrix(ldaps, gfs)
    print(
        f"candidate_rows={len(candidates.keys)} candidates={len(candidates.names)}",
        flush=True,
    )

    variant = "global_17wind_to_3group_pure_band_tcn"
    group_prediction_parts: list[pd.DataFrame] = []
    turbine_wind_parts: list[pd.DataFrame] = []
    training_rows: list[dict[str, object]] = []
    selection_parts: list[pd.DataFrame] = []
    completed_outer_folds = 0

    for pred_year in requested_years:
        print(f"\n=== global outer pred_year={pred_year} ===", flush=True)
        predicted_wind_by_group: dict[str, np.ndarray] = {}
        official_by_group: dict[str, np.ndarray] = {}
        canonical_reference: base.TurbinePanel | None = None

        for group_index, group in enumerate(TARGET_COLS):
            base_features = get_or_build_group_feature_cache(
                ldaps,
                gfs,
                group,
                cache_root=args.cache_root,
                rebuild=args.rebuild_feature_cache,
            )
            reference_feature = optimal_grid_input_columns(group)[0]
            reference = base.build_panel(
                base_features,
                scada_hourly_by_group[group],
                labels,
                group,
                [reference_feature],
            )
            if canonical_reference is None:
                canonical_reference = reference
            else:
                assert_time_alignment(canonical_reference, reference, group)
            outer_val_indices = np.flatnonzero(reference.years == pred_year)
            (
                predicted_wind,
                outer_optgrid,
                wind_scale,
                stage1_training,
                stage1_selections,
            ) = generate_stage1_wind(
                base_features,
                candidates,
                scada_hourly_by_group[group],
                labels,
                group,
                group_index,
                pred_year,
                requested_years,
                reference,
                args,
                device,
            )
            predicted_wind_by_group[group] = predicted_wind
            official_by_group[group] = reference.official.copy()
            training_rows.extend(stage1_training)
            selection_parts.extend(stage1_selections)

            for turbine_index, turbine in enumerate(reference.turbines):
                turbine_wind_parts.append(
                    pd.DataFrame(
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
                            "optgrid_ws_calibrated": outer_optgrid[
                                :, turbine_index
                            ].reshape(-1),
                            "predicted_scada_ws": predicted_wind[
                                outer_val_indices, turbine_index
                            ].reshape(-1),
                            "wind_scale_p99": wind_scale,
                        }
                    )
                )
            del base_features, reference
            base.release_cuda()

        if canonical_reference is None:
            raise RuntimeError("No canonical issue panel")
        outer_train_indices = np.flatnonzero(
            np.isin(
                canonical_reference.years,
                [year for year in requested_years if year != pred_year],
            )
        )
        outer_val_indices = np.flatnonzero(
            canonical_reference.years == pred_year
        )
        wind17 = np.concatenate(
            [predicted_wind_by_group[group] for group in TARGET_COLS], axis=1
        )
        if wind17.shape[1:] != (17, 24):
            raise ValueError(f"Expected [issue,17,24] wind, got {wind17.shape}")
        wind_features = wind17.transpose(0, 2, 1).astype(np.float32)
        official_kwh = np.stack(
            [official_by_group[group] for group in TARGET_COLS], axis=-1
        ).astype(np.float32)
        group_prediction_norm, power_stats = select_and_refit_joint_tcn2(
            wind_features,
            official_kwh,
            canonical_reference.years,
            canonical_reference.issue_times,
            canonical_reference.forecast_times,
            outer_train_indices,
            outer_val_indices,
            args,
            device,
            args.seed + pred_year * 1000 + 777,
        )
        power_stats.update(
            {
                "pred_year": pred_year,
                "train_years": ",".join(
                    map(
                        str,
                        [year for year in requested_years if year != pred_year],
                    )
                ),
            }
        )
        training_rows.append(power_stats)

        for group_index, group in enumerate(TARGET_COLS):
            capacity = float(GROUP_CAPACITY_KWH[group])
            prediction_kwh = (
                group_prediction_norm[:, :, group_index] * capacity
            )
            group_part = pd.DataFrame(
                {
                    "forecast_kst_dtm": canonical_reference.forecast_times[
                        outer_val_indices
                    ].reshape(-1),
                    "issue_kst_dtm": np.repeat(
                        canonical_reference.issue_times[outer_val_indices], 24
                    ),
                    "group": group,
                    "pred_year": pred_year,
                    "official_target": official_kwh[
                        outer_val_indices, :, group_index
                    ].reshape(-1),
                    "pred": prediction_kwh.reshape(-1),
                    "variant": variant,
                }
            ).dropna(subset=["official_target"])
            if group_part.duplicated(["forecast_kst_dtm", "group"]).any():
                raise ValueError(
                    f"Duplicate outer predictions for {group} pred{pred_year}"
                )
            group_prediction_parts.append(group_part)
            if len(group_part):
                nmae, ficr = group_nmae_ficr(
                    group_part["official_target"], group_part["pred"], capacity
                )
                print(
                    f"  TCN2 {group}: score={0.5 * (1.0 - nmae) + 0.5 * ficr:.6f} "
                    f"nMAE={nmae:.6f} FiCR={ficr:.6f}",
                    flush=True,
                )
        print(
            f"  TCN2 selected epoch={power_stats['best_epoch']} "
            f"inner mean FiCR={power_stats['epoch_val_mean_ficr']:.6f}",
            flush=True,
        )

        completed_outer_folds += 1
        if args.smoke_test:
            print("smoke test complete after one global outer fold", flush=True)
            break

    predictions = pd.concat(group_prediction_parts, ignore_index=True)
    turbine_predictions = pd.concat(turbine_wind_parts, ignore_index=True)
    folds = base.fold_score_rows(predictions)
    wind_scores = base.wind_score_rows(turbine_predictions)
    prefix = args.results_dir / args.stem
    predictions.to_csv(
        f"{prefix}_predictions.csv", index=False, encoding="utf-8-sig"
    )
    turbine_predictions.to_csv(
        f"{prefix}_turbine_predictions.csv", index=False, encoding="utf-8-sig"
    )
    folds.to_csv(
        f"{prefix}_fold_scores.csv", index=False, encoding="utf-8-sig"
    )
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
        summary.to_csv(
            f"{prefix}_summary.csv", index=False, encoding="utf-8-sig"
        )
        pooled.to_csv(
            f"{prefix}_pooled_group_scores.csv",
            index=False,
            encoding="utf-8-sig",
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

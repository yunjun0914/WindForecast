from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _bootstrap  # noqa: F401
from experiments import evaluate_global_scada_wind_to_group_tcn_oof_v1 as proto
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


base = proto.base
RUNNER_VERSION = "global_17wind_to_3group_foldbest_v1"


def raw_wind_anchor(
    panel: base.TurbinePanel,
    issue_indices: np.ndarray,
    base_index: int,
) -> np.ndarray:
    values = panel.features[issue_indices, :, :, base_index]
    return values.reshape(len(issue_indices) * len(panel.turbines), 24).astype(
        np.float32
    )


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def train_stage1_fold_best(
    panel: base.TurbinePanel,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    predict_indices: np.ndarray,
    group: str,
    val_year: int,
    cache_label: str,
    args,
    device: torch.device,
    seed: int,
    checkpoint_dir: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    """Train once, retain the actual outer-validation best checkpoint, no refit."""
    base_index = panel.feature_cols.index("optgrid_ws_calibrated")
    observed_train = panel.wind[train_indices]
    valid_train = observed_train[np.isfinite(observed_train)]
    if len(valid_train) < 500:
        raise ValueError(f"Too few wind targets for {group}/val{val_year}")
    scale = max(float(np.quantile(valid_train, 0.99)), 5.0)

    scaler = SequenceStandardScaler()
    scaler.fit(panel.features[train_indices])
    scaled_panel = copy.copy(panel)
    scaled_panel.features = scaler.transform(panel.features)
    x_train, y_train, _, m_train = base.flatten_wind_data(
        scaled_panel, train_indices, base_index
    )
    x_val, _, _, _ = base.flatten_wind_data(
        scaled_panel, val_indices, base_index
    )
    x_predict, _, _, _ = base.flatten_wind_data(
        scaled_panel, predict_indices, base_index
    )
    b_train = raw_wind_anchor(panel, train_indices, base_index)
    b_val = raw_wind_anchor(panel, val_indices, base_index)
    b_predict = raw_wind_anchor(panel, predict_indices, base_index)
    loader = base.wind_loader(
        x_train,
        y_train,
        b_train,
        m_train,
        args.batch_size,
        device,
    )

    base.set_seed(seed)
    model = base.new_wind_model(panel.features.shape[-1], args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.wind_lr,
        weight_decay=args.wind_weight_decay,
    )
    best_epoch = 0
    best_cubic_mae = np.inf
    best_wind_mae = np.nan
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    epochs_trained = 0
    for epoch in range(1, args.wind_epochs + 1):
        base.train_wind_epochs(model, loader, optimizer, 1, scale, args, device)
        val_prediction = base.predict_wind_flat(
            model, x_val, b_val, args.eval_batch_size, device
        )
        actual = panel.wind[val_indices].reshape(val_prediction.shape)
        valid = np.isfinite(actual) & np.isfinite(val_prediction)
        if not valid.any():
            raise ValueError(f"No wind validation rows for {group}/val{val_year}")
        cubic_difference = (val_prediction[valid] / scale) ** 3 - (
            actual[valid] / scale
        ) ** 3
        val_cubic_mae = float(np.mean(np.abs(cubic_difference)))
        epochs_trained = epoch
        if val_cubic_mae < best_cubic_mae - args.wind_min_delta:
            best_epoch = epoch
            best_cubic_mae = val_cubic_mae
            best_wind_mae = float(
                np.mean(np.abs(val_prediction[valid] - actual[valid]))
            )
            best_state = cpu_state_dict(model)
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.wind_patience:
            break
    if best_state is None or best_epoch <= 0:
        raise RuntimeError(f"Stage1 fold-best selection failed: {group}/val{val_year}")

    model.load_state_dict(best_state)
    flat_prediction = base.predict_wind_flat(
        model, x_predict, b_predict, args.eval_batch_size, device
    )
    prediction = flat_prediction.reshape(
        len(predict_indices), len(panel.turbines), 24
    ).astype(np.float32)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"stage1_{group}_val{val_year}.pt"
    torch.save(
        {
            "runner_version": RUNNER_VERSION,
            "stage": "TCN1",
            "group": group,
            "validation_year": int(val_year),
            "best_epoch": int(best_epoch),
            "wind_scale_p99": float(scale),
            "cache_label": cache_label,
            "model_state_dict": best_state,
            "scaler_mean": scaler.mean_,
            "scaler_std": scaler.std_,
            "feature_cols": panel.feature_cols,
            "turbines": panel.turbines,
        },
        checkpoint_path,
    )
    stats: dict[str, object] = {
        "stage": "wind",
        "role": "stage1_outer_fold_best_checkpoint",
        "protocol": "non_nested_outer_validation_checkpoint_no_refit",
        "group": group,
        "pred_year": int(val_year),
        "loss": "normalized_cubic_mae",
        "best_epoch": int(best_epoch),
        "epochs_trained": int(epochs_trained),
        "outer_val_cubic_mae": float(best_cubic_mae),
        "outer_val_wind_mae": float(best_wind_mae),
        "wind_scale_p99": float(scale),
        "n_train_issues": int(len(train_indices)),
        "n_val_issues": int(len(val_indices)),
        "n_predict_issues": int(len(predict_indices)),
        "n_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "checkpoint_path": str(checkpoint_path),
        "cache_label": cache_label,
    }
    del model, optimizer, loader, scaled_panel
    base.release_cuda()
    return prediction, stats


def build_stage1_group_oof(
    base_features: pd.DataFrame,
    candidates,
    scada_hourly: pd.DataFrame,
    labels: pd.DataFrame,
    group: str,
    group_index: int,
    requested_years: list[int],
    args,
    device: torch.device,
    checkpoint_dir: Path,
) -> tuple[
    base.TurbinePanel,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict[str, object]],
    list[pd.DataFrame],
]:
    reference_feature = optimal_grid_input_columns(group)[0]
    reference = base.build_panel(
        base_features,
        scada_hourly,
        labels,
        group,
        [reference_feature],
    )
    requested_indices = np.flatnonzero(np.isin(reference.years, requested_years))
    observed_issue = np.isfinite(reference.wind).any(axis=(1, 2))
    validation_years = [
        year
        for year in requested_years
        if observed_issue[reference.years == year].any()
    ]
    if group != "kpx_group_3" and validation_years != requested_years:
        raise ValueError(f"Unexpected stage1 wind coverage for {group}")
    if group == "kpx_group_3" and validation_years != [2023, 2024]:
        raise ValueError(f"Unexpected group3 wind years: {validation_years}")

    oof_wind = np.full(reference.wind.shape, np.nan, dtype=np.float32)
    oof_optgrid = np.full(reference.wind.shape, np.nan, dtype=np.float32)
    scale_by_issue = np.full(len(reference.years), np.nan, dtype=np.float32)
    missing_indices = requested_indices[~observed_issue[requested_indices]]
    missing_predictions: list[np.ndarray] = []
    missing_optgrids: list[np.ndarray] = []
    missing_scales: list[float] = []
    training_rows: list[dict[str, object]] = []
    selection_parts: list[pd.DataFrame] = []

    for val_year in validation_years:
        val_indices = np.flatnonzero(
            (reference.years == val_year) & observed_issue
        )
        train_indices = requested_indices[
            (reference.years[requested_indices] != val_year)
            & observed_issue[requested_indices]
        ]
        train_indices = base.remove_forecast_overlap(
            train_indices, val_indices, reference.forecast_times
        )
        if len(train_indices) < 20 or len(val_indices) < 20:
            raise ValueError(
                f"Insufficient non-nested stage1 fold {group}/val{val_year}: "
                f"train={len(train_indices)} val={len(val_indices)}"
            )
        cache_label = f"foldbest_val{val_year}"
        panel, selections = base.build_cubic_feature_panel(
            base_features,
            candidates,
            scada_hourly,
            labels,
            group,
            reference.forecast_times[train_indices],
            cache_label,
            args,
        )
        base.assert_panel_alignment(reference, panel)
        predict_indices = np.concatenate([val_indices, missing_indices])
        prediction, stats = train_stage1_fold_best(
            panel,
            train_indices,
            val_indices,
            predict_indices,
            group,
            val_year,
            cache_label,
            args,
            device,
            args.seed + group_index * 100000 + val_year * 100,
            checkpoint_dir,
        )
        n_val = len(val_indices)
        oof_wind[val_indices] = prediction[:n_val]
        base_index = panel.feature_cols.index("optgrid_ws_calibrated")
        oof_optgrid[val_indices] = panel.features[
            val_indices, :, :, base_index
        ]
        scale_by_issue[val_indices] = float(stats["wind_scale_p99"])
        if len(missing_indices):
            missing_predictions.append(prediction[n_val:])
            missing_optgrids.append(
                panel.features[missing_indices, :, :, base_index].astype(np.float32)
            )
            missing_scales.append(float(stats["wind_scale_p99"]))
        training_rows.append(stats)
        selections["group"] = group
        selections["pred_year"] = val_year
        selections["role"] = "stage1_outer_fold_best_checkpoint"
        selection_parts.append(selections)
        print(
            f"  TCN1 {group} val{val_year}: best_epoch={stats['best_epoch']} "
            f"wind_MAE={stats['outer_val_wind_mae']:.4f} no_refit",
            flush=True,
        )

    if len(missing_indices):
        if group != "kpx_group_3" or len(missing_predictions) != 2:
            raise ValueError(f"Unexpected missing-wind ensemble for {group}")
        oof_wind[missing_indices] = np.mean(
            np.stack(missing_predictions, axis=0), axis=0
        )
        oof_optgrid[missing_indices] = np.mean(
            np.stack(missing_optgrids, axis=0), axis=0
        )
        scale_by_issue[missing_indices] = float(np.mean(missing_scales))
        training_rows.append(
            {
                "stage": "wind",
                "role": "stage1_group3_2022_inference_ensemble",
                "protocol": "mean_of_val2023_and_val2024_fold_best_checkpoints",
                "group": group,
                "pred_year": 2022,
                "n_predict_issues": int(len(missing_indices)),
                "n_models": int(len(missing_predictions)),
            }
        )
        print(
            "  TCN1 kpx_group_3 2022: mean(val2023-best, val2024-best)",
            flush=True,
        )

    if not np.isfinite(oof_wind[requested_indices]).all():
        missing = int(np.count_nonzero(~np.isfinite(oof_wind[requested_indices])))
        raise ValueError(f"Incomplete stage1 OOF wind for {group}: {missing}")
    return (
        reference,
        oof_wind,
        oof_optgrid,
        scale_by_issue,
        training_rows,
        selection_parts,
    )


def available_group_hard_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    scored: np.ndarray,
    val_indices: np.ndarray,
) -> dict[str, object]:
    group_nmaes: list[float] = []
    group_ficrs: list[float] = []
    result: dict[str, object] = {}
    for group_index, group in enumerate(TARGET_COLS):
        mask = scored[val_indices, :, group_index].astype(bool)
        if not mask.any():
            result[f"{group}_nmae"] = np.nan
            result[f"{group}_ficr"] = np.nan
            continue
        capacity = float(GROUP_CAPACITY_KWH[group])
        actual = target[val_indices, :, group_index][mask] * capacity
        forecast = prediction[:, :, group_index][mask] * capacity
        nmae, ficr = group_nmae_ficr(actual, forecast, capacity)
        group_nmaes.append(float(nmae))
        group_ficrs.append(float(ficr))
        result[f"{group}_nmae"] = float(nmae)
        result[f"{group}_ficr"] = float(ficr)
    if not group_ficrs:
        raise ValueError("No scored groups in outer validation fold")
    mean_nmae = float(np.mean(group_nmaes))
    mean_ficr = float(np.mean(group_ficrs))
    result.update(
        {
            "mean_nmae": mean_nmae,
            "mean_ficr": mean_ficr,
            "mean_score": 0.5 * (1.0 - mean_nmae) + 0.5 * mean_ficr,
            "n_checkpoint_groups": int(len(group_ficrs)),
        }
    )
    return result


def train_tcn2_fold_best(
    wind_features: np.ndarray,
    official_kwh: np.ndarray,
    years: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    val_year: int,
    turbine_order: tuple[str, ...],
    args,
    device: torch.device,
    seed: int,
    checkpoint_dir: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    """Use the outer validation fold itself for checkpoint selection; no refit."""
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

    scaler = SequenceStandardScaler()
    scaler.fit(wind_features[train_indices])
    features = scaler.transform(wind_features)
    mean_targets, valid_fractions = proto.split_target_statistics(
        target, scored, train_indices
    )
    loader = proto.make_joint_loader(
        features,
        target,
        scored,
        train_indices,
        args.power_batch_size,
        device,
        seed,
    )
    base.set_seed(seed)
    model = proto.new_joint_model(args, device, mean_targets, output_floor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.power_lr, weight_decay=args.power_weight_decay
    )

    best_epoch = 0
    best_ficr = -np.inf
    best_metrics: dict[str, object] = {}
    best_temperature = np.nan
    best_train_loss = np.nan
    best_state: dict[str, torch.Tensor] | None = None
    bad_epochs = 0
    epochs_trained = 0
    for epoch in range(1, args.power_epochs + 1):
        temperature = proto.band_temperature(epoch, args.power_epochs)
        train_loss = proto.train_joint_epoch(
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
        val_prediction = proto.predict_joint(
            model,
            features[val_indices],
            args.eval_batch_size,
            output_floor,
            device,
        )
        metrics = available_group_hard_metrics(
            val_prediction, target, scored, val_indices
        )
        epochs_trained = epoch
        mean_ficr = float(metrics["mean_ficr"])
        if mean_ficr > best_ficr:
            best_epoch = epoch
            best_ficr = mean_ficr
            best_metrics = metrics
            best_temperature = temperature
            best_train_loss = train_loss
            best_state = cpu_state_dict(model)
            bad_epochs = 0
        elif epoch >= proto.BAND_MIN_EPOCHS:
            bad_epochs += 1
        if (
            epoch >= proto.BAND_MIN_EPOCHS
            and bad_epochs >= args.power_patience
        ):
            break
    if best_state is None or best_epoch <= 0:
        raise RuntimeError(f"TCN2 fold-best selection failed: val{val_year}")

    model.load_state_dict(best_state)
    prediction = proto.predict_joint(
        model,
        features[val_indices],
        args.eval_batch_size,
        output_floor,
        device,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"stage2_val{val_year}.pt"
    torch.save(
        {
            "runner_version": RUNNER_VERSION,
            "stage": "TCN2",
            "validation_year": int(val_year),
            "best_epoch": int(best_epoch),
            "model_state_dict": best_state,
            "scaler_mean": scaler.mean_,
            "scaler_std": scaler.std_,
            "turbine_order": turbine_order,
            "target_groups": tuple(TARGET_COLS),
            "mean_targets": mean_targets,
            "output_floor": output_floor,
        },
        checkpoint_path,
    )
    stats: dict[str, object] = {
        "stage": "joint_group_power",
        "role": "stage2_outer_fold_best_checkpoint",
        "protocol": "non_nested_outer_validation_checkpoint_no_refit",
        "architecture": "IssueBlockTCN_17_predicted_winds_to_3_groups",
        "loss": "equal_group_pure_band_ficr",
        "checkpoint_metric": "outer_validation_available_group_mean_hard_ficr",
        "pred_year": int(val_year),
        "best_epoch": int(best_epoch),
        "epochs_trained": int(epochs_trained),
        "best_epoch_temperature": float(best_temperature),
        "best_epoch_train_loss": float(best_train_loss),
        "n_train_issues": int(len(train_indices)),
        "n_val_issues": int(len(val_indices)),
        "n_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "checkpoint_path": str(checkpoint_path),
        "output_transform": f"{output_floor:.2f}+(1-{output_floor:.2f})*sigmoid(raw)",
    }
    for key, value in best_metrics.items():
        stats[f"outer_val_{key}"] = value
    for group_index, group in enumerate(TARGET_COLS):
        stats[f"{group}_mean_target_train"] = float(mean_targets[group_index])
        stats[f"{group}_valid_fraction_train"] = float(
            valid_fractions[group_index]
        )
    del model, optimizer, loader
    base.release_cuda()
    return prediction, stats


def main() -> None:
    args = base.parse_args()
    groups = base.parse_csv(args.groups)
    requested_years = [int(value) for value in base.parse_csv(args.years)]
    if groups != TARGET_COLS:
        raise ValueError("Canonical three-group order is required")
    if requested_years != [2022, 2023, 2024]:
        raise ValueError("This fold-best experiment requires years 2022,2023,2024")
    if args.power_epochs < proto.BAND_MIN_EPOCHS:
        raise ValueError(
            f"--power-epochs must be at least {proto.BAND_MIN_EPOCHS}"
        )
    args.results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.results_dir / f"{args.stem}_checkpoints"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} protocol=NON_NESTED_FOLD_BEST_NO_REFIT "
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
    raw_scada = {
        "kpx_group_1": scada_vestas,
        "kpx_group_2": scada_vestas,
        "kpx_group_3": scada_unison,
    }
    scada_hourly = {
        group: build_turbine_scada_hourly(raw_scada[group], group)
        for group in TARGET_COLS
    }
    candidates = build_wind_candidate_matrix(ldaps, gfs)
    print(
        f"candidate_rows={len(candidates.keys)} candidates={len(candidates.names)}",
        flush=True,
    )

    canonical_reference: base.TurbinePanel | None = None
    references: dict[str, base.TurbinePanel] = {}
    stage1_wind: dict[str, np.ndarray] = {}
    stage1_optgrid: dict[str, np.ndarray] = {}
    stage1_scale: dict[str, np.ndarray] = {}
    training_rows: list[dict[str, object]] = []
    selection_parts: list[pd.DataFrame] = []
    turbine_order: list[str] = []

    print("\n=== TCN1 non-nested fold-best OOF ===", flush=True)
    for group_index, group in enumerate(TARGET_COLS):
        base_features = get_or_build_group_feature_cache(
            ldaps,
            gfs,
            group,
            cache_root=args.cache_root,
            rebuild=args.rebuild_feature_cache,
        )
        (
            reference,
            oof_wind,
            oof_optgrid,
            scale_by_issue,
            group_training,
            group_selections,
        ) = build_stage1_group_oof(
            base_features,
            candidates,
            scada_hourly[group],
            labels,
            group,
            group_index,
            requested_years,
            args,
            device,
            checkpoint_dir,
        )
        if canonical_reference is None:
            canonical_reference = reference
        else:
            proto.assert_time_alignment(canonical_reference, reference, group)
        references[group] = reference
        stage1_wind[group] = oof_wind
        stage1_optgrid[group] = oof_optgrid
        stage1_scale[group] = scale_by_issue
        turbine_order.extend(reference.turbines)
        training_rows.extend(group_training)
        selection_parts.extend(group_selections)
        del base_features
        base.release_cuda()

    if canonical_reference is None:
        raise RuntimeError("No canonical panel")
    requested_indices = np.flatnonzero(
        np.isin(canonical_reference.years, requested_years)
    )
    wind17 = np.concatenate(
        [stage1_wind[group] for group in TARGET_COLS], axis=1
    )
    if wind17.shape[1:] != (17, 24):
        raise ValueError(f"Expected [issue,17,24], got {wind17.shape}")
    wind_features = wind17.transpose(0, 2, 1).astype(np.float32)
    official_kwh = np.stack(
        [references[group].official for group in TARGET_COLS], axis=-1
    ).astype(np.float32)

    variant = "global_17wind_to_3group_foldbest_pure_band_tcn"
    group_prediction_parts: list[pd.DataFrame] = []
    print("\n=== TCN2 non-nested fold-best OOF ===", flush=True)
    for pred_year in requested_years:
        train_indices = requested_indices[
            canonical_reference.years[requested_indices] != pred_year
        ]
        val_indices = requested_indices[
            canonical_reference.years[requested_indices] == pred_year
        ]
        prediction_norm, stats = train_tcn2_fold_best(
            wind_features,
            official_kwh,
            canonical_reference.years,
            train_indices,
            val_indices,
            pred_year,
            tuple(turbine_order),
            args,
            device,
            args.seed + pred_year * 1000 + 777,
            checkpoint_dir,
        )
        training_rows.append(stats)
        print(
            f"  TCN2 val{pred_year}: best_epoch={stats['best_epoch']} "
            f"FiCR={stats['outer_val_mean_ficr']:.6f} "
            f"groups={stats['outer_val_n_checkpoint_groups']} no_refit",
            flush=True,
        )
        for group_index, group in enumerate(TARGET_COLS):
            capacity = float(GROUP_CAPACITY_KWH[group])
            part = pd.DataFrame(
                {
                    "forecast_kst_dtm": canonical_reference.forecast_times[
                        val_indices
                    ].reshape(-1),
                    "issue_kst_dtm": np.repeat(
                        canonical_reference.issue_times[val_indices], 24
                    ),
                    "group": group,
                    "pred_year": pred_year,
                    "official_target": official_kwh[
                        val_indices, :, group_index
                    ].reshape(-1),
                    "pred": (
                        prediction_norm[:, :, group_index] * capacity
                    ).reshape(-1),
                    "variant": variant,
                }
            ).dropna(subset=["official_target"])
            group_prediction_parts.append(part)

    turbine_parts: list[pd.DataFrame] = []
    for group in TARGET_COLS:
        reference = references[group]
        for turbine_index, turbine in enumerate(reference.turbines):
            turbine_parts.append(
                pd.DataFrame(
                    {
                        "forecast_kst_dtm": reference.forecast_times[
                            requested_indices
                        ].reshape(-1),
                        "issue_kst_dtm": np.repeat(
                            reference.issue_times[requested_indices], 24
                        ),
                        "group": group,
                        "turbine_id": turbine,
                        "pred_year": np.repeat(
                            reference.years[requested_indices], 24
                        ),
                        "actual_scada_ws_cubic": reference.wind[
                            requested_indices, turbine_index
                        ].reshape(-1),
                        "optgrid_ws_calibrated": stage1_optgrid[group][
                            requested_indices, turbine_index
                        ].reshape(-1),
                        "predicted_scada_ws": stage1_wind[group][
                            requested_indices, turbine_index
                        ].reshape(-1),
                        "wind_scale_p99": np.repeat(
                            stage1_scale[group][requested_indices], 24
                        ),
                    }
                )
            )

    predictions = pd.concat(group_prediction_parts, ignore_index=True)
    turbine_predictions = pd.concat(turbine_parts, ignore_index=True)
    folds = base.fold_score_rows(predictions)
    wind_scores = base.wind_score_rows(turbine_predictions)
    summary, pooled = pooled_oof_summary(predictions)
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

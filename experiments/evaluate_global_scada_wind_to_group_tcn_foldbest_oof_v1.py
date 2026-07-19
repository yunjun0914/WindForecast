from __future__ import annotations

import copy
import os

import numpy as np
import pandas as pd

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from lightgbm import LGBMRegressor
from scipy.stats import kendalltau, spearmanr
from sklearn.isotonic import IsotonicRegression

import _bootstrap  # noqa: F401
from experiments import evaluate_global_scada_wind_to_group_tcn_oof_v1 as proto
from utils.metrics import GROUP_CAPACITY_KWH, TARGET_COLS, group_nmae_ficr
from utils.per_turbine_features import (
    LOCAL_TREE_FEATURES,
    get_or_build_group_feature_cache,
)
from utils.per_turbine_optimal_grid import (
    WAKE_FEATURES,
    optimal_grid_input_columns,
)
from utils.per_turbine_optimal_grid_builder import build_wind_candidate_matrix
from utils.per_turbine_scada import build_turbine_scada_hourly
from utils.per_turbine_sequence import SequenceStandardScaler


base = proto.base
base.GRID_CACHE_VERSION = "two_stage_scada_cubic_grid_v2_cubic_mae"

RESIDUAL_LOCAL_FEATURES = tuple(
    feature for feature in LOCAL_TREE_FEATURES if feature not in WAKE_FEATURES
)
RAW_TCN2_VARIANT = "tcn2_raw_tcn1"
RESIDUAL_TCN2_VARIANT = "tcn2_residual_corrected_tcn1"
ISOTONIC_TCN2_VARIANT = "tcn2_residual_corrected_tcn1_isotonic"
ISOTONIC_ALPHA_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)


def raw_wind_anchor(
    panel: base.TurbinePanel,
    issue_indices: np.ndarray,
    base_index: int,
) -> np.ndarray:
    values = panel.features[issue_indices, :, :, base_index]
    return values.reshape(len(issue_indices) * len(panel.turbines), 24).astype(
        np.float32
    )


def median_epoch(values: list[int], stage: str) -> int:
    if not values:
        raise ValueError(f"No best epochs were collected for {stage}")
    return max(1, int(np.rint(np.median(np.asarray(values, dtype=float)))))


def observed_issue_mask(panel: base.TurbinePanel) -> np.ndarray:
    return np.isfinite(panel.wind).any(axis=(1, 2))


def stage1_fold_indices(
    reference: base.TurbinePanel,
    requested_years: list[int],
    val_year: int,
) -> tuple[np.ndarray, np.ndarray]:
    requested = np.flatnonzero(np.isin(reference.years, requested_years))
    observed = observed_issue_mask(reference)
    val_indices = requested[(reference.years[requested] == val_year) & observed[requested]]
    train_indices = requested[(reference.years[requested] != val_year) & observed[requested]]
    train_indices = base.remove_forecast_overlap(
        train_indices, val_indices, reference.forecast_times
    )
    if len(train_indices) < 20 or len(val_indices) < 20:
        raise ValueError(
            f"Insufficient TCN1 fold val{val_year}: "
            f"train={len(train_indices)} val={len(val_indices)}"
        )
    return train_indices, val_indices


def discover_stage1_epoch(
    panel: base.TurbinePanel,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    group: str,
    val_year: int,
    args,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    base_index = panel.feature_cols.index("optgrid_ws_calibrated")
    valid_train = panel.wind[train_indices]
    valid_train = valid_train[np.isfinite(valid_train)]
    if len(valid_train) < 500:
        raise ValueError(f"Too few TCN1 wind targets for {group}/val{val_year}")
    scale = max(float(np.quantile(valid_train, 0.99)), 5.0)

    scaler = SequenceStandardScaler().fit(panel.features[train_indices])
    scaled_panel = copy.copy(panel)
    scaled_panel.features = scaler.transform(panel.features)
    x_train, y_train, _, m_train = base.flatten_wind_data(
        scaled_panel, train_indices, base_index
    )
    x_val, _, _, _ = base.flatten_wind_data(
        scaled_panel, val_indices, base_index
    )
    b_train = raw_wind_anchor(panel, train_indices, base_index)
    b_val = raw_wind_anchor(panel, val_indices, base_index)
    loader = base.wind_loader(
        x_train, y_train, b_train, m_train, args.batch_size, device
    )

    base.set_seed(seed)
    model = base.new_wind_model(panel.features.shape[-1], args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.wind_lr, weight_decay=args.wind_weight_decay
    )
    best_epoch = 0
    best_cubic_mae = np.inf
    best_wind_mae = np.nan
    bad_epochs = 0
    epochs_trained = 0
    epoch_curve: list[dict[str, float | int]] = []
    for epoch in range(1, args.wind_epochs + 1):
        base.train_wind_epochs(model, loader, optimizer, 1, scale, args, device)
        prediction = base.predict_wind_flat(
            model, x_val, b_val, args.eval_batch_size, device
        )
        actual = panel.wind[val_indices].reshape(prediction.shape)
        valid = np.isfinite(actual) & np.isfinite(prediction)
        cubic_mae = float(
            np.mean(
                np.abs(
                    (prediction[valid] / scale) ** 3
                    - (actual[valid] / scale) ** 3
                )
            )
        )
        wind_mae = float(np.mean(np.abs(prediction[valid] - actual[valid])))
        epoch_curve.append(
            {
                "epoch": int(epoch),
                "wind_cubic_mae": cubic_mae,
                "wind_mae": wind_mae,
            }
        )
        epochs_trained = epoch
        if cubic_mae < best_cubic_mae - args.wind_min_delta:
            best_epoch = epoch
            best_cubic_mae = cubic_mae
            best_wind_mae = wind_mae
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.wind_patience:
            break
    if best_epoch <= 0:
        raise RuntimeError(f"TCN1 epoch discovery failed: {group}/val{val_year}")

    stats = {
        "stage": "TCN1_epoch_discovery",
        "group": group,
        "pred_year": int(val_year),
        "best_epoch": int(best_epoch),
        "epochs_trained": int(epochs_trained),
        "val_cubic_mae": float(best_cubic_mae),
        "val_wind_mae": float(best_wind_mae),
        "wind_scale_p99": float(scale),
        "epoch_curve": epoch_curve,
    }
    del model, optimizer, loader, scaled_panel
    base.release_cuda()
    return stats


def train_stage1_fixed(
    panel: base.TurbinePanel,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    fixed_epoch: int,
    group: str,
    pred_year: int,
    args,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    base_index = panel.feature_cols.index("optgrid_ws_calibrated")
    valid_train = panel.wind[train_indices]
    valid_train = valid_train[np.isfinite(valid_train)]
    if len(valid_train) < 500:
        raise ValueError(f"Too few fixed TCN1 targets for {group}/pred{pred_year}")
    scale = max(float(np.quantile(valid_train, 0.99)), 5.0)

    scaler = SequenceStandardScaler().fit(panel.features[train_indices])
    scaled_panel = copy.copy(panel)
    scaled_panel.features = scaler.transform(panel.features)
    x_train, y_train, _, m_train = base.flatten_wind_data(
        scaled_panel, train_indices, base_index
    )
    x_predict, _, _, _ = base.flatten_wind_data(
        scaled_panel, predict_indices, base_index
    )
    b_train = raw_wind_anchor(panel, train_indices, base_index)
    b_predict = raw_wind_anchor(panel, predict_indices, base_index)
    loader = base.wind_loader(
        x_train, y_train, b_train, m_train, args.batch_size, device
    )

    base.set_seed(seed)
    model = base.new_wind_model(panel.features.shape[-1], args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.wind_lr, weight_decay=args.wind_weight_decay
    )
    base.train_wind_epochs(
        model, loader, optimizer, fixed_epoch, scale, args, device
    )
    flat_prediction = base.predict_wind_flat(
        model, x_predict, b_predict, args.eval_batch_size, device
    )
    prediction = flat_prediction.reshape(
        len(predict_indices), len(panel.turbines), 24
    ).astype(np.float32)

    actual = panel.wind[predict_indices]
    valid = np.isfinite(actual)
    wind_mae = float(np.mean(np.abs(prediction[valid] - actual[valid]))) if valid.any() else np.nan
    cubic_mae = (
        float(
            np.mean(
                np.abs(
                    (prediction[valid] / scale) ** 3
                    - (actual[valid] / scale) ** 3
                )
            )
        )
        if valid.any()
        else np.nan
    )
    stats = {
        "stage": "TCN1_fixed_epoch",
        "group": group,
        "pred_year": int(pred_year),
        "fixed_epoch": int(fixed_epoch),
        "wind_mae": wind_mae,
        "wind_cubic_mae": cubic_mae,
        "wind_scale_p99": float(scale),
        "n_train_issues": int(len(train_indices)),
        "n_predict_issues": int(len(predict_indices)),
        "n_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
    }
    del model, optimizer, loader, scaled_panel
    base.release_cuda()
    return prediction, stats


def reference_panel(
    base_features: pd.DataFrame,
    scada_hourly: pd.DataFrame,
    labels: pd.DataFrame,
    group: str,
) -> base.TurbinePanel:
    return base.build_panel(
        base_features,
        scada_hourly,
        labels,
        group,
        [optimal_grid_input_columns(group)[0]],
    )


def discover_stage1_group_epochs(
    base_features: pd.DataFrame,
    candidates,
    scada_hourly: pd.DataFrame,
    labels: pd.DataFrame,
    group: str,
    group_index: int,
    requested_years: list[int],
    reference: base.TurbinePanel,
    args,
    device: torch.device,
) -> list[dict[str, object]]:
    observed = observed_issue_mask(reference)
    validation_years = [
        year
        for year in requested_years
        if observed[reference.years == year].any()
    ]
    rows: list[dict[str, object]] = []
    for val_year in validation_years:
        train_indices, val_indices = stage1_fold_indices(
            reference, requested_years, val_year
        )
        panel, _ = base.build_cubic_feature_panel(
            base_features,
            candidates,
            scada_hourly,
            labels,
            group,
            reference.forecast_times[train_indices],
            f"fixed_epoch_val{val_year}_cubic_mae",
            args,
        )
        base.assert_panel_alignment(reference, panel)
        stats = discover_stage1_epoch(
            panel,
            train_indices,
            val_indices,
            group,
            val_year,
            args,
            device,
            args.seed + group_index * 100000 + val_year * 100,
        )
        rows.append(stats)
        print(
            f"  TCN1 discovery {group} val{val_year}: "
            f"best_epoch={stats['best_epoch']} "
            f"cubic_MAE={stats['val_cubic_mae']:.6f}",
            flush=True,
        )
    return rows


def build_stage1_group_fixed_oof(
    base_features: pd.DataFrame,
    candidates,
    scada_hourly: pd.DataFrame,
    labels: pd.DataFrame,
    group: str,
    group_index: int,
    requested_years: list[int],
    reference: base.TurbinePanel,
    fixed_epoch: int,
    args,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    requested_indices = np.flatnonzero(np.isin(reference.years, requested_years))
    observed = observed_issue_mask(reference)
    validation_years = [
        year
        for year in requested_years
        if observed[reference.years == year].any()
    ]
    oof_wind = np.full(reference.wind.shape, np.nan, dtype=np.float32)
    oof_optgrid = np.full(reference.wind.shape, np.nan, dtype=np.float32)

    for val_year in validation_years:
        train_indices, val_indices = stage1_fold_indices(
            reference, requested_years, val_year
        )
        panel, _ = base.build_cubic_feature_panel(
            base_features,
            candidates,
            scada_hourly,
            labels,
            group,
            reference.forecast_times[train_indices],
            f"fixed_epoch_val{val_year}_cubic_mae",
            args,
        )
        base.assert_panel_alignment(reference, panel)
        prediction, stats = train_stage1_fixed(
            panel,
            train_indices,
            val_indices,
            fixed_epoch,
            group,
            val_year,
            args,
            device,
            args.seed + group_index * 100000 + val_year * 100,
        )
        base_index = panel.feature_cols.index("optgrid_ws_calibrated")
        oof_wind[val_indices] = prediction
        oof_optgrid[val_indices] = panel.features[val_indices, :, :, base_index]
        print(
            f"  TCN1 fixed {group} val{val_year}: epoch={fixed_epoch} "
            f"wind_MAE={stats['wind_mae']:.4f}",
            flush=True,
        )

    missing_indices = requested_indices[~observed[requested_indices]]
    if len(missing_indices):
        if group != "kpx_group_3":
            raise ValueError(f"Unexpected missing SCADA-wind issues for {group}")
        train_indices = requested_indices[observed[requested_indices]]
        panel, _ = base.build_cubic_feature_panel(
            base_features,
            candidates,
            scada_hourly,
            labels,
            group,
            reference.forecast_times[train_indices],
            "fixed_epoch_all_observed_for_missing_cubic_mae",
            args,
        )
        base.assert_panel_alignment(reference, panel)
        prediction, stats = train_stage1_fixed(
            panel,
            train_indices,
            missing_indices,
            fixed_epoch,
            group,
            int(reference.years[missing_indices][0]),
            args,
            device,
            args.seed + group_index * 100000 + 2022 * 100 + 77,
        )
        base_index = panel.feature_cols.index("optgrid_ws_calibrated")
        oof_wind[missing_indices] = prediction
        oof_optgrid[missing_indices] = panel.features[
            missing_indices, :, :, base_index
        ]
        print(
            f"  TCN1 fixed {group} 2022: trained on all observed 2023-2024, "
            f"epoch={fixed_epoch}",
            flush=True,
        )

    if not np.isfinite(oof_wind[requested_indices]).all():
        raise ValueError(f"Incomplete fixed-epoch TCN1 OOF for {group}")
    return oof_wind, oof_optgrid


def residual_model_parameters(seed: int, rounds: int, n_jobs: int) -> dict[str, object]:
    return {
        "objective": "regression_l1",
        "learning_rate": 0.03,
        "n_estimators": int(rounds),
        "num_leaves": 15,
        "max_depth": 4,
        "min_child_samples": 250,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_alpha": 2.0,
        "reg_lambda": 20.0,
        "random_state": int(seed),
        "n_jobs": int(n_jobs),
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }


def residual_features(
    local_features: np.ndarray,
    tcn1_prediction: np.ndarray,
    scale: float,
) -> np.ndarray:
    normalized = tcn1_prediction.reshape(-1, 1) / float(scale)
    return np.concatenate(
        [
            local_features.reshape(-1, local_features.shape[-1]),
            normalized,
            normalized**3,
        ],
        axis=1,
    ).astype(np.float32)


def fit_turbine_residual_oof(
    reference: base.TurbinePanel,
    local_features: np.ndarray,
    tcn1_wind: np.ndarray,
    group: str,
    group_index: int,
    requested_years: list[int],
    rounds: int,
    n_jobs: int,
    seed: int,
) -> np.ndarray:
    corrected = tcn1_wind.copy()
    requested = np.flatnonzero(np.isin(reference.years, requested_years))
    issue_numbers = np.arange(len(reference.years))

    for val_year in requested_years:
        train_indices = requested[reference.years[requested] != val_year]
        val_indices = requested[reference.years[requested] == val_year]
        observed_train = reference.wind[train_indices]
        observed_train = observed_train[np.isfinite(observed_train)]
        if len(observed_train) < 500:
            raise ValueError(f"Too few residual targets for {group}/val{val_year}")
        scale = max(float(np.quantile(observed_train, 0.99)), 5.0)
        train_issue_rows = np.repeat(np.isin(issue_numbers, train_indices), 24)
        val_issue_rows = np.repeat(np.isin(issue_numbers, val_indices), 24)

        for turbine_index, turbine in enumerate(reference.turbines):
            actual = reference.wind[:, turbine_index].reshape(-1).astype(float)
            base_prediction = tcn1_wind[:, turbine_index].reshape(-1).astype(float)
            features = residual_features(
                local_features[:, turbine_index],
                tcn1_wind[:, turbine_index],
                scale,
            )
            train_rows = train_issue_rows & np.isfinite(actual) & np.isfinite(base_prediction)
            if int(train_rows.sum()) < 1000:
                raise ValueError(
                    f"Too few residual rows for {group}/{turbine}/val{val_year}"
                )
            target = (actual[train_rows] / scale) ** 3 - (
                base_prediction[train_rows] / scale
            ) ** 3
            model = LGBMRegressor(
                **residual_model_parameters(
                    seed + group_index * 10000 + turbine_index * 100 + val_year,
                    rounds,
                    n_jobs,
                )
            )
            model.fit(features[train_rows], target)
            correction = model.predict(features[val_issue_rows])
            base_val = base_prediction[val_issue_rows]
            corrected_cube = np.clip(
                (base_val / scale) ** 3 + correction,
                0.0,
                (40.0 / scale) ** 3,
            )
            corrected_val = scale * np.cbrt(corrected_cube)
            corrected[val_indices, turbine_index] = corrected_val.reshape(
                len(val_indices), 24
            ).astype(np.float32)
    return np.clip(corrected, 0.0, 40.0).astype(np.float32)


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
        raise ValueError("No scored groups in TCN2 validation fold")
    mean_nmae = float(np.mean(group_nmaes))
    mean_ficr = float(np.mean(group_ficrs))
    result.update(
        {
            "mean_nmae": mean_nmae,
            "mean_ficr": mean_ficr,
            "mean_score": 0.5 * (1.0 - mean_nmae) + 0.5 * mean_ficr,
            "n_scored_groups": int(len(group_ficrs)),
        }
    )
    return result


def tcn2_training_data(
    wind_features: np.ndarray,
    official_kwh: np.ndarray,
    train_indices: np.ndarray,
    args,
    device: torch.device,
    seed: int,
):
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
    scaler = SequenceStandardScaler().fit(wind_features[train_indices])
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
    return features, target, scored, mean_targets, valid_fractions, loader


def discover_tcn2_epoch(
    wind_features: np.ndarray,
    official_kwh: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    val_year: int,
    args,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    (
        features,
        target,
        scored,
        mean_targets,
        valid_fractions,
        loader,
    ) = tcn2_training_data(
        wind_features, official_kwh, train_indices, args, device, seed
    )
    output_floor = float(args.target_min_output_ratio)
    base.set_seed(seed)
    model = proto.new_joint_model(args, device, mean_targets, output_floor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.power_lr, weight_decay=args.power_weight_decay
    )

    best_epoch = 0
    best_ficr = -np.inf
    best_metrics: dict[str, object] = {}
    bad_epochs = 0
    epochs_trained = 0
    for epoch in range(1, args.power_epochs + 1):
        temperature = proto.band_temperature(epoch, args.power_epochs)
        proto.train_joint_epoch(
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
        if float(metrics["mean_ficr"]) > best_ficr:
            best_epoch = epoch
            best_ficr = float(metrics["mean_ficr"])
            best_metrics = metrics
            bad_epochs = 0
        elif epoch >= proto.BAND_MIN_EPOCHS:
            bad_epochs += 1
        if epoch >= proto.BAND_MIN_EPOCHS and bad_epochs >= args.power_patience:
            break
    if best_epoch <= 0:
        raise RuntimeError(f"TCN2 epoch discovery failed: val{val_year}")
    stats = {
        "stage": "TCN2_epoch_discovery",
        "pred_year": int(val_year),
        "best_epoch": int(best_epoch),
        "epochs_trained": int(epochs_trained),
        **best_metrics,
    }
    del model, optimizer, loader
    base.release_cuda()
    return stats


def train_tcn2_fixed(
    wind_features: np.ndarray,
    official_kwh: np.ndarray,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    val_year: int,
    fixed_epoch: int,
    args,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    (
        features,
        target,
        scored,
        mean_targets,
        valid_fractions,
        loader,
    ) = tcn2_training_data(
        wind_features, official_kwh, train_indices, args, device, seed
    )
    output_floor = float(args.target_min_output_ratio)
    base.set_seed(seed)
    model = proto.new_joint_model(args, device, mean_targets, output_floor)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.power_lr, weight_decay=args.power_weight_decay
    )
    for epoch in range(1, fixed_epoch + 1):
        proto.train_joint_epoch(
            model,
            loader,
            optimizer,
            mean_targets,
            valid_fractions,
            proto.band_temperature(epoch, args.power_epochs),
            output_floor,
            args,
            device,
        )
    prediction = proto.predict_joint(
        model,
        features[val_indices],
        args.eval_batch_size,
        output_floor,
        device,
    )
    metrics = available_group_hard_metrics(prediction, target, scored, val_indices)
    stats = {
        "stage": "TCN2_fixed_epoch",
        "pred_year": int(val_year),
        "fixed_epoch": int(fixed_epoch),
        "n_train_issues": int(len(train_indices)),
        "n_val_issues": int(len(val_indices)),
        "n_parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        **metrics,
    }
    del model, optimizer, loader
    base.release_cuda()
    return prediction, stats


def prediction_frame(
    reference: base.TurbinePanel,
    official_kwh: np.ndarray,
    val_indices: np.ndarray,
    pred_year: int,
    prediction_norm: np.ndarray,
    variant: str,
) -> pd.DataFrame:
    parts = []
    for group_index, group in enumerate(TARGET_COLS):
        capacity = float(GROUP_CAPACITY_KWH[group])
        part = pd.DataFrame(
            {
                "forecast_kst_dtm": reference.forecast_times[val_indices].reshape(-1),
                "issue_kst_dtm": np.repeat(reference.issue_times[val_indices], 24),
                "group": group,
                "pred_year": int(pred_year),
                "official_target": official_kwh[
                    val_indices, :, group_index
                ].reshape(-1),
                "pred": (
                    prediction_norm[:, :, group_index] * capacity
                ).reshape(-1),
                "variant": variant,
            }
        ).dropna(subset=["official_target"])
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def fit_isotonic(frame: pd.DataFrame, capacity: float) -> IsotonicRegression:
    valid = np.isfinite(frame["pred"]) & np.isfinite(frame["official_target"])
    if int(valid.sum()) < 1000:
        raise ValueError(f"Too few isotonic rows: {valid.sum()}")
    calibrator = IsotonicRegression(
        increasing=True,
        y_min=0.0,
        y_max=capacity,
        out_of_bounds="clip",
    )
    calibrator.fit(
        frame.loc[valid, "pred"].to_numpy(float),
        frame.loc[valid, "official_target"].to_numpy(float),
    )
    return calibrator


def isotonic_inner_folds(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    years = sorted(frame["pred_year"].unique())
    if len(years) >= 2:
        return [
            (
                np.flatnonzero(frame["pred_year"].ne(year).to_numpy()),
                np.flatnonzero(frame["pred_year"].eq(year).to_numpy()),
            )
            for year in years
        ]
    issues = np.sort(frame["issue_kst_dtm"].unique())
    issue_folds = [chunk for chunk in np.array_split(issues, 3) if len(chunk)]
    folds = []
    for heldout in issue_folds:
        val_mask = frame["issue_kst_dtm"].isin(heldout).to_numpy()
        folds.append((np.flatnonzero(~val_mask), np.flatnonzero(val_mask)))
    return folds


def apply_isotonic_outer_oof(
    predictions: pd.DataFrame,
    source_variant: str,
) -> pd.DataFrame:
    source = predictions.loc[predictions["variant"].eq(source_variant)].copy()
    parts = []
    for group in TARGET_COLS:
        group_frame = source.loc[source["group"].eq(group)]
        capacity = float(GROUP_CAPACITY_KWH[group])
        for val_year in sorted(group_frame["pred_year"].unique()):
            train = group_frame.loc[group_frame["pred_year"].ne(val_year)].reset_index(
                drop=True
            )
            val = group_frame.loc[group_frame["pred_year"].eq(val_year)].copy()
            crossfit_isotonic = np.full(len(train), np.nan, dtype=float)
            for fit_indices, heldout_indices in isotonic_inner_folds(train):
                calibrator = fit_isotonic(train.iloc[fit_indices], capacity)
                crossfit_isotonic[heldout_indices] = calibrator.predict(
                    train.iloc[heldout_indices]["pred"].to_numpy(float)
                )
            if not np.isfinite(crossfit_isotonic).all():
                raise ValueError(f"Incomplete isotonic inner OOF: {group}/val{val_year}")

            actual = train["official_target"].to_numpy(float)
            raw = train["pred"].to_numpy(float)
            alpha_scores = {}
            for alpha in ISOTONIC_ALPHA_GRID:
                blended = raw + alpha * (crossfit_isotonic - raw)
                nmae, ficr = group_nmae_ficr(actual, blended, capacity)
                alpha_scores[alpha] = 0.5 * (1.0 - nmae) + 0.5 * ficr
            selected_alpha = max(
                ISOTONIC_ALPHA_GRID,
                key=lambda alpha: (alpha_scores[alpha], -alpha),
            )

            final_calibrator = fit_isotonic(train, capacity)
            raw_val = val["pred"].to_numpy(float)
            isotonic_val = final_calibrator.predict(raw_val)
            val["pred"] = np.clip(
                raw_val + selected_alpha * (isotonic_val - raw_val),
                0.0,
                capacity,
            )
            val["variant"] = ISOTONIC_TCN2_VARIANT
            val["isotonic_alpha"] = float(selected_alpha)
            parts.append(val)
    if not parts:
        raise ValueError("No isotonic OOF predictions were generated")
    return pd.concat(parts, ignore_index=True)


def score_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (variant, group), part in predictions.groupby(["variant", "group"], sort=False):
        capacity = float(GROUP_CAPACITY_KWH[group])
        actual = part["official_target"].to_numpy(float)
        forecast = part["pred"].to_numpy(float)
        valid = np.isfinite(actual) & np.isfinite(forecast) & (
            actual >= 0.10 * capacity
        )
        actual = actual[valid]
        forecast = forecast[valid]
        error_rate = np.abs(forecast - actual) / capacity
        total_weight = float(actual.sum())
        weighted_6 = float(actual[error_rate <= 0.06].sum() / total_weight)
        weighted_8 = float(actual[error_rate <= 0.08].sum() / total_weight)
        nmae, ficr = group_nmae_ficr(actual, forecast, capacity)
        rows.append(
            {
                "scope": "group",
                "variant": variant,
                "group": group,
                "score": 0.5 * (1.0 - nmae) + 0.5 * ficr,
                "nmae": float(nmae),
                "ficr": float(ficr),
                "weighted_within_6pct": weighted_6,
                "weighted_within_8pct": weighted_8,
                "row_within_6pct": float(np.mean(error_rate <= 0.06)),
                "row_within_8pct": float(np.mean(error_rate <= 0.08)),
                "evaluated_rows": int(len(actual)),
                "mean_isotonic_alpha": (
                    float(part["isotonic_alpha"].mean())
                    if "isotonic_alpha" in part and part["isotonic_alpha"].notna().any()
                    else np.nan
                ),
            }
        )
    group_rows = pd.DataFrame(rows)
    summary_rows = []
    for variant, part in group_rows.groupby("variant", sort=False):
        mean_nmae = float(part["nmae"].mean())
        mean_ficr = float(part["ficr"].mean())
        summary_rows.append(
            {
                "scope": "group_equal_mean",
                "variant": variant,
                "group": "__mean__",
                "score": 0.5 * (1.0 - mean_nmae) + 0.5 * mean_ficr,
                "nmae": mean_nmae,
                "ficr": mean_ficr,
                "weighted_within_6pct": float(part["weighted_within_6pct"].mean()),
                "weighted_within_8pct": float(part["weighted_within_8pct"].mean()),
                "row_within_6pct": float(part["row_within_6pct"].mean()),
                "row_within_8pct": float(part["row_within_8pct"].mean()),
                "evaluated_rows": int(part["evaluated_rows"].sum()),
                "mean_isotonic_alpha": (
                    float(part["mean_isotonic_alpha"].mean())
                    if part["mean_isotonic_alpha"].notna().any()
                    else np.nan
                ),
            }
        )
    return pd.concat([group_rows, pd.DataFrame(summary_rows)], ignore_index=True)


def safe_rank(function, left: np.ndarray, right: np.ndarray) -> float:
    if len(left) < 2 or np.unique(left).size < 2 or np.unique(right).size < 2:
        return np.nan
    return float(function(left, right).statistic)


def rank_diagnostics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for variant, variant_frame in predictions.groupby("variant", sort=False):
        for group, group_frame in variant_frame.groupby("group", sort=True):
            scopes = [("group", np.nan, group_frame)]
            scopes.extend(
                ("group_year", int(year), year_frame)
                for year, year_frame in group_frame.groupby("pred_year", sort=True)
            )
            capacity = float(GROUP_CAPACITY_KWH[group])
            for scope, year, part in scopes:
                valid = (
                    np.isfinite(part["official_target"])
                    & np.isfinite(part["pred"])
                    & part["official_target"].ge(0.10 * capacity)
                )
                actual = part.loc[valid, "official_target"].to_numpy(float)
                forecast = part.loc[valid, "pred"].to_numpy(float)
                rows.append(
                    {
                        "scope": scope,
                        "variant": variant,
                        "group": group,
                        "pred_year": year,
                        "evaluated_rows": int(len(actual)),
                        "spearman_actual": safe_rank(spearmanr, forecast, actual),
                        "kendall_actual": safe_rank(kendalltau, forecast, actual),
                        "unique_predictions": int(np.unique(forecast).size),
                        "source_calibrated_spearman": np.nan,
                        "unique_retention": np.nan,
                        "monotonic_inversion_steps": np.nan,
                    }
                )

    keys = [
        "forecast_kst_dtm",
        "issue_kst_dtm",
        "group",
        "pred_year",
        "official_target",
    ]
    raw = predictions.loc[
        predictions["variant"].eq(RESIDUAL_TCN2_VARIANT), keys + ["pred"]
    ].rename(columns={"pred": "raw_pred"})
    calibrated = predictions.loc[
        predictions["variant"].eq(ISOTONIC_TCN2_VARIANT), keys + ["pred"]
    ].rename(columns={"pred": "calibrated_pred"})
    paired = raw.merge(calibrated, on=keys, how="inner", validate="one_to_one")
    for group, group_frame in paired.groupby("group", sort=True):
        scopes = [("isotonic_mapping_group", np.nan, group_frame)]
        scopes.extend(
            ("isotonic_mapping_group_year", int(year), year_frame)
            for year, year_frame in group_frame.groupby("pred_year", sort=True)
        )
        for scope, year, part in scopes:
            raw_values = part["raw_pred"].to_numpy(float)
            calibrated_values = part["calibrated_pred"].to_numpy(float)
            mapping = (
                pd.DataFrame({"raw": raw_values, "calibrated": calibrated_values})
                .groupby("raw", sort=True)["calibrated"]
                .first()
            )
            raw_unique = int(np.unique(raw_values).size)
            calibrated_unique = int(np.unique(calibrated_values).size)
            rows.append(
                {
                    "scope": scope,
                    "variant": ISOTONIC_TCN2_VARIANT,
                    "group": group,
                    "pred_year": year,
                    "evaluated_rows": int(len(part)),
                    "spearman_actual": np.nan,
                    "kendall_actual": np.nan,
                    "unique_predictions": calibrated_unique,
                    "source_calibrated_spearman": safe_rank(
                        spearmanr, raw_values, calibrated_values
                    ),
                    "unique_retention": (
                        float(calibrated_unique / raw_unique) if raw_unique else np.nan
                    ),
                    "monotonic_inversion_steps": int(
                        np.sum(np.diff(mapping.to_numpy(float)) < -1e-9)
                    ),
                }
            )
    return pd.DataFrame(rows)


def wind_diagnostics(
    references: dict[str, base.TurbinePanel],
    requested_indices: np.ndarray,
    optgrid: dict[str, np.ndarray],
    tcn1: dict[str, np.ndarray],
    corrected: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    models = {
        "optgrid_cubic_mae_anchor": optgrid,
        "tcn1_fixed_epoch": tcn1,
        "tcn1_plus_cubic_residual": corrected,
    }
    for model_name, model_by_group in models.items():
        for group in TARGET_COLS:
            reference = references[group]
            actual_group = reference.wind[requested_indices]
            pred_group = model_by_group[group][requested_indices]
            scopes = [("group", group, np.nan, np.nan, actual_group, pred_group)]
            for year in sorted(np.unique(reference.years[requested_indices])):
                mask = reference.years[requested_indices] == year
                scopes.append(
                    (
                        "group_year",
                        group,
                        int(year),
                        np.nan,
                        actual_group[mask],
                        pred_group[mask],
                    )
                )
            for turbine_index, turbine in enumerate(reference.turbines):
                scopes.append(
                    (
                        "turbine",
                        group,
                        np.nan,
                        turbine,
                        actual_group[:, turbine_index],
                        pred_group[:, turbine_index],
                    )
                )
            for scope, scope_group, year, turbine, actual, prediction in scopes:
                valid = np.isfinite(actual) & np.isfinite(prediction)
                if not valid.any():
                    continue
                scale = max(float(np.quantile(actual[valid], 0.99)), 5.0)
                rows.append(
                    {
                        "scope": scope,
                        "model": model_name,
                        "group": scope_group,
                        "pred_year": year,
                        "turbine_id": turbine,
                        "wind_mae": float(np.mean(np.abs(prediction[valid] - actual[valid]))),
                        "wind_cubic_mae": float(
                            np.mean(
                                np.abs(
                                    (prediction[valid] / scale) ** 3
                                    - (actual[valid] / scale) ** 3
                                )
                            )
                        ),
                        "wind_bias": float(np.mean(prediction[valid] - actual[valid])),
                        "n_wind": int(valid.sum()),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    args = base.parse_args()
    groups = base.parse_csv(args.groups)
    requested_years = [int(value) for value in base.parse_csv(args.years)]
    if groups != TARGET_COLS:
        raise ValueError("Canonical three-group order is required")
    if requested_years != [2022, 2023, 2024]:
        raise ValueError("Fixed-epoch runner requires years 2022,2023,2024")
    if args.power_epochs < proto.BAND_MIN_EPOCHS:
        raise ValueError(f"--power-epochs must be at least {proto.BAND_MIN_EPOCHS}")
    residual_rounds = int(getattr(args, "residual_rounds", 300))
    n_jobs = int(getattr(args, "n_jobs", -1))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"device={device} protocol=SHARED_HPARAM_MEDIAN_FIXED_EPOCH "
        f"TCN1=h{args.wind_hidden_size}/L{args.wind_num_layers} cubic-MAE; "
        f"residual=17xLGBM({residual_rounds}); "
        f"TCN2=h{args.power_hidden_size}/L{args.power_num_layers}",
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

    references: dict[str, base.TurbinePanel] = {}
    stage1_discovery: list[dict[str, object]] = []
    print("\n=== TCN1 epoch discovery with cubic-MAE grid selection ===", flush=True)
    for group_index, group in enumerate(TARGET_COLS):
        base_features = get_or_build_group_feature_cache(
            ldaps,
            gfs,
            group,
            cache_root=args.cache_root,
            rebuild=args.rebuild_feature_cache,
        )
        reference = reference_panel(base_features, scada_hourly[group], labels, group)
        references[group] = reference
        stage1_discovery.extend(
            discover_stage1_group_epochs(
                base_features,
                candidates,
                scada_hourly[group],
                labels,
                group,
                group_index,
                requested_years,
                reference,
                args,
                device,
            )
        )
        del base_features
    tcn1_fixed_epoch = median_epoch(
        [int(row["best_epoch"]) for row in stage1_discovery], "TCN1"
    )
    print(f"TCN1 shared fixed epoch = {tcn1_fixed_epoch}", flush=True)

    canonical_reference = references[TARGET_COLS[0]]
    for group in TARGET_COLS[1:]:
        proto.assert_time_alignment(canonical_reference, references[group], group)
    requested_indices = np.flatnonzero(
        np.isin(canonical_reference.years, requested_years)
    )

    stage1_wind: dict[str, np.ndarray] = {}
    stage1_corrected: dict[str, np.ndarray] = {}
    stage1_optgrid: dict[str, np.ndarray] = {}
    print("\n=== TCN1 fixed-epoch OOF and unused-feature residual ===", flush=True)
    for group_index, group in enumerate(TARGET_COLS):
        base_features = get_or_build_group_feature_cache(
            ldaps,
            gfs,
            group,
            cache_root=args.cache_root,
            rebuild=args.rebuild_feature_cache,
        )
        oof_wind, oof_optgrid = build_stage1_group_fixed_oof(
            base_features,
            candidates,
            scada_hourly[group],
            labels,
            group,
            group_index,
            requested_years,
            references[group],
            tcn1_fixed_epoch,
            args,
            device,
        )
        local_panel = base.build_panel(
            base_features,
            scada_hourly[group],
            labels,
            group,
            list(RESIDUAL_LOCAL_FEATURES),
        )
        base.assert_panel_alignment(references[group], local_panel)
        local_features = local_panel.features[..., : len(RESIDUAL_LOCAL_FEATURES)]
        corrected = fit_turbine_residual_oof(
            references[group],
            local_features,
            oof_wind,
            group,
            group_index,
            requested_years,
            residual_rounds,
            n_jobs,
            args.seed,
        )
        stage1_wind[group] = oof_wind
        stage1_corrected[group] = corrected
        stage1_optgrid[group] = oof_optgrid
        del base_features, local_panel, local_features

    raw_wind17 = np.concatenate(
        [stage1_wind[group] for group in TARGET_COLS], axis=1
    ).transpose(0, 2, 1).astype(np.float32)
    corrected_wind17 = np.concatenate(
        [stage1_corrected[group] for group in TARGET_COLS], axis=1
    ).transpose(0, 2, 1).astype(np.float32)
    if raw_wind17.shape[1:] != (24, 17):
        raise ValueError(f"Expected TCN2 wind shape [issue,24,17], got {raw_wind17.shape}")
    official_kwh = np.stack(
        [references[group].official for group in TARGET_COLS], axis=-1
    ).astype(np.float32)

    tcn2_discovery: list[dict[str, object]] = []
    print("\n=== TCN2 epoch discovery on residual-corrected TCN1 ===", flush=True)
    for pred_year in requested_years:
        train_indices = requested_indices[
            canonical_reference.years[requested_indices] != pred_year
        ]
        val_indices = requested_indices[
            canonical_reference.years[requested_indices] == pred_year
        ]
        stats = discover_tcn2_epoch(
            corrected_wind17,
            official_kwh,
            train_indices,
            val_indices,
            pred_year,
            args,
            device,
            args.seed + pred_year * 1000 + 777,
        )
        tcn2_discovery.append(stats)
        print(
            f"  TCN2 discovery val{pred_year}: best_epoch={stats['best_epoch']} "
            f"FiCR={stats['mean_ficr']:.6f}",
            flush=True,
        )
    tcn2_fixed_epoch = median_epoch(
        [int(row["best_epoch"]) for row in tcn2_discovery], "TCN2"
    )
    print(f"TCN2 shared fixed epoch = {tcn2_fixed_epoch}", flush=True)

    prediction_parts: list[pd.DataFrame] = []
    print("\n=== TCN2 fixed-epoch raw vs residual-corrected OOF ===", flush=True)
    for pred_year in requested_years:
        train_indices = requested_indices[
            canonical_reference.years[requested_indices] != pred_year
        ]
        val_indices = requested_indices[
            canonical_reference.years[requested_indices] == pred_year
        ]
        for variant, features in [
            (RAW_TCN2_VARIANT, raw_wind17),
            (RESIDUAL_TCN2_VARIANT, corrected_wind17),
        ]:
            prediction_norm, stats = train_tcn2_fixed(
                features,
                official_kwh,
                train_indices,
                val_indices,
                pred_year,
                tcn2_fixed_epoch,
                args,
                device,
                args.seed + pred_year * 1000 + 777,
            )
            prediction_parts.append(
                prediction_frame(
                    canonical_reference,
                    official_kwh,
                    val_indices,
                    pred_year,
                    prediction_norm,
                    variant,
                )
            )
            print(
                f"  {variant} val{pred_year}: epoch={tcn2_fixed_epoch} "
                f"score={stats['mean_score']:.6f}",
                flush=True,
            )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions = pd.concat(
        [
            predictions,
            apply_isotonic_outer_oof(predictions, RESIDUAL_TCN2_VARIANT),
        ],
        ignore_index=True,
    )
    scores = score_diagnostics(predictions)
    ranks = rank_diagnostics(predictions)
    wind = wind_diagnostics(
        references,
        requested_indices,
        stage1_optgrid,
        stage1_wind,
        stage1_corrected,
    )
    epoch_curve_rows = []
    for row in stage1_discovery:
        for point in row["epoch_curve"]:
            epoch_curve_rows.append(
                {
                    "scope": "epoch_discovery",
                    "model": "tcn1_validation_curve",
                    "group": row["group"],
                    "pred_year": row["pred_year"],
                    "turbine_id": np.nan,
                    "epoch": point["epoch"],
                    "wind_mae": point["wind_mae"],
                    "wind_cubic_mae": point["wind_cubic_mae"],
                    "wind_bias": np.nan,
                    "n_wind": np.nan,
                }
            )
    wind = pd.concat([wind, pd.DataFrame(epoch_curve_rows)], ignore_index=True)
    scores["tcn1_fixed_epoch"] = tcn1_fixed_epoch
    scores["tcn2_fixed_epoch"] = tcn2_fixed_epoch
    scores["residual_rounds"] = residual_rounds

    prefix = args.results_dir / args.stem
    predictions.to_csv(
        f"{prefix}_predictions.csv", index=False, encoding="utf-8-sig"
    )
    scores.to_csv(f"{prefix}_scores.csv", index=False, encoding="utf-8-sig")
    wind.to_csv(
        f"{prefix}_wind_diagnostics.csv", index=False, encoding="utf-8-sig"
    )
    ranks.to_csv(
        f"{prefix}_rank_diagnostics.csv", index=False, encoding="utf-8-sig"
    )

    print("\n=== fixed-epoch pooled OOF ===", flush=True)
    print(
        scores.loc[scores["scope"].eq("group_equal_mean")].to_string(index=False),
        flush=True,
    )
    print("\n=== wind diagnostics ===", flush=True)
    print(
        wind.loc[wind["scope"].eq("group")].to_string(index=False),
        flush=True,
    )
    print("\n=== rank diagnostics ===", flush=True)
    print(
        ranks.loc[ranks["scope"].eq("group")].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()

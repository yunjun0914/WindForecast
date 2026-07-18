from __future__ import annotations

import copy

import numpy as np
import torch

import _bootstrap  # noqa: F401
from experiments import evaluate_two_stage_scada_cubic_tcn_oof as base
from utils.per_turbine_sequence import SequenceStandardScaler


def wind_cubic_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
    observed: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    difference = torch.pow(prediction / scale, 3) - torch.pow(target / scale, 3)
    return (torch.abs(difference) * observed).sum() / torch.clamp(
        observed.sum(), min=1.0
    )


def wind_metrics_mae(
    actual: np.ndarray, prediction: np.ndarray, scale: float
) -> dict[str, float]:
    valid = np.isfinite(actual) & np.isfinite(prediction)
    actual = actual[valid].astype(float)
    prediction = prediction[valid].astype(float)
    cubic_difference = (prediction / scale) ** 3 - (actual / scale) ** 3
    cubic_mae = float(np.mean(np.abs(cubic_difference)))
    return {
        "wind_mae": float(np.mean(np.abs(prediction - actual))),
        "wind_rmse": float(np.sqrt(np.mean(np.square(prediction - actual)))),
        "wind_cubic_mae": cubic_mae,
        # Compatibility alias for the v1 caller's console formatting only.
        "wind_cubic_mse": cubic_mae,
        "wind_bias": float(np.mean(prediction - actual)),
        "n_wind": int(len(actual)),
    }


def select_and_refit_wind_mae(
    panel: base.TurbinePanel,
    train_indices: np.ndarray,
    predict_indices: np.ndarray,
    args,
    device: torch.device,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    base_index = panel.feature_cols.index("optgrid_ws_calibrated")
    epoch_train, epoch_val = base.make_epoch_split(
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
    x_train, y_train, b_train, m_train = base.flatten_wind_data(
        select_panel, epoch_train, base_index
    )
    x_val, _, b_val, _ = base.flatten_wind_data(
        select_panel, epoch_val, base_index
    )
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
        model.parameters(), lr=args.wind_lr, weight_decay=args.wind_weight_decay
    )
    best_epoch = 0
    best_loss = np.inf
    best_state = None
    bad_epochs = 0
    best_epoch_mae = np.nan
    for epoch in range(1, args.wind_epochs + 1):
        base.train_wind_epochs(model, loader, optimizer, 1, scale, args, device)
        prediction = base.predict_wind_flat(
            model, x_val, b_val, args.eval_batch_size, device
        )
        actual = panel.wind[epoch_val].reshape(prediction.shape)
        valid = np.isfinite(actual)
        cubic_difference = (prediction[valid] / scale) ** 3 - (
            actual[valid] / scale
        ) ** 3
        val_loss = float(np.mean(np.abs(cubic_difference)))
        if val_loss < best_loss - args.wind_min_delta:
            best_epoch = epoch
            best_loss = val_loss
            best_epoch_mae = float(
                np.mean(np.abs(prediction[valid] - actual[valid]))
            )
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.wind_patience:
            break
    if best_state is None or best_epoch <= 0:
        raise RuntimeError("Wind cubic-MAE epoch selection failed")
    del model, optimizer, loader, select_panel
    base.release_cuda()

    final_scaler = SequenceStandardScaler()
    final_scaler.fit(panel.features[train_indices])
    final_panel = copy.copy(panel)
    final_panel.features = final_scaler.transform(panel.features)
    x_train, y_train, b_train, m_train = base.flatten_wind_data(
        final_panel, train_indices, base_index
    )
    x_predict, _, b_predict, _ = base.flatten_wind_data(
        final_panel, predict_indices, base_index
    )
    loader = base.wind_loader(
        x_train,
        y_train,
        b_train,
        m_train,
        args.batch_size,
        device,
    )
    base.set_seed(seed + 1)
    final_model = base.new_wind_model(panel.features.shape[-1], args, device)
    final_optimizer = torch.optim.AdamW(
        final_model.parameters(),
        lr=args.wind_lr,
        weight_decay=args.wind_weight_decay,
    )
    base.train_wind_epochs(
        final_model,
        loader,
        final_optimizer,
        best_epoch,
        scale,
        args,
        device,
    )
    flat_prediction = base.predict_wind_flat(
        final_model, x_predict, b_predict, args.eval_batch_size, device
    )
    prediction = flat_prediction.reshape(
        len(predict_indices), len(panel.turbines), 24
    )
    stats = {
        "stage": "wind",
        "loss": "normalized_cubic_mae",
        "best_epoch": int(best_epoch),
        "epoch_val_cubic_mae": float(best_loss),
        "epoch_val_wind_mae": best_epoch_mae,
        "wind_scale_p99": scale,
        "n_epoch_train_issues": int(len(epoch_train)),
        "n_epoch_val_issues": int(len(epoch_val)),
        "n_refit_issues": int(len(train_indices)),
        "n_predict_issues": int(len(predict_indices)),
        "n_parameters": int(sum(p.numel() for p in final_model.parameters())),
    }
    del final_model, final_optimizer, loader, final_panel
    base.release_cuda()
    return prediction.astype(np.float32), stats


# v1 became read-only after OneDrive synchronization. Patch its runtime globals so
# every training step and every checkpoint decision uses |v^3-y^3| / s^3.
base.wind_cubic_mse = wind_cubic_mae
base.wind_metrics = wind_metrics_mae
base.select_and_refit_wind = select_and_refit_wind_mae


if __name__ == "__main__":
    base.main()

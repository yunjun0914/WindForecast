from __future__ import annotations

import copy

import numpy as np
import torch

import _bootstrap  # noqa: F401
from experiments import evaluate_two_stage_scada_cubic_tcn_oof_v2 as patch_v2
from utils.per_turbine_sequence import SequenceStandardScaler


base = patch_v2.base


def _raw_base(
    panel: base.TurbinePanel,
    issue_indices: np.ndarray,
    base_index: int,
) -> np.ndarray:
    values = panel.features[issue_indices, :, :, base_index]
    return values.reshape(len(issue_indices) * len(panel.turbines), 24).astype(
        np.float32
    )


def select_and_refit_wind_cubic_mae(
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
    scale = max(float(np.quantile(valid_train, 0.99)), 5.0)

    # TCN inputs are standardized, but the residual anchor remains raw m/s.
    select_scaler = SequenceStandardScaler()
    select_scaler.fit(panel.features[epoch_train])
    select_panel = copy.copy(panel)
    select_panel.features = select_scaler.transform(panel.features)
    x_train, y_train, _, m_train = base.flatten_wind_data(
        select_panel, epoch_train, base_index
    )
    x_val, _, _, _ = base.flatten_wind_data(select_panel, epoch_val, base_index)
    b_train = _raw_base(panel, epoch_train, base_index)
    b_val = _raw_base(panel, epoch_val, base_index)
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
    best_epoch_mae = np.nan
    bad_epochs = 0
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
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.wind_patience:
            break
    if best_epoch <= 0:
        raise RuntimeError("Wind cubic-MAE epoch selection failed")
    del model, optimizer, loader, select_panel
    base.release_cuda()

    final_scaler = SequenceStandardScaler()
    final_scaler.fit(panel.features[train_indices])
    final_panel = copy.copy(panel)
    final_panel.features = final_scaler.transform(panel.features)
    x_train, y_train, _, m_train = base.flatten_wind_data(
        final_panel, train_indices, base_index
    )
    x_predict, _, _, _ = base.flatten_wind_data(
        final_panel, predict_indices, base_index
    )
    b_train = _raw_base(panel, train_indices, base_index)
    b_predict = _raw_base(panel, predict_indices, base_index)
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
        "residual_anchor": "raw_optgrid_ws_calibrated_mps",
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


base.select_and_refit_wind = select_and_refit_wind_cubic_mae


if __name__ == "__main__":
    base.main()

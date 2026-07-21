import numpy as np
import torch

from experiments.evaluate_group_pinn_band_loss_oof import data_loss_weights
from models.group_unified import (
    BoundedResidualMLP,
    GroupPhysicsPINN,
    MultiHeadTCNPowerRegressor,
    normalized_metric_loss,
)


def test_group_pinn_band_loss_weights_are_direct():
    assert data_loss_weights("pure_band_ficr") == (0.0, 1.0)
    assert data_loss_weights("ficr_nmae") == (0.5, 0.5)


def test_normalized_metric_loss_rewards_closer_prediction():
    target = torch.tensor([0.20, 0.40, 0.70])
    close = torch.tensor([0.21, 0.39, 0.71])
    far = torch.tensor([0.35, 0.20, 0.90])
    close_loss, _, close_ficr = normalized_metric_loss(close, target, 0.01, 0.5, 0.5)
    far_loss, _, far_ficr = normalized_metric_loss(far, target, 0.01, 0.5, 0.5)
    assert float(close_loss) < float(far_loss)
    assert float(close_ficr) > float(far_ficr)


def test_group_pinn_and_residual_are_bounded():
    model = GroupPhysicsPINN(
        n_turbines=6,
        rotor_area_m2=12468.0,
        rated_power_w=21_600_000.0,
        c_max=0.45,
    )
    teacher = torch.tensor(
        [[0.0, 0.2, 0.0, 1.0], [8.0, 1.0, 0.5, 0.5], [30.0, 2.0, -0.5, 0.5]]
    )
    prediction = model(
        teacher,
        torch.full((3,), 1.2),
        torch.tensor([1.0, 180.0, 365.0]),
        torch.tensor([12.0, 24.0, 35.0]),
    )
    assert torch.all((prediction >= 0.0) & (prediction <= 1.0))

    residual = BoundedResidualMLP(input_size=5, max_delta=0.12)
    delta = residual(torch.from_numpy(np.ones((4, 5), dtype=np.float32)))
    assert torch.all(delta.abs() <= 0.120001)


def test_multihead_tcn_returns_one_value_per_group():
    model = MultiHeadTCNPowerRegressor(
        input_size=12,
        n_groups=3,
        hidden_size=8,
        num_layers=2,
        kernel_size=3,
        dropout=0.0,
    )
    prediction = model(torch.randn(5, 12, 12))
    assert prediction.shape == (5, 3)

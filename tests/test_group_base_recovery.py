import numpy as np
import pandas as pd
import torch

from experiments.evaluate_group_base_recovery_oof import CausalWindowDataset
from experiments.evaluate_group_tcn_band_loss_oof import soft_band_objective


def test_causal_window_does_not_cross_year_boundary():
    features = np.arange(8, dtype=np.float32).reshape(4, 2)
    targets = np.zeros((4, 3), dtype=np.float32)
    weights = np.ones((4, 3), dtype=np.float32)
    observed = np.ones((4, 3), dtype=np.float32)
    times = pd.to_datetime(
        ["2022-12-31 22:00", "2022-12-31 23:00", "2023-01-01 00:00", "2023-01-01 01:00"]
    )
    dataset = CausalWindowDataset(
        features,
        targets,
        weights,
        observed,
        times,
        window=3,
        keep_observed=False,
    )

    first_2023_window = dataset[2][0]
    assert np.array_equal(first_2023_window, np.repeat(features[2:3], 3, axis=0))


def test_causal_window_can_drop_rows_without_observed_group():
    features = np.zeros((3, 2), dtype=np.float32)
    targets = np.zeros((3, 3), dtype=np.float32)
    weights = np.ones((3, 3), dtype=np.float32)
    observed = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    dataset = CausalWindowDataset(
        features,
        targets,
        weights,
        observed,
        pd.date_range("2023-01-01", periods=3, freq="h"),
        window=2,
        keep_observed=True,
    )
    assert dataset.indices.tolist() == [1, 2]


def test_soft_band_objectives_reward_inside_band_prediction():
    target = torch.full((3, 3), 0.50)
    observed = torch.ones_like(target)
    inside = target + 0.05
    outside = target + 0.10
    for mode in ("pure_band_ficr", "ficr_mae"):
        inside_loss, _, inside_ficr = soft_band_objective(
            inside, target, observed, mode, gamma=0.00682273
        )
        outside_loss, _, outside_ficr = soft_band_objective(
            outside, target, observed, mode, gamma=0.00682273
        )
        assert float(inside_loss) < float(outside_loss)
        assert float(inside_ficr) > float(outside_ficr)

import unittest

import numpy as np
import pandas as pd
import torch

from models.seqnn import TCNRegimeClassifier
from utils.bin_moe import (
    add_centered_weather_regime,
    adjacent_quartile_weights,
    capacity_bin_indices,
    empirical_weather_percentiles,
    fit_weather_quantile_boundaries,
    hard_mix_expert_predictions,
    mix_expert_predictions,
    oracle_mix_expert_predictions,
    weather_quantile_bins,
)


class BinMoeTests(unittest.TestCase):
    def test_capacity_bin_edges(self):
        actual = np.asarray([0.0, 24.9, 25.0, 49.9, 50.0, 74.9, 75.0, 110.0])
        bins = capacity_bin_indices(actual, capacity=100.0)
        np.testing.assert_array_equal(bins, [0, 0, 1, 1, 2, 2, 3, 3])

    def test_soft_hard_and_oracle_mixtures(self):
        experts = np.asarray([[10.0, 20.0, 30.0, 40.0], [1.0, 2.0, 3.0, 4.0]])
        gate = np.asarray([[0.0, 0.25, 0.75, 0.0], [0.6, 0.1, 0.2, 0.1]])

        np.testing.assert_allclose(mix_expert_predictions(experts, gate), [27.5, 1.8])
        np.testing.assert_allclose(hard_mix_expert_predictions(experts, gate), [30.0, 1.0])
        np.testing.assert_allclose(
            oracle_mix_expert_predictions(experts, [1, 3]), [20.0, 4.0]
        )

    def test_regime_classifier_returns_four_logits(self):
        model = TCNRegimeClassifier(
            input_size=5,
            n_classes=4,
            hidden_size=8,
            num_layers=1,
            dropout=0.0,
        )
        output = model(torch.randn(3, 12, 5))
        self.assertEqual(tuple(output.shape), (3, 4))

    def test_centered_weather_regime_stays_inside_issue(self):
        table = pd.DataFrame(
            {
                "forecast_kst_dtm": pd.date_range("2024-01-01", periods=6, freq="h"),
                "data_available_kst_dtm": [
                    pd.Timestamp("2023-12-31 18:00:00")
                ]
                * 3
                + [pd.Timestamp("2024-01-01 00:00:00")] * 3,
                "turbine_id": ["T1"] * 6,
                "optgrid_ws_calibrated": [1.0, 2.0, 100.0, 1000.0, 20.0, 30.0],
            }
        )
        result = add_centered_weather_regime(table)
        np.testing.assert_allclose(
            result["weather_regime_ws"],
            [2.0, 2.0, 2.0, 30.0, 30.0, 30.0],
        )

    def test_weather_quantile_router(self):
        reference = np.arange(8, dtype=float)
        boundaries = fit_weather_quantile_boundaries(reference)
        np.testing.assert_allclose(boundaries, [1.75, 3.5, 5.25])
        np.testing.assert_array_equal(
            weather_quantile_bins([0.0, 2.0, 4.0, 7.0], boundaries),
            [0, 1, 2, 3],
        )
        percentiles = empirical_weather_percentiles([0.5, 2.5, 4.5, 6.5], reference)
        weights = adjacent_quartile_weights(percentiles)
        np.testing.assert_allclose(weights.sum(axis=1), 1.0)
        np.testing.assert_allclose(weights, np.eye(4))


if __name__ == "__main__":
    unittest.main()

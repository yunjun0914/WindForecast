import unittest

import numpy as np
import torch

from models.seqnn import TCNRegimeClassifier
from utils.bin_moe import (
    capacity_bin_indices,
    hard_mix_expert_predictions,
    mix_expert_predictions,
    oracle_mix_expert_predictions,
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


if __name__ == "__main__":
    unittest.main()

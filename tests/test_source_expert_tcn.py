from __future__ import annotations

import unittest

import torch

from experiments.evaluate_source_experts_oof import build_fold_masks
from models.source_expert_tcn import SourceExpertTCN
from utils.source_expert_loss import group_balanced_pure_six_loss, smooth_six_reward


class SourceExpertTCNTest(unittest.TestCase):
    def test_single_source_shape_bounds_and_receptive_field(self):
        model = SourceExpertTCN(
            component_channels=[9],
            spatial_masks=[torch.ones(4, 4)],
            component_hidden_sizes=[64],
            component_embedding_sizes=[64],
            dropout=0.0,
        ).eval()
        output = model(
            [torch.randn(2, 24, 9, 4, 4)],
            torch.randn(2, 24, 5),
        )
        self.assertEqual(tuple(output.shape), (2, 24, 3))
        self.assertTrue(bool(torch.all((output >= 0.0) & (output <= 1.0))))
        self.assertEqual(model.receptive_field, 29)

    def test_gefs_uses_separate_pressure_and_gust_encoders(self):
        model = SourceExpertTCN(
            component_channels=[9, 1],
            spatial_masks=[torch.ones(7, 7), torch.ones(9, 9)],
            component_hidden_sizes=[64, 16],
            component_embedding_sizes=[64, 16],
            dropout=0.0,
        ).eval()
        output = model(
            [torch.randn(1, 24, 9, 7, 7), torch.randn(1, 24, 1, 9, 9)],
            torch.randn(1, 24, 5),
        )
        self.assertEqual(tuple(output.shape), (1, 24, 3))

    def test_pure_six_matches_band_and_generation_weighting(self):
        target = torch.tensor([0.20, 0.80, 0.50])
        prediction = torch.tensor([0.20, 0.80, 0.57])
        reward = smooth_six_reward(target, prediction, 0.50, 0.001)
        self.assertAlmostEqual(float(reward[1] / reward[0]), 4.0, places=4)
        self.assertLess(float(reward[2]), 1e-4)

    def test_group_loss_skips_group_without_labels(self):
        target = torch.tensor([[[0.50, 0.50, 0.0]]])
        prediction = torch.tensor([[[0.50, 0.55, 0.90]]], requires_grad=True)
        valid = torch.tensor([[[1.0, 1.0, 0.0]]])
        loss = group_balanced_pure_six_loss(
            target,
            prediction,
            valid,
            torch.tensor([0.50, 0.50, 1.0]),
            0.01,
        )
        loss.backward()
        self.assertEqual(float(prediction.grad[..., 2]), 0.0)

    def test_fold_masks_drop_cross_year_and_training_fallback(self):
        years = torch.tensor(
            [
                [2022, 2022],
                [2022, 2023],
                [2023, 2023],
                [2024, 2024],
            ]
        ).numpy()
        fallback = torch.tensor([False, False, True, False]).numpy()
        train, validation, cross_year, fallback_removed = build_fold_masks(
            years, fallback, pred_year=2024
        )
        self.assertEqual(train.tolist(), [True, False, False, False])
        self.assertEqual(validation.tolist(), [False, False, False, True])
        self.assertEqual(cross_year.tolist(), [False, True, False, False])
        self.assertEqual(fallback_removed.tolist(), [False, False, True, False])


if __name__ == "__main__":
    unittest.main()

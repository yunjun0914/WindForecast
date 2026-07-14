import unittest

import numpy as np
import pandas as pd
import torch

from models.issue_block_tcn import IssueBlockTCN
from utils.issue_block_dataset import make_issue_blocks
from utils.metrics import TARGET_COLS


class IssueBlockDatasetTests(unittest.TestCase):
    def test_builds_complete_issue_and_drops_cross_year_issue(self):
        issue_one = pd.Timestamp("2023-01-01 00:00:00")
        issue_cross_year = pd.Timestamp("2023-12-30 20:00:00")
        rows = []
        for issue_index, issue in enumerate([issue_one, issue_cross_year]):
            for lead in range(12, 36):
                forecast = issue + pd.Timedelta(hours=lead)
                rows.append(
                    {
                        "forecast_kst_dtm": forecast,
                        "data_available_kst_dtm": issue,
                        "feature": float(issue_index * 100 + lead),
                    }
                )
        weather = pd.DataFrame(rows)
        labels = pd.DataFrame({"kst_dtm": weather["forecast_kst_dtm"].unique()})
        for group_index, group in enumerate(TARGET_COLS):
            labels[group] = np.arange(len(labels), dtype=float) + group_index

        blocks = make_issue_blocks(weather, labels, ["feature"])

        self.assertEqual(blocks.features.shape, (1, 24, 1))
        self.assertEqual(blocks.targets.shape, (1, 24, 3))
        self.assertEqual(blocks.years.tolist(), [2023])
        self.assertEqual(blocks.features[0, :, 0].tolist(), list(map(float, range(12, 36))))


class IssueBlockTcnTests(unittest.TestCase):
    def test_causal_outputs_do_not_use_later_issue_hours(self):
        torch.manual_seed(7)
        model = IssueBlockTCN(
            input_size=4,
            output_size=1,
            hidden_size=8,
            num_layers=2,
            kernel_size=3,
            dropout=0.0,
            full_context=False,
        ).eval()
        original = torch.randn(2, 24, 4)
        changed = original.clone()
        changed[:, 12:, :] += 100.0

        with torch.no_grad():
            original_pred = model(original)
            changed_pred = model(changed)

        self.assertEqual(tuple(original_pred.shape), (2, 24, 1))
        torch.testing.assert_close(original_pred[:, :12], changed_pred[:, :12])

    def test_full_context_shared_heads_output_complete_issue(self):
        model = IssueBlockTCN(
            input_size=4,
            output_size=3,
            hidden_size=8,
            num_layers=2,
            kernel_size=3,
            dropout=0.0,
            full_context=True,
        )
        output = model(torch.randn(2, 24, 4))
        self.assertEqual(tuple(output.shape), (2, 24, 3))


if __name__ == "__main__":
    unittest.main()

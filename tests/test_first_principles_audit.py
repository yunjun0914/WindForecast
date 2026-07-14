import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.audit_first_principles_structure import (
    add_issue_time_offsets,
    prediction_oracle_decomposition,
)
from utils.metrics import TARGET_COLS


class FirstPrinciplesAuditTests(unittest.TestCase):
    def test_issue_offsets_do_not_cross_issue_boundaries(self):
        rows = []
        for issue_index, issue in enumerate(
            [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-02")]
        ):
            for lead in range(12, 15):
                rows.append(
                    {
                        "data_available_kst_dtm": issue,
                        "forecast_kst_dtm": issue + pd.Timedelta(hours=lead),
                        "wind": float(issue_index * 100 + lead),
                    }
                )
        shifted = add_issue_time_offsets(
            pd.DataFrame(rows), wind_columns=["wind"], offsets=(-1, 0, 1)
        )

        first_issue = shifted.loc[
            shifted["data_available_kst_dtm"].eq(pd.Timestamp("2023-01-01"))
        ].reset_index(drop=True)
        self.assertEqual(first_issue.loc[0, "wind_offset_+1"], 13.0)
        self.assertTrue(np.isnan(first_issue.loc[2, "wind_offset_+1"]))
        self.assertTrue(np.isnan(first_issue.loc[0, "wind_offset_-1"]))

    def test_oracle_total_removes_a_pure_scale_error(self):
        times = pd.date_range("2023-01-01", periods=8, freq="h")
        actual_values = {
            "kpx_group_1": 5000.0,
            "kpx_group_2": 6000.0,
            "kpx_group_3": 7000.0,
        }
        parts = []
        for group in TARGET_COLS:
            parts.append(
                pd.DataFrame(
                    {
                        "forecast_kst_dtm": times,
                        "group": group,
                        "pred_year": 2023,
                        "official_target": actual_values[group],
                        "pred": actual_values[group] * 0.8,
                    }
                )
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            pd.concat(parts, ignore_index=True).to_csv(path, index=False)
            scores = prediction_oracle_decomposition(path)

        means = scores.groupby("variant")["mean_score"].first()
        self.assertAlmostEqual(means["oracle_total_keep_pred_share"], 1.0)
        self.assertLess(means["base"], means["oracle_total_keep_pred_share"])
        self.assertAlmostEqual(
            means["base"], means["oracle_share_keep_pred_total"]
        )


if __name__ == "__main__":
    unittest.main()

import unittest

import pandas as pd

from utils.metrics import (
    GROUP_CAPACITY_KWH,
    TARGET_COLS,
    group_nmae_ficr,
    pooled_oof_summary,
    total_score,
)


class PooledOofSummaryTests(unittest.TestCase):
    def test_pools_years_before_equal_group_weighting(self):
        rows = []
        years_by_group = {
            "kpx_group_1": [2022, 2023, 2024],
            "kpx_group_2": [2022, 2023, 2024],
            "kpx_group_3": [2023, 2024],
        }
        errors = {
            "kpx_group_1": 0.04,
            "kpx_group_2": 0.07,
            "kpx_group_3": 0.12,
        }
        for group, years in years_by_group.items():
            capacity = GROUP_CAPACITY_KWH[group]
            for year in years:
                rows.append(
                    {
                        "variant": "candidate",
                        "group": group,
                        "pred_year": year,
                        "official_target": 0.50 * capacity,
                        "pred": (0.50 + errors[group]) * capacity,
                    }
                )

        predictions = pd.DataFrame(rows)
        summary, group_scores = pooled_oof_summary(predictions)

        expected_nmaes = []
        expected_ficrs = []
        for group in TARGET_COLS:
            group_rows = predictions.loc[predictions["group"].eq(group)]
            nmae, ficr = group_nmae_ficr(
                group_rows["official_target"],
                group_rows["pred"],
                GROUP_CAPACITY_KWH[group],
            )
            expected_nmaes.append(nmae)
            expected_ficrs.append(ficr)
        expected_score, _, _ = total_score(expected_nmaes, expected_ficrs)

        self.assertEqual(group_scores["group"].tolist(), TARGET_COLS)
        self.assertAlmostEqual(summary.loc[0, "mean_score"], expected_score)
        self.assertEqual(summary.loc[0, "n_groups"], 3)

    def test_rejects_variant_with_missing_group(self):
        predictions = pd.DataFrame(
            {
                "variant": ["candidate"],
                "group": ["kpx_group_1"],
                "official_target": [12000.0],
                "pred": [12000.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "missing groups"):
            pooled_oof_summary(predictions)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from experiments.blend_source_experts_oof import (
    local_candidates,
    load_aligned_predictions,
    score_prediction,
    select_nested_weights,
    simplex_candidates,
)


class SourceExpertBlendTest(unittest.TestCase):
    def test_loader_accepts_custom_variant_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_inputs = {}
            for source, variant in [
                ("ldaps_core", "ldaps_core_sp"),
                ("gfs_core", "gfs_core_sp"),
                ("gefs_mean_core", "gefs_mean_core"),
            ]:
                path = root / f"{variant}.csv"
                pd.DataFrame(
                    {
                        "forecast_kst_dtm": ["2023-01-01 01:00:00"],
                        "pred_year": [2023],
                        "lead": [12],
                        "variant": [variant],
                        "group": ["kpx_group_1"],
                        "official_target": [10800.0],
                        "pred": [10800.0],
                    }
                ).to_csv(path, index=False)
                source_inputs[source] = (path, variant)
            aligned = load_aligned_predictions(source_inputs)
            self.assertEqual(len(aligned), 1)
            for column in (
                "pred_ldaps_core",
                "pred_gfs_core",
                "pred_gefs_mean_core",
            ):
                self.assertIn(column, aligned.columns)

    def test_simplex_and_local_candidates_respect_constraints(self):
        coarse = simplex_candidates(40)
        self.assertEqual(len(coarse), 861)
        self.assertTrue(all(sum(candidate) == 40 for candidate in coarse))
        local = local_candidates((28, 8, 4))
        self.assertTrue(all(sum(candidate) == 200 for candidate in local))
        self.assertTrue(all(abs(candidate[0] - 140) <= 5 for candidate in local))

        four_expert = simplex_candidates(40, dimensions=4)
        self.assertEqual(len(four_expert), 12341)
        self.assertTrue(all(len(candidate) == 4 for candidate in four_expert))
        self.assertTrue(all(sum(candidate) == 40 for candidate in four_expert))

    def test_nested_weight_selection_never_uses_held_out_year(self):
        rows = []
        for year in [2022, 2023, 2024]:
            for group in ["kpx_group_1", "kpx_group_2", "kpx_group_3"]:
                capacity = 21600 if group != "kpx_group_3" else 21000
                for index in range(8):
                    actual = capacity * (0.30 + 0.05 * index)
                    rows.append(
                        {
                            "forecast_kst_dtm": pd.Timestamp(year, 1, 1)
                            + pd.Timedelta(hours=index),
                            "pred_year": year,
                            "lead": 12 + index,
                            "group": group,
                            "official_target": actual,
                            "pred_ldaps_core": actual,
                            "pred_gfs_core": actual + 0.12 * capacity,
                            "pred_gefs_mean_core": actual - 0.12 * capacity,
                        }
                    )
        frame = pd.DataFrame(rows)
        result = select_nested_weights(frame, held_out_year=2023)
        self.assertEqual(result["train_years"], [2022, 2024])
        self.assertNotIn(2023, result["train_years"])
        np.testing.assert_allclose(result["weights"], [1.0, 0.0, 0.0])

    def test_score_uses_equal_group_weighting(self):
        frame = pd.DataFrame(
            {
                "group": ["kpx_group_1", "kpx_group_2"],
                "official_target": [10800.0, 10800.0],
            }
        )
        score, nmae, ficr, rows = score_prediction(
            frame, np.array([10800.0, 10800.0])
        )
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(nmae, 0.0)
        self.assertAlmostEqual(ficr, 1.0)
        self.assertAlmostEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()

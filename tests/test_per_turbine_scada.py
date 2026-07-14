import unittest

import numpy as np
import pandas as pd

from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.per_turbine_scada import (
    apply_turbine_share_shrinkage,
    build_static_turbine_share_priors,
)


class PerTurbineScadaTests(unittest.TestCase):
    def test_static_share_shrinkage_uses_train_years_and_preserves_group_target(self):
        group = "kpx_group_1"
        turbines = GROUP_TURBINE_PREFIXES[group]
        train_shares = [
            [0.10, 0.12, 0.14, 0.18, 0.20, 0.26],
            [0.12, 0.14, 0.16, 0.18, 0.18, 0.22],
        ]
        val_shares = [0.30, 0.20, 0.15, 0.15, 0.10, 0.10]
        rows = []
        for year, shares in [(2022, train_shares[0]), (2023, train_shares[1]), (2024, val_shares)]:
            for turbine, share in zip(turbines, shares):
                rows.append(
                    {
                        "forecast_kst_dtm": pd.Timestamp(f"{year}-01-01 01:00:00"),
                        "turbine_id": turbine,
                        "year": year,
                        "official_target": 10000.0,
                        "scada_share": share,
                        "turbine_target": 10000.0 * share,
                    }
                )
        targets = pd.DataFrame(rows)

        priors = build_static_turbine_share_priors(targets, group, [2022, 2023])
        expected = np.mean(np.asarray(train_shares), axis=0)
        np.testing.assert_allclose(priors.to_numpy(), expected)
        self.assertAlmostEqual(float(priors.sum()), 1.0)

        shrunk = apply_turbine_share_shrinkage(targets, priors, dynamic_weight=0.25)
        validation = shrunk.loc[shrunk["year"].eq(2024)]
        expected_share = 0.25 * np.asarray(val_shares) + 0.75 * expected
        np.testing.assert_allclose(validation["training_scada_share"], expected_share)
        self.assertAlmostEqual(float(validation["turbine_target"].sum()), 10000.0)

    def test_missing_dynamic_share_remains_excluded_for_matched_ablation(self):
        targets = pd.DataFrame(
            {
                "turbine_id": ["vestas_wtg01"],
                "official_target": [10000.0],
                "scada_share": [np.nan],
                "turbine_target": [np.nan],
            }
        )
        priors = pd.Series({"vestas_wtg01": 1.0})

        shrunk = apply_turbine_share_shrinkage(targets, priors, dynamic_weight=0.0)

        self.assertTrue(shrunk["turbine_target"].isna().all())


if __name__ == "__main__":
    unittest.main()

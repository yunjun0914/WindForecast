from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from utils.per_turbine_fixed_grid import (
    FIXED_TURBINE_GRID_FEATURES,
    FIXED_TURBINE_LDAPS_CONTRACT,
    build_fixed_turbine_grid_features,
    fixed_turbine_grid_contract,
)
from utils.power_curve import GROUP_TURBINE_PREFIXES


class PerTurbineFixedGridTest(unittest.TestCase):
    def test_contract_is_complete_and_has_no_fold_fields(self):
        for group, turbines in GROUP_TURBINE_PREFIXES.items():
            contract = fixed_turbine_grid_contract(group)
            self.assertEqual(contract["turbine_id"].tolist(), list(turbines))
            self.assertEqual(
                len(contract), len(FIXED_TURBINE_LDAPS_CONTRACT[group])
            )
            self.assertEqual(set(contract["source"]), {"ldaps"})
            self.assertNotIn("pred_year", contract.columns)
            self.assertNotIn("train_years", contract.columns)
            self.assertNotIn("slope", contract.columns)
            self.assertNotIn("intercept", contract.columns)

    def test_group1_mapping_matches_dashboard(self):
        contract = fixed_turbine_grid_contract("kpx_group_1")
        actual = {
            row.turbine_id: (row.grid_id, row.level)
            for row in contract.itertuples(index=False)
        }
        self.assertEqual(
            actual,
            {
                "vestas_wtg01": (2, "ws50_midpoint"),
                "vestas_wtg02": (2, "ws50_midpoint"),
                "vestas_wtg03": (2, "ws50_midpoint"),
                "vestas_wtg04": (12, "ws10"),
                "vestas_wtg05": (12, "ws10"),
                "vestas_wtg06": (12, "ws10"),
            },
        )

    def test_midpoint_and_10m_features_use_fixed_raw_vectors(self):
        times = pd.date_range("2022-01-01", periods=2, freq="h")
        rows = []
        for grid_id in (2, 12):
            for time in times:
                rows.append(
                    {
                        "forecast_kst_dtm": time,
                        "data_available_kst_dtm": time - pd.Timedelta(hours=12),
                        "grid_id": grid_id,
                        "heightAboveGround_10_10u": 3.0,
                        "heightAboveGround_10_10v": 4.0,
                        "heightAboveGround_50_50MUmax": 6.0,
                        "heightAboveGround_50_50MUmin": 2.0,
                        "heightAboveGround_50_50MVmax": 2.0,
                        "heightAboveGround_50_50MVmin": -2.0,
                    }
                )
        features, contract = build_fixed_turbine_grid_features(
            pd.DataFrame(rows), "kpx_group_1"
        )
        self.assertEqual(len(features), len(times) * 6)
        self.assertEqual(
            list(features.columns[-4:]), FIXED_TURBINE_GRID_FEATURES
        )

        midpoint = features.loc[features["turbine_id"].eq("vestas_wtg01")]
        self.assertTrue(np.allclose(midpoint["fixedgrid_ws_raw"], 4.0))
        self.assertTrue(np.allclose(midpoint["fixedgrid_ws_cube"], 64.0))
        self.assertTrue(np.allclose(midpoint["fixedgrid_wd_sin"], 0.0))
        self.assertTrue(np.allclose(midpoint["fixedgrid_wd_cos"], 1.0))

        ten_metre = features.loc[features["turbine_id"].eq("vestas_wtg04")]
        self.assertTrue(np.allclose(ten_metre["fixedgrid_ws_raw"], 5.0))
        self.assertTrue(np.allclose(ten_metre["fixedgrid_ws_cube"], 125.0))
        self.assertTrue(np.allclose(ten_metre["fixedgrid_wd_sin"], 0.8))
        self.assertTrue(np.allclose(ten_metre["fixedgrid_wd_cos"], 0.6))
        self.assertEqual(len(contract), 6)


if __name__ == "__main__":
    unittest.main()

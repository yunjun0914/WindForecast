from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils.group_quota_v2 import (
    GROUP_FAMILY_QUOTA64_V2_FEATURES,
    GROUP_QUOTA_V2_FIXED_GRID_COUNTS,
    build_fixed_source_panel,
    fixed_group_grid_contract,
)
from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.tree_feature_profiles import GROUP_FAMILY_QUOTA65_V1_FEATURES


class GroupQuotaV2Test(unittest.TestCase):
    def test_feature_contract_keeps_64_per_group_and_76_union(self):
        self.assertEqual(
            {
                group: len(names)
                for group, names in GROUP_FAMILY_QUOTA64_V2_FEATURES.items()
            },
            {group: 64 for group in GROUP_FAMILY_QUOTA65_V1_FEATURES},
        )
        union = set().union(*map(set, GROUP_FAMILY_QUOTA64_V2_FEATURES.values()))
        self.assertEqual(len(union), 76)
        self.assertTrue(all(name.startswith("gqv2__") for name in union))

    def test_fixed_grid_contract_matches_group_mapping(self):
        expected = {
            "ldaps": {
                "kpx_group_1": {2: 3, 12: 3},
                "kpx_group_2": {2: 2, 3: 1, 7: 2, 12: 1},
                "kpx_group_3": {3: 1, 12: 1, 13: 3},
            },
            "gfs": {
                "kpx_group_1": {2: 2, 4: 3, 7: 1},
                "kpx_group_2": {2: 1, 4: 5},
                "kpx_group_3": {2: 4, 4: 1},
            },
        }
        self.assertEqual(GROUP_QUOTA_V2_FIXED_GRID_COUNTS, expected)
        for source, groups in expected.items():
            for group, counts in groups.items():
                contract = fixed_group_grid_contract(group, source)
                self.assertEqual(
                    dict(zip(contract["grid_id"], contract["group_weight_count"])),
                    counts,
                )
                self.assertEqual(
                    int(contract["group_weight_count"].sum()),
                    len(GROUP_TURBINE_PREFIXES[group]),
                )
                self.assertAlmostEqual(float(contract["group_weight"].sum()), 1.0)
                self.assertNotIn("train_years", contract.columns)
                self.assertNotIn("turbine_id", contract.columns)

    @patch("utils.group_quota_v2.group_site_summary")
    def test_fixed_panel_anchor_uses_immutable_group_weights(self, site_summary):
        site_summary.return_value = {
            "kpx_group_1": {"latitude": 37.28, "longitude": 128.96}
        }
        times = pd.date_range("2022-01-01", periods=2, freq="h")
        rows = []
        for grid_id, value in [(2, 1.0), (12, 3.0)]:
            for time in times:
                rows.append(
                    {
                        "forecast_kst_dtm": time,
                        "data_available_kst_dtm": time - pd.Timedelta(hours=12),
                        "grid_id": grid_id,
                        "latitude": 37.0 + grid_id * 0.01,
                        "longitude": 129.0,
                        "wind": value,
                    }
                )
        raw = pd.DataFrame(rows)
        contract = fixed_group_grid_contract("kpx_group_1", "ldaps")
        panel = build_fixed_source_panel(
            raw, contract, "kpx_group_1", "ldaps"
        )
        anchor = panel.loc[panel["grid_id"].eq("group_anchor")]
        self.assertEqual(len(anchor), 2)
        self.assertTrue(np.allclose(anchor["wind"], (3 * 1.0 + 3 * 3.0) / 6.0))
        self.assertEqual(
            len(panel), 2 * (len(GROUP_TURBINE_PREFIXES["kpx_group_1"]) + 1)
        )

    def test_panel_rejects_modified_contract(self):
        contract = fixed_group_grid_contract("kpx_group_1", "ldaps")
        contract.loc[contract["grid_id"].eq(2), "group_weight_count"] = 2
        with self.assertRaisesRegex(ValueError, "Modified fixed grid contract"):
            build_fixed_source_panel(
                pd.DataFrame(
                    {
                        "forecast_kst_dtm": [],
                        "data_available_kst_dtm": [],
                        "grid_id": [],
                        "latitude": [],
                        "longitude": [],
                    }
                ),
                contract,
                "kpx_group_1",
                "ldaps",
            )


if __name__ == "__main__":
    unittest.main()

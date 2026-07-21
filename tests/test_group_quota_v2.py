from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from utils.group_quota_v2 import (
    GROUP_FAMILY_QUOTA64_V2_FEATURES,
    build_selected_source_panel,
    select_group_source_grids,
)
from utils.per_turbine_optimal_grid_builder import WindCandidateMatrix
from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.tree_feature_profiles import GROUP_FAMILY_QUOTA65_V1_FEATURES


class GroupQuotaV2Test(unittest.TestCase):
    def test_feature_contract_keeps_64_per_group_and_76_union(self):
        self.assertEqual(
            {group: len(names) for group, names in GROUP_FAMILY_QUOTA64_V2_FEATURES.items()},
            {group: 64 for group in GROUP_FAMILY_QUOTA65_V1_FEATURES},
        )
        union = set().union(*map(set, GROUP_FAMILY_QUOTA64_V2_FEATURES.values()))
        self.assertEqual(len(union), 76)
        self.assertTrue(all(name.startswith("gqv2__") for name in union))

    def test_source_selection_uses_only_requested_train_year(self):
        n_per_year = 120
        forecast = pd.date_range("2022-01-01", periods=n_per_year, freq="h").append(
            pd.date_range("2023-01-01", periods=n_per_year, freq="h")
        )
        keys = pd.DataFrame(
            {
                "forecast_kst_dtm": forecast,
                "data_available_kst_dtm": forecast - pd.Timedelta(hours=12),
            }
        )
        target = np.concatenate(
            [np.full(n_per_year, 5.0), np.full(n_per_year, 9.0)]
        ).astype(np.float32)
        ws = np.column_stack(
            [
                np.full(len(forecast), 5.0),
                np.full(len(forecast), 9.0),
                np.full(len(forecast), 20.0),
            ]
        ).astype(np.float32)
        candidates = WindCandidateMatrix(
            keys=keys,
            names=("ldaps|1|ws10", "ldaps|2|ws10", "gfs|5|ws100"),
            u=ws.copy(),
            v=np.zeros_like(ws),
            ws=ws,
        )
        target_parts = []
        for turbine in GROUP_TURBINE_PREFIXES["kpx_group_1"]:
            target_parts.append(
                pd.DataFrame(
                    {
                        "forecast_kst_dtm": forecast,
                        "turbine_id": turbine,
                        "scada_ws_mean": target,
                    }
                )
            )
        targets = pd.concat(target_parts, ignore_index=True)

        selected_2022 = select_group_source_grids(
            candidates,
            targets,
            "kpx_group_1",
            [2022],
            "ldaps",
        )
        selected_2023 = select_group_source_grids(
            candidates,
            targets,
            "kpx_group_1",
            [2023],
            "ldaps",
        )
        self.assertEqual(set(selected_2022["grid_id"]), {1})
        self.assertEqual(set(selected_2023["grid_id"]), {2})

    @patch("utils.group_quota_v2.group_site_summary")
    def test_selected_panel_anchor_is_turbine_weighted(self, site_summary):
        site_summary.return_value = {
            "kpx_group_1": {"latitude": 37.28, "longitude": 128.96}
        }
        times = pd.date_range("2022-01-01", periods=2, freq="h")
        rows = []
        for grid_id, value in [(1, 1.0), (2, 3.0)]:
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
        turbines = GROUP_TURBINE_PREFIXES["kpx_group_1"]
        selections = pd.DataFrame(
            {
                "group": "kpx_group_1",
                "turbine_id": turbines,
                "source": "ldaps",
                "grid_id": [1, 2, 2, 2, 2, 2],
            }
        )
        panel = build_selected_source_panel(
            raw, selections, "kpx_group_1", "ldaps"
        )
        anchor = panel.loc[panel["grid_id"].eq("group_anchor")]
        self.assertEqual(len(anchor), 2)
        self.assertTrue(np.allclose(anchor["wind"], (1.0 + 5 * 3.0) / 6.0))
        self.assertEqual(len(panel), 2 * (len(turbines) + 1))


if __name__ == "__main__":
    unittest.main()

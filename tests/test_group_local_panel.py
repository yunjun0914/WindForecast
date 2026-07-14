import unittest

import numpy as np
import pandas as pd

from utils.group_local_panel import build_group_local_panel


class GroupLocalPanelTests(unittest.TestCase):
    def test_builds_stable_mean_and_turbine_columns(self):
        rows = []
        for hour in range(2):
            for turbine, offset in [("t1", 0.0), ("t2", 10.0)]:
                rows.append(
                    {
                        "forecast_kst_dtm": pd.Timestamp("2024-01-01")
                        + pd.Timedelta(hours=hour),
                        "data_available_kst_dtm": pd.Timestamp("2023-12-31 13:00"),
                        "turbine_id": turbine,
                        "common": 100.0 + hour,
                        "local": offset + hour,
                    }
                )
        panel = build_group_local_panel(
            pd.DataFrame(rows),
            "kpx_group_1",
            turbines=["t1", "t2"],
            common_feature_cols=["common"],
            local_feature_cols=["local"],
        )

        self.assertEqual(panel.mean_feature_cols, ("common", "mean__local"))
        self.assertEqual(
            panel.full_feature_cols,
            ("common", "t1__local", "t2__local"),
        )
        np.testing.assert_allclose(panel.table["mean__local"], [5.0, 6.0])
        np.testing.assert_allclose(panel.table["t1__local"], [0.0, 1.0])
        np.testing.assert_allclose(panel.table["t2__local"], [10.0, 11.0])

    def test_rejects_common_weather_mismatch(self):
        table = pd.DataFrame(
            {
                "forecast_kst_dtm": ["2024-01-01"] * 2,
                "data_available_kst_dtm": ["2023-12-31 13:00"] * 2,
                "turbine_id": ["t1", "t2"],
                "common": [1.0, 2.0],
                "local": [3.0, 4.0],
            }
        )
        with self.assertRaisesRegex(ValueError, "Common weather differs"):
            build_group_local_panel(
                table,
                "kpx_group_1",
                turbines=["t1", "t2"],
                common_feature_cols=["common"],
                local_feature_cols=["local"],
            )


if __name__ == "__main__":
    unittest.main()

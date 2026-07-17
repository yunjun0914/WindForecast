from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from utils.temporal_representation import (
    apply_issue_temporal_representation,
    temporal_aggregation_columns,
)


class TemporalRepresentationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.table = pd.DataFrame(
            {
                "turbine_id": ["A"] * 6,
                "data_available_kst_dtm": pd.to_datetime(
                    ["2024-01-01 00:00"] * 3 + ["2024-01-02 00:00"] * 3
                ),
                "forecast_kst_dtm": pd.to_datetime(
                    [
                        "2024-01-01 01:00",
                        "2024-01-01 02:00",
                        "2024-01-01 03:00",
                        "2024-01-02 01:00",
                        "2024-01-02 02:00",
                        "2024-01-02 03:00",
                    ]
                ),
                "speed": [1.0, 5.0, 3.0, 100.0, 200.0, 300.0],
                "sin_hod": [0.0, 0.5, 1.0, 0.0, -0.5, -1.0],
                "gfs_isobaricInhPa_850_u": [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
            }
        )
        self.features = ["speed", "sin_hod", "gfs_isobaricInhPa_850_u"]

    def test_only_scalar_weather_is_selected(self) -> None:
        self.assertEqual(temporal_aggregation_columns(self.features), ["speed"])

    def test_centered_median_stays_inside_issue(self) -> None:
        output, columns = apply_issue_temporal_representation(
            self.table,
            self.features,
            "median",
            window=3,
        )
        np.testing.assert_allclose(output["speed"], [3.0, 3.0, 4.0, 150.0, 200.0, 250.0])
        np.testing.assert_allclose(output["sin_hod"], self.table["sin_hod"])
        np.testing.assert_allclose(
            output["gfs_isobaricInhPa_850_u"],
            self.table["gfs_isobaricInhPa_850_u"],
        )
        self.assertEqual(columns, ["speed"])

    def test_min_and_max_replace_the_same_column(self) -> None:
        minimum, _ = apply_issue_temporal_representation(
            self.table, self.features, "min", window=3
        )
        maximum, _ = apply_issue_temporal_representation(
            self.table, self.features, "max", window=3
        )
        np.testing.assert_allclose(minimum["speed"], [1.0, 1.0, 3.0, 100.0, 100.0, 200.0])
        np.testing.assert_allclose(maximum["speed"], [5.0, 5.0, 5.0, 200.0, 300.0, 300.0])


if __name__ == "__main__":
    unittest.main()

import unittest

import pandas as pd

from utils.weather_time_features import add_weather_time_features
from utils.per_turbine_optimal_grid import (
    add_optimal_grid_issue_context,
    add_optimal_grid_issue_relative_features,
)


class WeatherTimeFeatureTests(unittest.TestCase):
    def test_time_context_does_not_cross_forecast_issue(self):
        table = pd.DataFrame(
            {
                "forecast_kst_dtm": pd.date_range(
                    "2024-01-01 01:00:00", periods=8, freq="h"
                ),
                "data_available_kst_dtm": [
                    "2023-12-31 13:00:00",
                    "2023-12-31 13:00:00",
                    "2023-12-31 13:00:00",
                    "2023-12-31 13:00:00",
                    "2024-01-01 13:00:00",
                    "2024-01-01 13:00:00",
                    "2024-01-01 13:00:00",
                    "2024-01-01 13:00:00",
                ],
                "wind": [1.0, 2.0, 3.0, 4.0, 100.0, 200.0, 300.0, 400.0],
            }
        )

        out = add_weather_time_features(table, base_cols=["wind"])

        self.assertEqual(out.loc[0, "wind_lead1"], 2.0)
        self.assertEqual(out.loc[0, "wind_lead3"], 4.0)
        self.assertEqual(out.loc[2, "wind_lead3"], 3.0)
        self.assertEqual(out.loc[3, "wind_lead1"], 4.0)
        self.assertEqual(out.loc[3, "wind_lead3"], 4.0)
        self.assertEqual(out.loc[4, "wind_lag1"], 100.0)
        self.assertEqual(out.loc[4, "wind_lag3"], 100.0)
        self.assertEqual(out.loc[4, "wind_roll6_mean"], 100.0)
        self.assertEqual(out.loc[5, "wind_roll6_mean"], 150.0)

    def test_optimal_grid_future_context_stays_inside_issue(self):
        table = pd.DataFrame(
            {
                "forecast_kst_dtm": pd.date_range(
                    "2024-01-01 01:00:00", periods=8, freq="h"
                ),
                "data_available_kst_dtm": [
                    "2023-12-31 13:00:00",
                    "2023-12-31 13:00:00",
                    "2023-12-31 13:00:00",
                    "2023-12-31 13:00:00",
                    "2024-01-01 13:00:00",
                    "2024-01-01 13:00:00",
                    "2024-01-01 13:00:00",
                    "2024-01-01 13:00:00",
                ],
                "turbine_id": ["wtg01"] * 8,
                "optgrid_ws_calibrated": [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    100.0,
                    200.0,
                    300.0,
                    400.0,
                ],
            }
        )

        out = add_optimal_grid_issue_context(table)

        self.assertEqual(out.loc[0, "optgrid_ws_calibrated_lead3"], 4.0)
        self.assertEqual(out.loc[2, "optgrid_ws_calibrated_lead3"], 3.0)
        self.assertEqual(out.loc[3, "optgrid_ws_calibrated_lead1"], 4.0)
        self.assertEqual(out.loc[4, "optgrid_ws_calibrated_lead1"], 200.0)
        self.assertEqual(out.loc[3, "optgrid_ws_calibrated_center7_max"], 4.0)
        self.assertEqual(out.loc[4, "optgrid_ws_calibrated_center7_max"], 400.0)

    def test_optimal_grid_issue_relative_features_stay_inside_turbine_issue(self):
        table = pd.DataFrame(
            {
                "forecast_kst_dtm": list(
                    pd.date_range("2024-01-01 01:00:00", periods=4, freq="h")
                )
                * 2,
                "data_available_kst_dtm": ["2023-12-31 13:00:00"] * 8,
                "turbine_id": ["wtg01"] * 4 + ["wtg02"] * 4,
                "optgrid_ws_calibrated": [1.0, 2.0, 4.0, 3.0, 10.0, 20.0, 40.0, 30.0],
            }
        )

        out = add_optimal_grid_issue_relative_features(table)
        first = out.loc[out["turbine_id"].eq("wtg01")].reset_index(drop=True)
        second = out.loc[out["turbine_id"].eq("wtg02")].reset_index(drop=True)

        self.assertTrue((first["optgrid_ws_calibrated_issue24_mean"] == 2.5).all())
        self.assertTrue((first["optgrid_ws_calibrated_issue24_max"] == 4.0).all())
        self.assertEqual(first.loc[2, "optgrid_ws_calibrated_issue24_rank_pct"], 1.0)
        self.assertEqual(first.loc[0, "optgrid_ws_calibrated_issue24_rank_pct"], 0.25)
        self.assertTrue((second["optgrid_ws_calibrated_issue24_mean"] == 25.0).all())
        self.assertTrue((second["optgrid_ws_calibrated_issue24_max"] == 40.0).all())


if __name__ == "__main__":
    unittest.main()

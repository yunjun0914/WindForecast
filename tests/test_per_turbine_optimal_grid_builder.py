import unittest

import numpy as np
import pandas as pd

from utils.per_turbine_features import GFS_LEVELS, LDAPS_LEVELS
from utils.per_turbine_optimal_grid_builder import (
    WindCandidateMatrix,
    build_wind_candidate_matrix,
    select_optimal_grid_features,
)
from utils.power_curve import GROUP_TURBINE_PREFIXES


def raw_weather(n_grids, levels):
    rows = []
    forecast = pd.Timestamp("2023-01-01 12:00:00")
    issue = pd.Timestamp("2023-01-01 00:00:00")
    for grid_id in range(n_grids):
        row = {
            "forecast_kst_dtm": forecast,
            "data_available_kst_dtm": issue,
            "grid_id": grid_id,
        }
        for level_index, (u_column, v_column) in enumerate(levels.values()):
            row[u_column] = float(grid_id + level_index + 1)
            row[v_column] = float(level_index)
        rows.append(row)
    return pd.DataFrame(rows)


class OptimalGridBuilderTests(unittest.TestCase):
    def test_builds_all_118_grid_level_candidates(self):
        candidates = build_wind_candidate_matrix(
            raw_weather(16, LDAPS_LEVELS),
            raw_weather(9, GFS_LEVELS),
        )
        self.assertEqual(len(candidates.names), 118)
        self.assertEqual(candidates.ws.shape, (1, 118))
        self.assertEqual(len(set(candidates.names)), 118)

    def test_selection_uses_only_requested_train_years(self):
        first_times = pd.date_range("2022-01-01", periods=120, freq="h")
        second_times = pd.date_range("2023-01-01", periods=120, freq="h")
        forecast = first_times.append(second_times)
        issue = forecast - pd.Timedelta(hours=12)
        phase = np.linspace(0.0, 12.0, len(forecast))
        candidate_zero = 5.0 + np.sin(phase)
        candidate_one = 6.0 + np.cos(phase * 1.7)
        target = np.where(
            forecast.year == 2022,
            2.0 * candidate_zero + 1.0,
            2.0 * candidate_one + 1.0,
        )
        matrix = WindCandidateMatrix(
            keys=pd.DataFrame(
                {
                    "forecast_kst_dtm": forecast,
                    "data_available_kst_dtm": issue,
                }
            ),
            names=("good_in_2022", "good_in_2023"),
            u=np.column_stack([candidate_zero, candidate_one]).astype(np.float32),
            v=np.zeros((len(forecast), 2), dtype=np.float32),
            ws=np.column_stack([candidate_zero, candidate_one]).astype(np.float32),
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

        features, selection = select_optimal_grid_features(
            matrix,
            pd.concat(target_parts, ignore_index=True),
            "kpx_group_1",
            [2022],
        )

        self.assertTrue(selection["candidate"].eq("good_in_2022").all())
        self.assertTrue(np.allclose(selection["slope"], 2.0, atol=1e-4))
        self.assertTrue(np.allclose(selection["intercept"], 1.0, atol=1e-4))
        self.assertEqual(
            len(features), len(forecast) * len(GROUP_TURBINE_PREFIXES["kpx_group_1"])
        )


if __name__ == "__main__":
    unittest.main()

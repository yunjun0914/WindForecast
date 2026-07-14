import unittest

import numpy as np
import pandas as pd

from experiments.evaluate_direct_total_decomposition_oof import (
    make_redistributed_variant,
    redistribute_total,
)


class DirectTotalDecompositionTests(unittest.TestCase):
    def test_redistribution_preserves_total_and_base_share(self):
        base = np.array([[20.0, 30.0], [40.0, 10.0]])
        total = np.array([100.0, 80.0])
        redistributed = redistribute_total(total, base, np.array([50.0, 50.0]))

        np.testing.assert_allclose(redistributed.sum(axis=1), total)
        np.testing.assert_allclose(redistributed[0], [40.0, 60.0])
        np.testing.assert_allclose(redistributed[1], [64.0, 16.0])

    def test_zero_base_total_uses_capacity_share(self):
        redistributed = redistribute_total(
            np.array([60.0]),
            np.array([[0.0, 0.0, 0.0]]),
            np.array([20.0, 20.0, 20.0]),
        )
        np.testing.assert_allclose(redistributed, [[20.0, 20.0, 20.0]])

    def test_alpha_zero_reproduces_base_group_predictions(self):
        times = pd.date_range("2023-01-01", periods=2, freq="h")
        base_prediction = pd.DataFrame(
            {
                "kpx_group_1": [4000.0, 5000.0],
                "kpx_group_2": [6000.0, 7000.0],
                "kpx_group_3": [3000.0, 3500.0],
            },
            index=times,
        )
        base_actual = base_prediction + 100.0
        direct = pd.DataFrame(
            {
                "forecast_kst_dtm": times,
                "pred_year": 2023,
                "target": "s12",
                "model": "tcn",
                "actual": [10200.0, 12200.0],
                "pred": [9000.0, 11000.0],
            }
        )

        reconstructed = make_redistributed_variant(
            direct,
            base_prediction,
            base_actual,
            "s12",
            "tcn",
            0.0,
        ).pivot(index="forecast_kst_dtm", columns="group", values="pred")

        pd.testing.assert_frame_equal(
            reconstructed[base_prediction.columns],
            base_prediction.rename_axis(index="forecast_kst_dtm", columns="group"),
            check_freq=False,
        )


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from utils.two_sided_kneedle import (
    KneedleFitError,
    fit_two_sided_kneedle,
    kneedle_mid_mask,
)


class TwoSidedKneedleTests(unittest.TestCase):
    def test_finds_both_sides_of_s_curve(self):
        wind = np.linspace(0.0, 20.0, 6000)
        power = 1.0 / (1.0 + np.exp(-(wind - 10.0) / 1.5))

        result = fit_two_sided_kneedle(wind, power)

        self.assertLess(result.lower_wind, 10.0)
        self.assertGreater(result.upper_wind, 10.0)
        self.assertGreater(result.upper_wind - result.lower_wind, 3.0)
        self.assertLess(result.difference[result.lower_index], 0.0)
        self.assertGreater(result.difference[result.upper_index], 0.0)

    def test_mid_mask_includes_both_knees(self):
        wind = np.linspace(0.0, 20.0, 6000)
        power = 1.0 / (1.0 + np.exp(-(wind - 10.0) / 1.5))
        result = fit_two_sided_kneedle(wind, power)
        query = np.array(
            [result.lower_wind - 0.01, result.lower_wind, 10.0, result.upper_wind]
        )

        np.testing.assert_array_equal(
            kneedle_mid_mask(query, result),
            np.array([False, True, True, True]),
        )

    def test_linear_curve_has_no_two_sided_knees(self):
        wind = np.linspace(0.0, 20.0, 6000)
        power = wind / wind.max()

        with self.assertRaises(KneedleFitError):
            fit_two_sided_kneedle(wind, power)

    def test_rejects_too_few_rows(self):
        with self.assertRaises(KneedleFitError):
            fit_two_sided_kneedle(np.arange(20.0), np.linspace(0.0, 1.0, 20))


if __name__ == "__main__":
    unittest.main()

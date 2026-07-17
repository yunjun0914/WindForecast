import unittest

import numpy as np

from utils.scada_operating_envelope import (
    OperatingEnvelopeFitError,
    fit_operating_envelope,
    soft_operating_gate,
)


class ScadaOperatingEnvelopeTests(unittest.TestCase):
    def test_recovers_model_specific_boundaries(self):
        rng = np.random.default_rng(42)
        wind = rng.uniform(0.0, 32.0, 120_000)
        active_probability = (
            1.0 / (1.0 + np.exp(-(wind - 3.2) / 0.25))
            * 1.0
            / (1.0 + np.exp(-(26.0 - wind) / 0.35))
        )
        active = rng.random(len(wind)) < active_probability
        power_ratio = np.where(active, 0.05 + 0.90 * rng.random(len(wind)), 0.0)

        result = fit_operating_envelope(wind, power_ratio)

        self.assertAlmostEqual(result.cut_in_speed, 3.2, delta=0.6)
        self.assertTrue(result.cut_out_detected)
        self.assertAlmostEqual(result.cut_out_speed, 26.0, delta=0.8)

    def test_leaves_cutout_undetected_without_crossing(self):
        wind = np.linspace(0.0, 22.0, 40_000)
        power_ratio = np.where(wind >= 3.0, 0.4, 0.0)

        result = fit_operating_envelope(wind, power_ratio)

        self.assertFalse(result.cut_out_detected)
        self.assertTrue(np.isnan(result.cut_out_speed))

    def test_soft_gate_uses_detected_boundaries(self):
        wind = np.array([1.0, 5.0, 15.0, 27.0])
        gate = soft_operating_gate(wind, 3.0, 25.0, tau_in=0.5, tau_out=0.5)

        self.assertLess(gate[0], 0.05)
        self.assertGreater(gate[1], 0.95)
        self.assertGreater(gate[2], 0.99)
        self.assertLess(gate[3], 0.05)

    def test_rejects_too_few_rows(self):
        with self.assertRaises(OperatingEnvelopeFitError):
            fit_operating_envelope(np.arange(100.0), np.ones(100))


if __name__ == "__main__":
    unittest.main()

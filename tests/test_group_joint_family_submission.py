import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.predict_group_joint_family_submission import (
    BRANCH_WEIGHTS,
    FINAL_FLOOR,
    PINN_BASE_SHARE,
    PINN_FLOOR,
    TCN_BASE_SHARE,
    build_turbine_mix,
    component_feature_columns,
    epoch_map,
)
from experiments.analyze_jointmix_oof import (
    fit_jointmix_isotonic,
    representative_residual_prediction,
    safe_spearman,
    select_ficr_aware_alpha,
    select_residual_lambda,
)
from experiments.analyze_representative_turbines import (
    binned_power_curve_noise,
    l3_error,
)
from utils.per_turbine_scada import turbine_capacity_kwh
from utils.per_turbine_teacher import TEACHER_FEATURE_COLS


class GroupJointFamilySubmissionTests(unittest.TestCase):
    def test_l3_error_uses_absolute_error_cubed(self):
        actual = np.array([0.0, 4.0])
        prediction = np.array([1.0, 2.0])

        self.assertAlmostEqual(l3_error(actual, prediction), 4.5 ** (1.0 / 3.0))

    def test_binned_power_curve_noise_is_zero_for_deterministic_curve(self):
        wind = np.repeat([4.0, 5.0, 6.0, 7.0], 40)
        power = np.repeat([0.1, 0.2, 0.4, 0.7], 40)

        self.assertAlmostEqual(binned_power_curve_noise(wind, power), 0.0)

    def test_restored_jointmix_constants(self):
        self.assertEqual(PINN_BASE_SHARE, 0.50)
        self.assertEqual(TCN_BASE_SHARE, 0.25)
        self.assertEqual(BRANCH_WEIGHTS, {"pinn": 0.50, "tree": 0.05, "tcn": 0.45})
        self.assertEqual(PINN_FLOOR, 0.20)
        self.assertEqual(FINAL_FLOOR, 0.10)

    def test_pinn_uses_teacher_but_tcn_is_weather_only(self):
        weather = ["optgrid_ws_calibrated", "optgrid_ws_cube"]
        columns = component_feature_columns(weather)
        self.assertEqual(columns["tcn"], weather)
        self.assertEqual(columns["pinn"], [*weather, *TEACHER_FEATURE_COLS])

    def test_pinn_floor_is_applied_to_each_member_before_mixing(self):
        group = "kpx_group_1"
        capacity = turbine_capacity_kwh(group)
        table = pd.DataFrame(
            {
                "group": [group],
                "pinn_base": [0.0],
                "pinn_joint": [capacity],
                "tcn_base": [100.0],
                "tcn_joint": [300.0],
            }
        )
        output = build_turbine_mix(table)
        expected_pinn = 0.50 * (0.20 * capacity) + 0.50 * capacity
        self.assertAlmostEqual(output.loc[0, "pinn_mix"], expected_pinn)
        self.assertAlmostEqual(output.loc[0, "tcn_mix"], 250.0)

    def test_epoch_map_uses_per_turbine_fold_median(self):
        table = pd.DataFrame(
            {
                "group": ["kpx_group_1"] * 3,
                "turbine_id": ["vestas_wtg01"] * 3,
                "best_epoch": [3, 8, 5],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.csv"
            table.to_csv(path, index=False, encoding="utf-8-sig")
            result = epoch_map(path)
        self.assertEqual(result[("kpx_group_1", "vestas_wtg01")], 5)

    def test_mix_has_no_nonfinite_values_for_finite_inputs(self):
        table = pd.DataFrame(
            {
                "group": ["kpx_group_3"],
                "pinn_base": [100.0],
                "pinn_joint": [200.0],
                "tcn_base": [300.0],
                "tcn_joint": [400.0],
            }
        )
        output = build_turbine_mix(table)
        self.assertTrue(np.isfinite(output[["pinn_mix", "tcn_mix"]]).all().all())

    def test_jointmix_isotonic_mapping_is_monotone(self):
        group = "kpx_group_1"
        capacity = 21600.0
        raw = np.linspace(0.10 * capacity, capacity, 800)
        actual = np.clip(0.95 * raw + 400.0 * np.sin(raw / 1500.0), 0.10 * capacity, capacity)
        frame = pd.DataFrame(
            {
                "jointmix_floor10": raw,
                "official_target": actual,
            }
        )
        calibrator = fit_jointmix_isotonic(frame, group)
        calibrated = calibrator.predict(raw)
        self.assertTrue(np.all(np.diff(calibrated) >= -1e-8))
        self.assertGreater(safe_spearman(raw, calibrated), 0.99)

    def test_ficr_aware_alpha_can_reject_harmful_isotonic(self):
        group = "kpx_group_1"
        capacity = 21600.0
        actual = np.full(600, 0.50 * capacity)
        raw = actual.copy()
        harmful = actual + 0.10 * capacity
        frame = pd.DataFrame(
            {
                "jointmix_floor10": raw,
                "official_target": actual,
            }
        )
        alpha, scores = select_ficr_aware_alpha(frame, harmful, group)
        self.assertEqual(alpha, 0.0)
        self.assertGreater(scores[0.0], scores[1.0])

    def test_representative_residual_lambda_one_is_full_sum(self):
        frame = pd.DataFrame(
            {
                "group": ["kpx_group_1"],
                "representative_baseline": [12000.0],
                "spatial_residual": [2000.0],
                "tree": [10000.0],
                "official_target": [13800.0],
            }
        )
        prediction = representative_residual_prediction(frame, 1.0)
        self.assertAlmostEqual(prediction[0], 0.95 * 14000.0 + 0.05 * 10000.0)

    def test_residual_lambda_selection_can_reject_noisy_residual(self):
        frame = pd.DataFrame(
            {
                "group": ["kpx_group_1"] * 600,
                "representative_baseline": [10000.0] * 600,
                "spatial_residual": [3000.0] * 600,
                "tree": [10000.0] * 600,
                "official_target": [10000.0] * 600,
            }
        )
        selected, scores = select_residual_lambda(frame, "kpx_group_1")
        self.assertEqual(selected, 0.0)
        self.assertGreater(scores[0.0], scores[1.0])


if __name__ == "__main__":
    unittest.main()

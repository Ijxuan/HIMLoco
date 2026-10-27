"""Offline checks of heating semantics, without importing Isaac Gym."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    "minich_heat_power", Path(__file__).parents[1] / "scripts/minich_heat_power.py")
heat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heat)


class HeatPowerTests(unittest.TestCase):
    def test_stall_heats_without_mechanical_power(self):
        result = heat.whole_body_statistics([[2., -2.]], [[0., 0.]], [3., 3.])
        self.assertEqual(result["mechanical_abs_mean_W"], 0.)
        self.assertEqual(result["estimated_copper_loss_mean_W"], 24.)

    def test_quadratic_sign_independent_loss(self):
        np.testing.assert_allclose(heat.heat_power([[-2., 4.]], [1., 1.]), [[4., 16.]])

    def test_peak_is_simultaneous_not_sum_of_joint_peaks(self):
        result = heat.whole_body_statistics([[3., 0.], [0., 3.]],
                                            [[0., 0.], [0., 0.]], [1., 1.])
        self.assertEqual(result["estimated_copper_loss_peak_W"], 9.)

    def test_joint_mapping_and_validation(self):
        m = heat.heat_model(["RR_calf_joint", "FL_hip_joint", "FR_thigh_joint"])
        self.assertLess(m["coefficients_W_per_Nm2"][0], m["coefficients_W_per_Nm2"][1])
        with self.assertRaises(ValueError):
            heat.heat_power([[float("nan")]], [1.])
        with self.assertRaises(ValueError):
            heat.heat_model(["unknown"])


if __name__ == "__main__":
    unittest.main()

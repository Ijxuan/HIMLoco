"""Reward timing, dt and clipping checks without Isaac Gym."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location(
    "minich_torque_penalty", Path(__file__).parents[1] / "scripts/minich_torque_penalty.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TorquePenaltyTests(unittest.TestCase):
    def test_final_substep_is_not_physics_average(self):
        np.testing.assert_allclose(
            module.policy_torque_squares([[1., 0.], [3., 4.]], 2), [25.])

    def test_replace_existing_scale_without_double_counting(self):
        s = module.rescore_torque_scale([100.], [.98], -.0003,
                                        existing_scale=-.0002, dt=.02)
        self.assertAlmostEqual(s['baseline_unclipped_rate_mean'], 1.)
        self.assertAlmostEqual(s['new_unclipped_rate_mean'], .97)
        self.assertAlmostEqual(s['mean_penalty_per_policy_step'], .0006)

    def test_clip_after_summing_all_rewards(self):
        s = module.rescore_torque_scale([100., 100.], [.01, .1], -.0002)
        self.assertEqual(s['baseline_zero_clipping_fraction'], 0.)
        self.assertEqual(s['new_zero_clipping_fraction'], .5)


if __name__ == '__main__':
    unittest.main()

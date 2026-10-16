"""Numerical and scope checks for opt-in relative XY command tracking."""
import math
import unittest

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.minich.minich_config import MiniChRoughCfg
import torch


class TestRelativeTrackingLinVel(unittest.TestCase):
    def robot(self, commands, velocities, minich=True, device='cpu'):
        robot = LeggedRobot.__new__(LeggedRobot)
        robot.cfg = MiniChRoughCfg() if minich else LeggedRobotCfg()
        robot.commands = torch.tensor(commands, dtype=torch.float32, device=device)
        robot.base_lin_vel = torch.tensor(velocities, dtype=torch.float32, device=device)
        robot.base_ang_vel = torch.zeros_like(robot.base_lin_vel)
        return robot

    def test_stationary_low_speed_examples(self):
        r = self.robot([[.2, 0., 0.], [.5, 0., 0.], [1., 0., 0.]], [[0., 0., 0.]] * 3)
        torch.testing.assert_close(r._reward_tracking_lin_vel(),
                                   torch.tensor([math.exp(-16/9), math.exp(-4), math.exp(-4)]))

    def test_zero_turn_and_inclusive_translation_gate(self):
        r = self.robot([[0., 0., 0.], [0., 0., .5], [.099, 0., 0.],
                        [.1, 0., 0.], [-.1, 0., 0.]], [[0., 0., 0.]] * 5)
        expected = [1., 1., math.exp(-.099**2/.25), math.exp(-4/9), math.exp(-4/9)]
        torch.testing.assert_close(r._reward_tracking_lin_vel(), torch.tensor(expected))
        r.base_lin_vel[:2, 1] = .1
        torch.testing.assert_close(r._reward_tracking_lin_vel()[:2],
                                   torch.full((2,), math.exp(-.04)))

    def test_direction_symmetry_and_zero_axis_still_penalizes_drift(self):
        r = self.robot([[.5, 0., 0.], [0., .5, 0.], [-.5, 0., 0.], [.3, .4, 0.]],
                       [[.4, .1, 0.], [.1, .4, 0.], [-.4, -.1, 0.], [.2, .3, 0.]])
        torch.testing.assert_close(r._reward_tracking_lin_vel(),
                                   torch.full((4,), math.exp(-.32)))

    def test_upper_cap_and_other_task_legacy_behavior(self):
        commands = [[2., 0., 0.], [.5, 0., 0.], [0., 0., 1.]]
        velocities = [[1.5, 0., 0.], [0., 0., 0.], [.1, .1, 0.]]
        legacy = self.robot(commands, velocities, minich=False)
        torch.testing.assert_close(legacy._reward_tracking_lin_vel(),
                                   torch.tensor([math.exp(-1), math.exp(-1), math.exp(-.08)]))
        relative = self.robot(commands, velocities)
        self.assertAlmostEqual(relative._reward_tracking_lin_vel()[0].item(), math.exp(-1), places=6)
        self.assertEqual(relative.cfg.rewards.scales.tracking_lin_vel, 1.)

    def test_linear_sigma_is_independent_of_angular_sigma(self):
        r = self.robot([[.5, 0., .5]], [[0., 0., 0.]])
        before = r._reward_tracking_lin_vel().clone()
        r.cfg.rewards.tracking_sigma = .5
        torch.testing.assert_close(r._reward_tracking_lin_vel(), before)
        self.assertAlmostEqual(r._reward_tracking_ang_vel().item(), math.exp(-.5), places=6)
        angular = r._reward_tracking_ang_vel().clone()
        r.cfg.rewards.tracking_lin_vel_sigma = .5
        torch.testing.assert_close(r._reward_tracking_ang_vel(), angular)
        self.assertAlmostEqual(r._reward_tracking_lin_vel().item(), math.exp(-2), places=6)

    def test_perfect_tracking_and_reward_registration(self):
        r = self.robot([[0., 0., 0.], [.1, 0., 0.], [0., -.5, 0.], [2., 1., 0.]],
                       [[0., 0., 0.], [.1, 0., 0.], [0., -.5, 0.], [2., 1., 0.]])
        torch.testing.assert_close(r._reward_tracking_lin_vel(), torch.ones(4))
        r.reward_scales = {'tracking_lin_vel': r.cfg.rewards.scales.tracking_lin_vel}
        r.dt, r.num_envs, r.device = .02, 4, 'cpu'
        r._prepare_reward_function()
        torch.testing.assert_close(r.reward_functions[0]() * r.reward_scales['tracking_lin_vel'],
                                   torch.full((4,), .02))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_matches_cpu(self):
        commands = [[0., 0., 1.], [.1, 0., 0.], [.3, .4, 0.], [2., 0., 0.]]
        velocities = [[.1, .1, 0.], [0., 0., 0.], [.1, .2, 0.], [1.8, .1, 0.]]
        cpu = self.robot(commands, velocities)
        gpu = self.robot(commands, velocities, device='cuda:0')
        torch.testing.assert_close(gpu._reward_tracking_lin_vel().cpu(), cpu._reward_tracking_lin_vel())


if __name__ == '__main__':
    unittest.main()

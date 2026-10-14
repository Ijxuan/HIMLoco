"""Check hip mirror signs, independent pairs, command gates and reward wiring."""
import unittest

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.minich.minich_config import MiniChRoughCfg
import torch


class TestHipMirrorReward(unittest.TestCase):
    def robot(self, hips, commands=None):
        robot = LeggedRobot.__new__(LeggedRobot)
        robot.dof_pos = torch.zeros(len(hips), 12)
        robot.dof_pos[:, [0, 3, 6, 9]] = torch.tensor(hips)
        robot.commands = torch.tensor(
            commands if commands is not None else [[0., 0., .2]] * len(hips))
        return robot

    def test_mirrored_inward_and_outward_allow_different_front_rear_angles(self):
        robot = self.robot([[.2, -.2, .4, -.4], [-.3, .3, -.1, .1]])
        torch.testing.assert_close(robot._reward_hip_mirror(), torch.zeros(2))

    def test_pair_errors_add_without_cancellation(self):
        robot = self.robot([[.2, -.1, -.3, .2], [.2, .2, 0., 0.]])
        torch.testing.assert_close(robot._reward_hip_mirror(), torch.tensor([.2, .4]))
        # Thigh/calf angles must not affect hip mirror error.
        robot.dof_pos[:, [1, 2, 4, 5, 7, 8, 10, 11]] = 1.5
        torch.testing.assert_close(robot._reward_hip_mirror(), torch.tensor([.2, .4]))

    def test_command_gate_boundaries_and_both_directions(self):
        commands = [[0., 0., .2], [0., 0., -.2], [.099, 0., .3],
                    [.1, 0., .3], [0., .1, -.3], [.08, .08, .3],
                    [0., 0., .199], [0., 0., -.199], [0., 0., 0.]]
        robot = self.robot([[.2, -.1, 0., 0.]] * len(commands), commands)
        torch.testing.assert_close(robot._reward_hip_mirror(),
                                   torch.tensor([.1, .1, .1, 0., 0., 0., 0., 0., 0.]))

    def test_minich_negative_scale_is_registered_with_dt(self):
        self.assertEqual(LeggedRobotCfg.rewards.scales.hip_mirror, 0.)
        robot = self.robot([[.2, -.1, 0., 0.]])
        robot.reward_scales = {'hip_mirror': MiniChRoughCfg.rewards.scales.hip_mirror}
        robot.dt = .02
        robot.num_envs = 1
        robot.device = 'cpu'
        robot._prepare_reward_function()
        self.assertEqual(robot.reward_names, ['hip_mirror'])
        weighted = robot.reward_functions[0]() * robot.reward_scales['hip_mirror']
        torch.testing.assert_close(weighted, torch.tensor([-.002]))


if __name__ == '__main__':
    unittest.main()

"""COM deadband, world projection, assigned COM offset and reward registration."""
import math
from types import SimpleNamespace
import unittest

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.minich.minich_config import MiniChRoughCfg
import torch


class TestComFootCenterReward(unittest.TestCase):
    def robot(self, xy):
        robot = LeggedRobot.__new__(LeggedRobot)
        robot.num_envs = len(xy)
        robot.device = 'cpu'
        robot.cfg = SimpleNamespace(rewards=MiniChRoughCfg.rewards)
        robot.root_states = torch.zeros(len(xy), 13)
        robot.root_states[:, :2] = torch.tensor(xy)
        robot.root_states[:, 6] = 1.
        robot.base_quat = robot.root_states[:, 3:7]
        robot.base_com_offset = torch.zeros(len(xy), 3)
        robot.feet_pos = torch.tensor([
            [.2, .1, 0.], [.2, -.1, 0.], [-.2, .1, 0.], [-.2, -.1, 0.]
        ]).repeat(len(xy), 1, 1)
        return robot

    def test_deadband_and_radial_excess_in_both_directions(self):
        robot = self.robot([[0., 0.], [.01, 0.], [.02, 0.],
                            [.04, 0.], [-.04, 0.], [0., -.04], [.03, .04]])
        torch.testing.assert_close(robot._reward_com_foot_center(),
                                   torch.tensor([0., 0., 0., .0004, .0004, .0004, .0009]))

    def test_all_feet_contribute_even_when_elevated(self):
        robot = self.robot([[0., 0.]])
        robot.feet_pos[:, 0, 0] += .16  # Center moves 4 cm with one swing foot.
        robot.feet_pos[:, 0, 2] = .3
        torch.testing.assert_close(robot._reward_com_foot_center(), torch.tensor([.0004]))

    def test_world_translation_and_yaw_invariance(self):
        robot = self.robot([[.04, 0.]])
        expected = robot._reward_com_foot_center()
        # Rotate world positions by 90 degrees, then translate the entire robot.
        for positions in (robot.root_states[:, :3], robot.feet_pos):
            xy = positions[..., :2].clone()
            positions[..., 0] = -xy[..., 1] + 3.
            positions[..., 1] = xy[..., 0] - 2.
        robot.base_quat[:] = torch.tensor([0., 0., math.sqrt(.5), math.sqrt(.5)])
        torch.testing.assert_close(robot._reward_com_foot_center(), expected)

    def test_actual_local_com_is_rotated_including_pitch(self):
        robot = self.robot([[0., 0.]])
        robot.base_com_offset[:, 2] = .08
        robot.base_quat[:] = torch.tensor([0., math.sin(math.pi / 12), 0.,
                                          math.cos(math.pi / 12)])
        # Pitch 30 degrees moves the COM horizontally by .08*sin(30) = .04 m.
        torch.testing.assert_close(robot._reward_com_foot_center(), torch.tensor([.0004]))

    def test_negative_weight_and_control_dt_are_applied_once(self):
        self.assertEqual(LeggedRobotCfg.rewards.scales.com_foot_center, 0.)
        robot = self.robot([[.04, 0.]])
        robot.reward_scales = {'com_foot_center': MiniChRoughCfg.rewards.scales.com_foot_center}
        robot.dt = .02
        robot._prepare_reward_function()
        self.assertEqual(robot.reward_names, ['com_foot_center'])
        weighted = robot.reward_functions[0]() * robot.reward_scales['com_foot_center']
        torch.testing.assert_close(weighted, torch.tensor([-.0008]))


if __name__ == '__main__':
    unittest.main()

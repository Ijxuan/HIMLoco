"""Turn gate, default references, deadband, quadratic growth and reward wiring."""
import math
import types
import unittest

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.envs.minich.minich_config import MiniChRoughCfg
import torch


class TestHipPosPenalty(unittest.TestCase):
    def robot(self, offsets_deg, commands=None):
        r = LeggedRobot.__new__(LeggedRobot)
        r.cfg = types.SimpleNamespace(rewards=MiniChRoughCfg.rewards)
        names = ['FL', 'FR', 'RL', 'RR']
        r.default_dof_pos = torch.zeros(1, 12, dtype=torch.float64)
        r.default_dof_pos[:, [0, 3, 6, 9]] = torch.tensor([
            MiniChRoughCfg.init_state.default_joint_angles[n + '_hip_joint']
            for n in names], dtype=torch.float64)
        r.dof_pos = r.default_dof_pos.repeat(len(offsets_deg), 1)
        r.dof_pos[:, [0, 3, 6, 9]] += torch.deg2rad(
            torch.tensor(offsets_deg, dtype=torch.float64))
        r.commands = torch.tensor(commands if commands is not None else
                                  [[0., 0., .2]] * len(offsets_deg), dtype=torch.float64)
        return r

    def test_default_pose_deadband_and_inclusive_edges(self):
        r = self.robot([[0, 0, 0, 0], [3, -3, 3, -3], [-2, 2, -1, 1]])
        torch.testing.assert_close(r._reward_hip_pos_penalty(), torch.zeros(3, dtype=torch.float64))

    def test_each_hip_both_signs_and_quadratic_growth(self):
        # Excess 1 degree and 2 degrees must have a 1:4 cost ratio.
        r = self.robot([[4, 0, 0, 0], [-5, 0, 0, 0], [4, -4, 5, -5]])
        expected = torch.tensor([1., 4., 10.], dtype=torch.float64) * math.radians(1)**2
        torch.testing.assert_close(r._reward_hip_pos_penalty(), expected)
        r.dof_pos[:, [1, 2, 4, 5, 7, 8, 10, 11]] = 2.
        torch.testing.assert_close(r._reward_hip_pos_penalty(), expected)

    def test_command_gate_norm_boundaries_and_both_directions(self):
        commands = [[0, 0, .2], [0, 0, -.2], [.099, 0, .3],
                    [.1, 0, .3], [0, .1, -.3], [.08, .08, .3],
                    [0, 0, .199], [0, 0, -.199], [0, 0, 0]]
        r = self.robot([[4, 0, 0, 0]] * len(commands), commands)
        expected = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float64) * math.radians(1)**2
        torch.testing.assert_close(r._reward_hip_pos_penalty(), expected)

    def test_configurable_deadband(self):
        r = self.robot([[4, -4, 4, -4]])
        r.cfg = types.SimpleNamespace(rewards=types.SimpleNamespace(hip_pos_deadband_deg=5.))
        torch.testing.assert_close(r._reward_hip_pos_penalty(), torch.zeros(1, dtype=torch.float64))

    def test_minich_enabled_other_tasks_disabled_and_dt_scaling(self):
        self.assertEqual(LeggedRobotCfg.rewards.scales.hip_pos_penalty, 0.)
        self.assertEqual(MiniChRoughCfg.rewards.hip_pos_deadband_deg, 3.)
        r = self.robot([[4, -4, 4, -4]])
        r.reward_scales = {'hip_pos_penalty': MiniChRoughCfg.rewards.scales.hip_pos_penalty}
        r.dt, r.num_envs, r.device = .02, 1, 'cpu'
        r._prepare_reward_function()
        self.assertEqual(r.reward_names, ['hip_pos_penalty'])
        actual = r.reward_functions[0]() * r.reward_scales['hip_pos_penalty']
        expected = torch.tensor([-4 * math.radians(1)**2 * .02], dtype=torch.float64)
        torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()

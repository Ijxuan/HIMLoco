"""Synthetic contact trajectories; no simulator or training run."""
import isaacgym
import math
import unittest
import torch
from legged_gym.envs.minich.minich_config import MiniChRoughCfg
from legged_gym.utils.stair_step import StairStep
from legged_gym.envs.base.legged_robot import LeggedRobot
import test_command_stages


class TestStairStep(unittest.TestCase):
    def setUp(self):
        self.r = StairStep(2, ['FR_foot', 'RL_foot', 'RR_foot', 'FL_foot'], 'cpu', MiniChRoughCfg.rewards, .02)
        self.normal = torch.tensor([[math.sin(.2), 0., math.cos(.2)]]).repeat(2, 1)
        self.cmd = torch.zeros(2, 3)
        self.ground = torch.tensor([[0., .08, .08, 0.]]).repeat(2, 1)
        self.z = self.ground.clone()
        self.support = torch.ones(2, 4, dtype=torch.bool)

    def step(self, count=1):
        score = torch.zeros(2)
        for _ in range(count):
            self.r.update(self.normal, self.cmd, self.z, self.ground, self.support)
            score += self.r.score * .02
        return score

    def swing(self, group=0, height=.035, steps=12):
        self.step(3)
        feet = [2, 3] if group == 0 else [0, 1]
        self.support[:, feet] = False
        self.z[:, feet] += height
        score = self.step(steps)
        self.z[:, feet] -= height
        self.support[:, feet] = True
        return score + self.step(3)

    def test_alternation_and_once_only(self):
        torch.testing.assert_close(self.swing(), torch.ones(2))
        self.assertEqual(self.step(5).sum(), 0)
        self.assertEqual(self.swing().sum(), 0)
        torch.testing.assert_close(self.swing(1), torch.ones(2))
        torch.testing.assert_close(self.swing(0), torch.ones(2))

    def test_either_diagonal_can_start(self):
        torch.testing.assert_close(self.swing(1), torch.ones(2))

    def test_sync_tolerance_and_failed_support(self):
        self.step(3)
        self.support[:, 2] = False
        self.z[:, 2] += .035
        self.step(2)
        self.support[:, 3] = False
        self.z[:, 3] += .035
        self.step(10)
        self.support.fill_(True)
        torch.testing.assert_close(self.step(3), torch.full((2,), .75))
        self.setUp()
        self.step(3)
        self.support[:, [2, 3]] = False
        self.z[:, [2, 3]] += .035
        self.step(4)
        self.support[:, 0] = False
        self.step()
        self.support[:, 0] = True
        self.step(7)
        self.support.fill_(True)
        self.assertEqual(self.step(3).sum(), 0)

    def test_small_hops_cannot_accumulate_air_time(self):
        self.step(3)
        self.z[:, [2, 3]] += .035
        for _ in range(3):
            self.support[:, [2, 3]] = False
            self.step(4)
            self.support[:, [2, 3]] = True
            self.step()
        self.assertEqual(self.step(3).sum(), 0)

    def test_stationary_overdue_and_bounded(self):
        self.assertEqual(self.step(100).sum(), 0)
        torch.testing.assert_close(self.r.overdue, torch.ones(2))
        self.swing()
        self.assertEqual(self.r.overdue.sum(), 0)

    def test_gates_and_hysteresis(self):
        self.normal[:] = torch.tensor([0., 0., 1.])
        self.assertEqual(self.swing().sum(), 0)
        for degrees, expected in [(6, True), (4, True), (2, False), (4, False)]:
            angle = math.radians(degrees)
            self.normal[:] = torch.tensor([math.sin(angle), 0., math.cos(angle)])
            self.step()
            self.assertEqual(self.r.enabled.all().item(), expected)
        self.normal[:] = torch.tensor([.2, 0., 1.])
        for axis in (0, 1, 2):
            self.cmd.zero_()
            self.cmd[:, axis] = .11
            self.assertEqual(self.swing().sum(), 0)
            self.assertEqual(self.r.overdue.sum(), 0)

    def test_short_dropout_and_hanging_never_pay(self):
        self.assertEqual(self.swing(steps=1).sum(), 0)
        self.assertEqual(self.swing(steps=50).sum(), 0)
        self.assertTrue((self.r.overdue > 0).all())

    def test_height_bounds(self):
        self.assertTrue((self.swing(height=.005) < 0).all())
        self.assertTrue((self.swing(height=.10) < 0).all())
        torch.testing.assert_close(self.swing(), torch.ones(2))

    def test_single_foot_and_jump_do_not_pay(self):
        self.step(3)
        self.support[:, 2] = False
        self.z[:, 2] += .035
        self.step(12)
        self.support.fill_(True)
        self.assertEqual(self.step(3).sum(), 0)
        self.z[:] = self.ground
        self.step(3)
        self.support.zero_()
        self.z += .035
        self.step(12)
        self.support.fill_(True)
        self.assertEqual(self.step(3).sum(), 0)

    def test_late_partner_and_lost_support(self):
        for lose_support in (False, True):
            self.setUp()
            self.step(3)
            self.support[:, 2] = False
            self.z[:, 2] += .035
            self.step(6)
            self.support[:, 3] = False
            self.z[:, 3] += .035
            if lose_support:
                self.support[:, 0] = False
            self.step(12)
            self.support.fill_(True)
            self.assertEqual(self.step(3).sum(), 0)

    def test_partial_reset_and_command_cancel(self):
        self.step(3)
        self.support[:, [2, 3]] = False
        self.z[:, [2, 3]] += .035
        self.step(12)
        self.r.reset(torch.tensor([0]))
        self.support.fill_(True)
        torch.testing.assert_close(self.step(3), torch.tensor([0., 1.]))
        self.setUp()
        self.step(3)
        self.support[:, [2, 3]] = False
        self.z[:, [2, 3]] += .035
        self.step(12)
        self.cmd[:, 0] = .5
        self.step()
        self.cmd.zero_()
        self.support.fill_(True)
        self.assertEqual(self.step(3).sum(), 0)

    def test_reward_accessors_are_pure(self):
        self.swing()
        robot = LeggedRobot.__new__(LeggedRobot)
        robot.stair_step = self.r
        elapsed = self.r.elapsed.clone()
        for _ in range(3):
            robot._reward_stair_step()
            robot._reward_stair_step_overdue()
        torch.testing.assert_close(self.r.elapsed, elapsed)

    def test_stair_zero_sampling_and_heading_hold(self):
        robot = test_command_stages.TestCommandStages().make_robot()
        robot.cfg.commands.stair_step_stand_probability = 1.
        robot.cfg.rewards = MiniChRoughCfg.rewards
        robot.orientation_target_normal = torch.tensor([[.2, 0., 1.]]).repeat(8, 1)
        robot.orientation_target_normal[0] = torch.tensor([0., 0., 1.])
        robot.orientation_reference_initialized = torch.ones(8, dtype=torch.bool)
        robot.orientation_reference_initialized[1] = False
        robot.episode_length_buf[2] = 0
        robot._sample_stair_stand_commands(torch.arange(8))
        self.assertTrue((robot.command_stage[:3] == robot.COMMAND_NORMAL).all())
        self.assertTrue((robot.command_stage[3:] == robot.COMMAND_STAND).all())
        self.assertTrue((robot.command_steps_remaining[3:] >= 100).all())
        robot._post_physics_step_callback()
        self.assertEqual(robot.commands[3:, :3].abs().sum(), 0)


if __name__ == '__main__':
    unittest.main()

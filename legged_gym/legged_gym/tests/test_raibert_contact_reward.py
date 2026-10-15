import math
import types
import unittest

from legged_gym.envs.base.legged_robot import LeggedRobot
import torch


class TestRaibertTouchdownReward(unittest.TestCase):
    def make_robot(self):
        robot = LeggedRobot.__new__(LeggedRobot)
        robot.device = "cpu"
        robot.num_envs = 1
        robot.raibert_foot_indices = torch.tensor([3, 1, 4, 2], dtype=torch.long)
        robot.raibert_foot_order = torch.tensor([1, 0, 3, 2], dtype=torch.long)
        robot.contact_forces = torch.zeros(1, 5, 3)
        robot.feet_pos = torch.zeros(1, 4, 3)
        robot.root_states = torch.zeros(1, 13)
        robot.root_states[:, 6] = 1.0
        robot.base_quat = robot.root_states[:, 3:7]
        robot.commands = torch.zeros(1, 4)
        robot.raibert_air_steps = torch.zeros(1, 4, dtype=torch.long)
        robot.raibert_last_contacts = torch.zeros(1, 4, dtype=torch.bool)
        robot.raibert_contact_initialized = torch.zeros(1, dtype=torch.bool)
        robot.cfg = types.SimpleNamespace(
            rewards=types.SimpleNamespace(
                raibert_stance_center_x=-0.011010,
                raibert_stance_center_y=0.0,
                raibert_stance_length=0.38,
                raibert_stance_width=0.276335,
                raibert_prediction_time=0.25,
                raibert_min_air_steps=2,
                raibert_contact_force_threshold=1.0,
                raibert_vx_command_threshold=0.05,
                raibert_yaw_command_threshold=0.05,
            )
        )
        return robot

    def set_feet_body_positions(self, robot, positions):
        # feet_pos uses the original [FR, FL, RR, RL] feet_names order.
        for foot_offset, body_position in zip(robot.raibert_foot_order, positions):
            robot.feet_pos[0, foot_offset, :2] = torch.tensor(body_position)

    def test_first_contact_only_initializes_state(self):
        robot = self.make_robot()
        robot.commands[0, 0] = 1.0
        robot.contact_forces[0, robot.raibert_foot_indices, 2] = 2.0

        reward = robot._reward_raibert_heuristic()

        self.assertEqual(float(reward), 0.0)
        self.assertTrue(bool(robot.raibert_contact_initialized[0]))
        self.assertTrue(torch.all(robot.raibert_last_contacts))

    def test_touchdown_is_counted_once_and_uses_xy_only(self):
        robot = self.make_robot()
        robot.commands[0, 0] = 1.0
        robot.raibert_contact_initialized[:] = True
        robot.raibert_last_contacts[:] = False
        robot.raibert_air_steps[:] = 2

        center_x = -0.011010
        half_length = 0.19
        half_width = 0.5 * 0.276335
        target_x = center_x + half_length + 0.5 * 0.25 * 1.0
        positions = [
            (target_x, half_width),
            (target_x, -half_width),
            (center_x - half_length + 0.5 * 0.25, half_width),
            (center_x - half_length + 0.5 * 0.25, -half_width),
        ]
        self.set_feet_body_positions(robot, positions)
        robot.feet_pos[:, :, 2] = 100.0
        robot.contact_forces[0, robot.raibert_foot_indices, 2] = 2.0

        reward = robot._reward_raibert_heuristic()
        repeated_reward = robot._reward_raibert_heuristic()

        self.assertAlmostEqual(float(reward), 0.0, places=6)
        self.assertEqual(float(repeated_reward), 0.0)
        self.assertTrue(torch.all(robot.raibert_last_contacts))

    def test_wz_target_has_expected_sign_for_front_and_rear_feet(self):
        robot = self.make_robot()
        robot.commands[0, 2] = 1.0
        robot.raibert_contact_initialized[:] = True
        robot.raibert_last_contacts[:] = False
        robot.raibert_air_steps[:] = 2
        robot.contact_forces[0, robot.raibert_foot_indices, 2] = 2.0

        center_x = -0.011010
        half_length = 0.19
        half_width = 0.5 * 0.276335
        x_front = center_x + half_length
        x_rear = center_x - half_length
        yaw_shift_front = 0.5 * 0.25 * x_front
        yaw_shift_rear = 0.5 * 0.25 * x_rear
        left_x_shift = -0.5 * 0.25 * half_width
        right_x_shift = -left_x_shift
        positions = [
            (x_front + left_x_shift, half_width + yaw_shift_front),
            (x_front + right_x_shift, -half_width + yaw_shift_front),
            (x_rear + left_x_shift, half_width + yaw_shift_rear),
            (x_rear + right_x_shift, -half_width + yaw_shift_rear),
        ]
        self.set_feet_body_positions(robot, positions)

        reward = robot._reward_raibert_heuristic()

        self.assertAlmostEqual(float(reward), 0.0, places=6)

    def test_inactive_command_does_not_penalize_touchdown(self):
        robot = self.make_robot()
        robot.raibert_contact_initialized[:] = True
        robot.raibert_last_contacts[:] = False
        robot.raibert_air_steps[:] = 2
        robot.contact_forces[0, robot.raibert_foot_indices, 2] = 2.0
        robot.feet_pos[:, robot.raibert_foot_order, :2] = 0.0

        reward = robot._reward_raibert_heuristic()

        self.assertEqual(float(reward), 0.0)

    def test_force_dropout_does_not_repeat_touchdown(self):
        robot = self.make_robot()
        robot.commands[0, 2] = .25
        index = robot.raibert_foot_indices[0]
        values = []
        # Initialize support, reject short dropouts, then allow a true departure.
        for force in (2., .9, 2., .9, 2., 0., 0., 2., 2., .9, 2.):
            robot.contact_forces[0, index, 2] = force
            values.append(robot._reward_raibert_heuristic().item())
        self.assertGreater(values[7], 0.)
        self.assertTrue(all(x == 0. for i, x in enumerate(values) if i != 7))

    def test_each_foot_requires_consecutive_samples(self):
        robot = self.make_robot();robot.commands[0, 2] = -.25
        robot.contact_forces[:, robot.raibert_foot_indices, 2] = 2.
        robot._reward_raibert_heuristic()
        robot.contact_forces[:, robot.raibert_foot_indices[:2], 2] = 0.
        robot._reward_raibert_heuristic()
        # FL returns too early; FR remains below the threshold a second step.
        robot.contact_forces[:, robot.raibert_foot_indices[0], 2] = 2.
        self.assertEqual(robot._reward_raibert_heuristic().item(), 0.)
        robot.contact_forces[:, robot.raibert_foot_indices[1], 2] = 2.
        targets = robot._get_raibert_desired_foot_xy()
        expected = targets[0, 1].square().sum().item()
        self.assertAlmostEqual(robot._reward_raibert_heuristic().item(), expected)
        self.assertEqual(robot._reward_raibert_heuristic().item(), 0.)

    def test_command_gate_does_not_leave_pending_touchdown(self):
        robot = self.make_robot()
        for _ in range(3):robot._reward_raibert_heuristic()
        robot.contact_forces[:, robot.raibert_foot_indices, 2] = 2.
        self.assertEqual(robot._reward_raibert_heuristic().item(), 0.)
        robot.commands[:, 2] = .25
        self.assertEqual(robot._reward_raibert_heuristic().item(), 0.)

    def test_reset_clears_only_selected_contact_counters(self):
        import test_observation_history_reset as reset_tests
        robot = reset_tests.TestObservationHistoryReset().make_robot()
        robot.raibert_air_steps[:] = 2
        robot.reset_idx(torch.tensor([1]))
        self.assertTrue((robot.raibert_air_steps[1] == 0).all())
        self.assertTrue((robot.raibert_air_steps[[0, 2]] == 2).all())
        self.assertFalse(robot.raibert_contact_initialized[1])

    def test_playback_reuses_targets_without_mutating_commands(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from play_minich_flat_height import get_raibert_desired_foot_xy
        robot = self.make_robot()
        saved = robot.commands.clone()
        for command in ((0., 0., .25), (0., 0., -.25), (1., 0., 0.), (0., 0., 0.)):
            displayed = get_raibert_desired_foot_xy(robot, command)
            expected = robot._get_raibert_desired_foot_xy(torch.tensor([command]))
            torch.testing.assert_close(displayed, expected)
            torch.testing.assert_close(robot.commands, saved)
        # Zero command still includes the nonzero stance-center offset.
        xy = get_raibert_desired_foot_xy(robot, (0., 0., 0.))
        self.assertAlmostEqual(xy[0, 0, 0].item(), .19-.011010, places=6)
        # Batched current commands, with opposite turns, retain distinct targets.
        robot.num_envs = 2
        robot.commands = torch.tensor([[0., 0., .25], [0., 0., -.25]])
        xy = get_raibert_desired_foot_xy(robot)
        self.assertEqual(tuple(xy.shape), (2, 4, 2))
        self.assertFalse(torch.allclose(xy[0], xy[1]))

    def test_playback_reads_refreshed_rigid_states_not_cached_feet(self):
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        from play_minich_flat_height import get_raibert_foot_body_xy
        robot = self.make_robot()
        robot.num_bodies = 5
        robot.rigid_body_states = torch.zeros(5, 13)
        target = robot._get_raibert_desired_foot_xy()
        robot.rigid_body_states[robot.raibert_foot_indices, :2] = target[0]
        robot.feet_pos[:] = 99. # stale cached advanced-indexing copy
        robot.sim = None
        robot.gym = types.SimpleNamespace(
            refresh_actor_root_state_tensor=lambda sim: None,
            refresh_rigid_body_state_tensor=lambda sim: None)
        torch.testing.assert_close(get_raibert_foot_body_xy(robot), target)

    def test_world_yaw_rotation_does_not_change_body_frame_error(self):
        from isaacgym.torch_utils import quat_rotate
        robot = self.make_robot();robot.commands[0, 2] = .25
        target = robot._get_raibert_desired_foot_xy()[0]
        xyz = torch.cat((target, torch.full((4, 1), -.26)), dim=1)
        xyz[0, 0] += .02
        for yaw in (0., 1.2):
            quat = torch.tensor([[0., 0., math.sin(yaw/2),
                                  math.cos(yaw/2)]])
            robot.root_states[0, :3] = torch.tensor([1., 2., .3])
            robot.root_states[0, 3:7] = quat[0]
            robot.feet_pos[:, robot.raibert_foot_order, :] = (
                quat_rotate(quat.repeat(4, 1), xyz) + robot.root_states[0, :3])
            robot.raibert_air_steps[:] = 2
            robot.raibert_contact_initialized[:] = True
            robot.contact_forces[:, robot.raibert_foot_indices, 2] = 2.
            self.assertAlmostEqual(robot._reward_raibert_heuristic().item(), .02**2, places=6)


if __name__ == "__main__":
    unittest.main()

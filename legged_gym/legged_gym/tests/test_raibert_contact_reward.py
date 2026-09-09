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
        robot.raibert_last_contacts = torch.zeros(1, 4, dtype=torch.bool)
        robot.raibert_contact_initialized = torch.zeros(1, dtype=torch.bool)
        robot.cfg = types.SimpleNamespace(
            rewards=types.SimpleNamespace(
                raibert_stance_center_x=-0.011010,
                raibert_stance_center_y=0.0,
                raibert_stance_length=0.38,
                raibert_stance_width=0.276335,
                raibert_prediction_time=0.25,
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
        robot.contact_forces[0, robot.raibert_foot_indices, 2] = 2.0
        robot.feet_pos[:, robot.raibert_foot_order, :2] = 0.0

        reward = robot._reward_raibert_heuristic()

        self.assertEqual(float(reward), 0.0)


if __name__ == "__main__":
    unittest.main()

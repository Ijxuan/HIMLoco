"""Exercise command sampling and callback integration without a simulator."""
import isaacgym
import unittest
from unittest.mock import Mock
import torch
import test_command_stages as stages


class TestLongStairCommands(unittest.TestCase):
    def robot(self, heading=True, staged=True):
        r = stages.TestCommandStages().make_robot(20, enabled=staged)
        r.cfg.commands.long_stairs_forward_only = True
        r.cfg.commands.long_stairs_speed_range = [.2, .4]
        r.cfg.commands.gait_training_fraction = .2
        r.cfg.commands.gait_training_speed_range = [.2, .5]
        r.cfg.commands.heading_command = heading
        r.long_stair_mask = torch.arange(20) >= 17
        return r

    def test_sampling_and_callback_preserve_straight_commands(self):
        for heading in (False, True):
            for staged in (False, True):
                r = self.robot(heading, staged)
                r.cfg.commands.in_place_turn_probability = 1.
                r.cfg.commands.turn_to_stand_probability = 0.
                r._resample_commands(torch.arange(20))
                self.assertTrue((r.commands[17:, 0] >= .2).all())
                self.assertTrue((r.commands[17:, 0] <= .4).all())
                self.assertFalse(r.in_place_turn_buf[17:].any())
                self.assertTrue(r.in_place_turn_buf[4:17].all())
                if staged:
                    self.assertTrue((r.command_stage[17:] == r.COMMAND_NORMAL).all())
                vx = r.commands[17:, 0].clone()
                r.commands[17:, 3] = 2.
                for _ in range(5):
                    r._post_physics_step_callback()
                torch.testing.assert_close(r.commands[17:, 0], vx)
                self.assertTrue((r.commands[17:, 1:3] == 0).all())

    def test_lower_boundary_survives_deadband_and_expiry(self):
        r = self.robot()
        r.cfg.commands.long_stairs_speed_range = [.2, .2]
        r.cfg.commands.in_place_turn_probability = 0.
        r.cfg.commands.turn_to_stand_probability = 1.
        r._resample_commands(torch.arange(20))
        r.command_steps_remaining[:] = 1
        r._update_command_stages()
        torch.testing.assert_close(r.commands[17:, 0], torch.full((3,), .2))
        self.assertTrue((r.command_stage[17:] == r.COMMAND_NORMAL).all())
        self.assertTrue((r.command_stage[4:17] == r.COMMAND_STAND).all())

    def test_explicit_zero_range_stays_zero(self):
        r = self.robot()
        r.command_ranges['lin_vel_x'] = [0., 0.]
        r._resample_commands(torch.arange(20))
        r._post_physics_step_callback()
        self.assertTrue((r.commands[17:, :3] == 0).all())

    def test_disabled_or_missing_mask_uses_ordinary_commands(self):
        for missing in (False, True):
            r = self.robot()
            if missing:
                del r.long_stair_mask
            else:
                r.cfg.commands.long_stairs_forward_only = False
            r.cfg.commands.in_place_turn_probability = 1.
            r.cfg.commands.turn_to_stand_probability = 0.
            r._resample_commands(torch.arange(20))
            self.assertTrue(r.in_place_turn_buf[17:].all())
            self.assertTrue((r.command_stage[17:] == r.COMMAND_TURN).all())

    def test_partial_reset_and_stair_stop_exclusion(self):
        r = self.robot()
        r.commands[:] = .7
        r._sample_stair_stand_commands = Mock()
        r._resample_commands(torch.tensor([5, 18]))
        torch.testing.assert_close(r.commands[17], torch.full((4,), .7))
        self.assertEqual(r._sample_stair_stand_commands.call_args.args[0].tolist(), [5])
        self.assertTrue((r.commands[18, 1:3] == 0).all())

    def test_invalid_speed_range_rejected(self):
        for bounds in ([0., .4], [.4, .2], [.2, float('nan')]):
            r = self.robot()
            r.cfg.commands.long_stairs_speed_range = bounds
            with self.assertRaisesRegex(ValueError, 'Long stair speeds'):
                r._resample_commands(torch.tensor([18]))

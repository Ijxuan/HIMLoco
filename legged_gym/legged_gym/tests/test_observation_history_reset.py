import types
import unittest

from legged_gym.envs.base.legged_robot import LeggedRobot
import torch


class TestObservationHistoryReset(unittest.TestCase):
    def make_robot(self):
        robot = LeggedRobot.__new__(LeggedRobot)
        robot.num_envs = 3
        robot.num_actions = 12
        robot.num_one_step_obs = 45
        robot.history_length = 6
        robot.num_one_step_privileged_obs = 51
        robot.num_privileged_obs = 51
        robot.cfg = types.SimpleNamespace(
            terrain=types.SimpleNamespace(curriculum=False, measure_heights=False),
            commands=types.SimpleNamespace(curriculum=False),
            domain_rand=types.SimpleNamespace(randomize_kp=False, randomize_kd=False,
                                            randomize_motor_strength=False),
            env=types.SimpleNamespace(send_timeouts=True),
        )
        robot.obs_scales = types.SimpleNamespace(ang_vel=1., dof_pos=1., dof_vel=1., lin_vel=1.)
        robot.commands = torch.zeros(3, 4)
        robot.commands_scale = torch.ones(3)
        robot.root_states = torch.zeros(3, 13)
        robot.root_states[:, 6] = 1.
        robot.base_quat = robot.root_states[:, 3:7]
        robot.gravity_vec = torch.tensor([[0., 0., -1.]]).repeat(3, 1)
        robot.base_lin_vel = torch.full((3, 3), 99.)
        robot.base_ang_vel = torch.full((3, 3), 99.)
        robot.projected_gravity = torch.full((3, 3), 99.)
        robot.default_dof_pos = torch.zeros(1, 12)
        for name in ('dof_pos', 'dof_vel', 'actions', 'last_actions', 'last_last_actions', 'last_dof_vel'):
            setattr(robot, name, torch.ones(3, 12))
        robot.disturbance = torch.ones(3, 1, 3)
        robot.feet_air_time = torch.ones(3, 4)
        robot.raibert_last_contacts = torch.ones(3, 4, dtype=torch.bool)
        robot.raibert_contact_initialized = torch.ones(3, dtype=torch.bool)
        robot.obs_history_reset_pending = torch.zeros(3, dtype=torch.bool)
        robot.reset_buf = torch.zeros(3, dtype=torch.long)
        robot.time_out_buf = torch.zeros(3, dtype=torch.bool)
        robot.episode_length_buf = torch.full((3,), 1001)
        robot.episode_sums = {}
        robot.extras = {}
        robot.obs_buf = torch.arange(810, dtype=torch.float).view(3, 270)
        robot.privileged_obs_buf = torch.full((3, 51), -99.)
        robot.add_noise = True
        robot.noise_scale_vec = torch.full((45,), 0.01)

        def reset_dofs(ids):
            robot.dof_pos[ids] = 0.2
            robot.dof_vel[ids] = 0.

        def reset_root(ids):
            robot.root_states[ids, 7:10] = 0.3
            robot.root_states[ids, 10:13] = 0.4

        robot._reset_dofs = reset_dofs
        robot._reset_root_states = reset_root
        robot._resample_commands = lambda ids: None
        robot.refresh_actor_rigid_shape_props = lambda ids: None
        return robot

    def test_reset_first_frame_and_subsequent_roll(self):
        for timeout in (False, True):
            with self.subTest(timeout=timeout):
                robot = self.make_robot()
                robot.time_out_buf[1] = timeout
                old_history = robot.obs_buf.clone()
                robot.reset_idx(torch.tensor([1]))
                self.assertTrue(torch.equal(robot.obs_history_reset_pending, torch.tensor([False, True, False])))
                torch.testing.assert_close(robot.base_lin_vel[1], torch.full((3,), 0.3))
                torch.testing.assert_close(robot.base_ang_vel[1], torch.full((3,), 0.4))
                torch.testing.assert_close(robot.projected_gravity[1], robot.gravity_vec[1])
                self.assertTrue(torch.all(robot.actions[1] == 0))
                self.assertTrue(torch.all(robot.disturbance[1] == 0))
                robot.compute_observations()
                frames = robot.obs_buf.view(3, 6, 45)
                self.assertTrue(torch.equal(frames[1], frames[1, :1].repeat(6, 1)))
                self.assertTrue(torch.equal(robot.obs_buf[[0, 2], 45:], old_history[[0, 2], :-45]))
                torch.testing.assert_close(robot.privileged_obs_buf[1, 45:48], torch.full((3,), 0.3))
                self.assertFalse(robot.obs_history_reset_pending.any())
                history = robot.obs_buf.clone()
                robot.dof_pos += 1.
                robot.compute_observations()
                self.assertTrue(torch.equal(robot.obs_buf[:, 45:], history[:, :-45]))

    def test_all_and_empty_resets(self):
        robot = self.make_robot()
        robot.reset_idx(torch.tensor([], dtype=torch.long))
        self.assertFalse(robot.obs_history_reset_pending.any())
        robot.reset_idx(torch.arange(3))
        robot.compute_observations()
        frames = robot.obs_buf.view(3, 6, 45)
        self.assertTrue(torch.equal(frames, frames[:, :1].repeat(1, 6, 1)))


if __name__ == '__main__':
    unittest.main()
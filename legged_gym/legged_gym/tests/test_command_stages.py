"""Command scheduling tests; no PhysX simulator is needed."""
import ast
import copy
import math
from pathlib import Path
import subprocess
import types
import unittest
from unittest.mock import Mock

from legged_gym.envs.base.legged_robot import LeggedRobot
import torch
import test_observation_history_reset as history_tests


class TestCommandStages(unittest.TestCase):
    def make_robot(self, n=8, enabled=True):
        r = LeggedRobot.__new__(LeggedRobot)
        r.device, r.num_envs, r.dt = 'cpu', n, 0.02
        r.cfg = types.SimpleNamespace(
            commands=types.SimpleNamespace(enable_turn_to_stand=enabled,
                turn_to_stand_probability=0.1, turn_duration_range_s=[1., 3.],
                stand_duration_range_s=[2., 5.], resampling_time=10.,
                in_place_turn_probability=0.1, in_place_turn_duration_s=10., heading_command=True),
            terrain=types.SimpleNamespace(measure_heights=False),
            domain_rand=types.SimpleNamespace(push_robots=False, disturbance=False))
        r.command_ranges = {'lin_vel_x': [-1., 1.], 'lin_vel_y': [-1., 1.],
                            'ang_vel_yaw': [-1., 1.], 'heading': [-3.14, 3.14]}
        r.commands = torch.zeros(n, 4)
        r.in_place_turn_buf = torch.zeros(n, dtype=torch.bool)
        r.command_stage = torch.zeros(n, dtype=torch.long)
        r.command_steps_remaining = torch.full((n,), 500, dtype=torch.long)
        r.turn_duration_steps = r._command_duration_steps([1., 3.])
        r.long_turn_duration_steps = 500
        r.stand_duration_steps = r._command_duration_steps([2., 5.])
        r.episode_length_buf = torch.ones(n, dtype=torch.long)
        r.common_step_counter = 1
        r.base_quat = torch.tensor([[0., 0., 0., 1.]]).repeat(n, 1)
        r.forward_vec = torch.tensor([[1., 0., 0.]]).repeat(n, 1)
        return r

    def expire_turns(self, r):
        r.command_stage[:] = r.COMMAND_TURN_TO_STAND
        r.in_place_turn_buf[:] = True
        r.commands[:, 2] = 0.5
        r.command_steps_remaining[:] = 1

    def test_probability_endpoints_and_no_double_transition(self):
        for short in [False, True]:
            r = self.make_robot()
            r.cfg.commands.in_place_turn_probability = 0. if short else 1.
            r.cfg.commands.turn_to_stand_probability = 1. if short else 0.
            r._resample_commands(torch.arange(r.num_envs))
            expected = r.COMMAND_TURN_TO_STAND if short else r.COMMAND_TURN
            self.assertTrue(torch.all(r.command_stage == expected))
            r.command_steps_remaining[:] = 1
            sample = r._resample_commands
            r._resample_commands = Mock(wraps=sample)
            r._update_command_stages()
            ids = r._resample_commands.call_args.args[0]
            self.assertEqual(len(ids), 0 if short else 8)
            if short:
                self.assertTrue(torch.all(r.command_stage == r.COMMAND_STAND))
                self.assertTrue(torch.all(r.commands[:, :3] == 0))
                self.assertFalse(r.in_place_turn_buf.any())
                self.assertTrue(torch.all(r.command_steps_remaining >= 100))
            else:
                self.assertTrue(torch.all(r.command_stage == r.COMMAND_TURN))
                self.assertTrue(torch.all(r.command_steps_remaining == 500))

    def test_scenario_probabilities_and_uniform_yaw(self):
        torch.manual_seed(2026)
        r = self.make_robot(120000)
        r._resample_commands(torch.arange(r.num_envs))
        for stage, expected in [(r.COMMAND_NORMAL, .8), (r.COMMAND_TURN, .1), (r.COMMAND_TURN_TO_STAND, .1)]:
            self.assertAlmostEqual((r.command_stage == stage).float().mean().item(), expected, delta=.003)
        long = r.command_stage == r.COMMAND_TURN
        short = r.command_stage == r.COMMAND_TURN_TO_STAND
        self.assertTrue(torch.all(r.command_steps_remaining[long] == 500))
        self.assertTrue(torch.all((r.command_steps_remaining[short] >= 50) & (r.command_steps_remaining[short] <= 150)))
        self.assertTrue(torch.all(r.commands[long | short, :2] == 0))
        yaw = r.commands[long | short, 2]
        self.assertTrue(torch.all((yaw >= -1) & (yaw <= 1)))
        self.assertAlmostEqual(yaw.mean().item(), 0, delta=.015)
        self.assertAlmostEqual(yaw.square().mean().item(), 1 / 3, delta=.015)
        # Only the selected short-turn scenarios enter stand on expiry.
        r.command_steps_remaining[:] = 1
        r.cfg.commands.in_place_turn_probability = 0.
        r.cfg.commands.turn_to_stand_probability = 0.
        r._update_command_stages()
        self.assertTrue(torch.equal(r.command_stage == r.COMMAND_STAND, short))

    def test_duration_bounds_and_independent_clocks(self):
        r = self.make_robot(10000)
        r.cfg.commands.in_place_turn_probability = 0.
        r.cfg.commands.turn_to_stand_probability = 1.
        r._resample_commands(torch.arange(r.num_envs))
        self.assertTrue(torch.all(r.command_stage == r.COMMAND_TURN_TO_STAND))
        self.assertEqual(r.command_steps_remaining.min().item(), 50)
        self.assertEqual(r.command_steps_remaining.max().item(), 150)
        r.command_steps_remaining[:] = 7
        r.command_steps_remaining[0] = 1
        r.cfg.commands.turn_to_stand_probability = 1.
        commands = r.commands.clone()
        r._update_command_stages()
        self.assertTrue(torch.equal(commands[1:], r.commands[1:]))
        self.assertTrue(torch.all(r.command_steps_remaining[1:] == 6))
        r.command_steps_remaining[0] = 1
        r.cfg.commands.in_place_turn_probability = 0.
        r.cfg.commands.turn_to_stand_probability = 0.
        r._update_command_stages()
        self.assertEqual(r.command_stage[0].item(), r.COMMAND_NORMAL)
        self.assertEqual(r.command_steps_remaining[0].item(), 500)

    def test_heading_and_zero_protection(self):
        r = self.make_robot(3)
        r.command_stage[:] = torch.tensor([r.COMMAND_NORMAL, r.COMMAND_TURN, r.COMMAND_STAND])
        r.in_place_turn_buf[1] = True
        r.commands[:, 3] = 1.
        r.commands[:, 2] = 0.7
        for turn_stage in (r.COMMAND_TURN, r.COMMAND_TURN_TO_STAND):
            r.command_stage[1] = turn_stage
            for _ in range(5):
                r._post_physics_step_callback()
                self.assertAlmostEqual(r.commands[0, 2].item(), 0.5)
                self.assertAlmostEqual(r.commands[1, 2].item(), 0.7)
                self.assertTrue(torch.all(r.commands[2, :3] == 0))

    def test_long_turn_keeps_command_for_ten_seconds(self):
        r = self.make_robot(1)
        r.cfg.commands.in_place_turn_probability = 1.
        r.cfg.commands.turn_to_stand_probability = 0.
        r._resample_commands(torch.tensor([0]))
        command = r.commands.clone()
        r.cfg.commands.in_place_turn_probability = 0.
        for _ in range(499):
            r._update_command_stages()
            self.assertTrue(torch.equal(r.commands, command))
            self.assertEqual(r.command_stage[0].item(), r.COMMAND_TURN)
        self.assertEqual(r.command_steps_remaining[0].item(), 1)
        r._update_command_stages()
        self.assertEqual(r.command_stage[0].item(), r.COMMAND_NORMAL)

    def test_continuity_and_observation_roll(self):
        r = history_tests.TestObservationHistoryReset().make_robot()
        stage = self.make_robot(3)
        r.device, r.dt = stage.device, stage.dt
        r.cfg.commands = stage.cfg.commands
        for name in ['command_stage', 'command_steps_remaining', 'in_place_turn_buf',
                     'turn_duration_steps', 'long_turn_duration_steps', 'stand_duration_steps', 'command_ranges']:
            setattr(r, name, getattr(stage, name))
        self.expire_turns(r)
        names = ['root_states', 'dof_pos', 'dof_vel', 'actions', 'last_actions',
                 'last_last_actions', 'feet_air_time', 'raibert_last_contacts',
                 'raibert_contact_initialized', 'obs_history_reset_pending', 'obs_buf',
                 'privileged_obs_buf']
        before = {name: getattr(r, name).clone() for name in names}
        r._update_command_stages()
        for name in names:
            self.assertTrue(torch.equal(before[name], getattr(r, name)), name)
        r.noise_scale_vec[:3] = 0.  # Commands have no observation noise in production.
        r.compute_observations()
        self.assertTrue(torch.equal(r.obs_buf[:, 45:], before['obs_buf'][:, :-45]))
        self.assertTrue(torch.all(r.obs_buf[:, :3] == 0))

    def test_physical_reset_discards_old_stage(self):
        for timeout in [False, True]:
            r = history_tests.TestObservationHistoryReset().make_robot()
            stage = self.make_robot(3)
            r.device, r.dt = stage.device, stage.dt
            r.cfg.commands = stage.cfg.commands
            r.cfg.commands.curriculum = False
            r.cfg.commands.in_place_turn_probability = 0.
            r.cfg.commands.turn_to_stand_probability = 0.
            for name in ['command_stage', 'command_steps_remaining', 'in_place_turn_buf',
                         'turn_duration_steps', 'long_turn_duration_steps', 'stand_duration_steps', 'command_ranges']:
                setattr(r, name, getattr(stage, name))
            r._resample_commands = types.MethodType(LeggedRobot._resample_commands, r)
            r.command_stage[:] = r.COMMAND_STAND
            r.time_out_buf[1] = timeout
            r.reset_idx(torch.tensor([], dtype=torch.long))
            self.assertTrue(torch.all(r.command_stage == r.COMMAND_STAND))
            r.reset_idx(torch.tensor([1]))
            self.assertEqual(r.command_stage.tolist(), [2, 0, 2])
            self.assertEqual(r.command_steps_remaining[1].item(), 500)
            r.compute_observations()
            frames = r.obs_buf.view(3, 6, 45)
            self.assertTrue(torch.equal(frames[1], frames[1, :1].repeat(6, 1)))
            r.reset_idx(torch.arange(3))
            self.assertTrue(torch.all(r.command_stage == r.COMMAND_NORMAL))

    def test_legacy_sampling_and_timing(self):
        # Execute the pre-change methods against identical fixtures and RNG seeds.
        path = 'legged_gym/legged_gym/envs/base/legged_robot.py'
        source = subprocess.check_output(['git', 'show', '83fec985ffe9a30903d41e5891a1b18b74e25e8d:' + path], text=True)
        tree = ast.parse(source)
        cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'LeggedRobot')
        methods = [x for x in cls.body if isinstance(x, ast.FunctionDef) and
                   x.name in ['_resample_commands', '_post_physics_step_callback']]
        namespace = dict(LeggedRobot._resample_commands.__globals__)
        exec(compile(ast.Module(body=methods, type_ignores=[]), '<legacy>', 'exec'), namespace)
        for enabled in [False, None]:
            new = self.make_robot(enabled=False)
            if enabled is None:
                del new.cfg.commands.enable_turn_to_stand
            old = copy.deepcopy(new)
            old._resample_commands = types.MethodType(namespace['_resample_commands'], old)
            for step in [1, 499, 500, 501, 1000]:
                new.episode_length_buf[:] = old.episode_length_buf[:] = step
                torch.manual_seed(42)
                new._post_physics_step_callback()
                new_rng = torch.get_rng_state()
                torch.manual_seed(42)
                namespace['_post_physics_step_callback'](old)
                self.assertTrue(torch.equal(new.commands, old.commands))
                self.assertTrue(torch.equal(new_rng, torch.get_rng_state()))

    def test_rewards_unchanged(self):
        path = 'legged_gym/legged_gym/envs/base/legged_robot.py'
        before = subprocess.check_output(['git', 'show', '83fec985ffe9a30903d41e5891a1b18b74e25e8d:' + path], text=True)
        after = Path(path).read_text()
        def rewards(source):
            cls = next(x for x in ast.parse(source).body if isinstance(x, ast.ClassDef) and x.name == 'LeggedRobot')
            return {x.name: ast.dump(x) for x in cls.body if isinstance(x, ast.FunctionDef)
                    and (x.name.startswith('_reward_') or x.name in ['compute_reward', 'post_physics_step'])}
        self.assertEqual(rewards(before), rewards(after))

    def test_invalid_durations(self):
        r = self.make_robot()
        for bounds in [[0., 1.], [3., 1.], [math.nan, 2.], [0.001, 0.002], [1.]]:
            with self.assertRaises(ValueError):
                r._command_duration_steps(bounds)

    def test_fixed_duration_float_rounding(self):
        r = self.make_robot()
        self.assertEqual(r._command_duration_steps([1.14, 1.14]), (57, 57))
        r.dt = 4 * float(torch.tensor(.005).item())
        self.assertEqual(r._command_duration_steps([10., 10.]), (500, 500))
        self.assertEqual(r._command_duration_steps([1., 3.]), (50, 150))

    def test_invalid_probability_and_resampling_time(self):
        for probability in [-0.1, 1.1, math.nan, math.inf]:
            r = self.make_robot()
            r.cfg.control = types.SimpleNamespace(decimation=4)
            r.sim_params = types.SimpleNamespace(dt=0.005)
            r.cfg.commands.turn_to_stand_probability = probability
            with self.assertRaisesRegex(ValueError, 'probability'):
                r._parse_cfg(r.cfg)
        r = self.make_robot()
        r.cfg.control = types.SimpleNamespace(decimation=4)
        r.sim_params = types.SimpleNamespace(dt=0.005)
        r.cfg.commands.resampling_time = 0.001
        with self.assertRaisesRegex(ValueError, 'resampling_time'):
            r._parse_cfg(r.cfg)
        r.cfg.commands.resampling_time = 10.
        r.cfg.commands.in_place_turn_probability = .95
        with self.assertRaisesRegex(ValueError, 'probability sum'):
            r._parse_cfg(r.cfg)

    def test_stand_air_time_reward_remains_disabled(self):
        r = self.make_robot(1)
        r.cfg.rewards = types.SimpleNamespace(feet_air_time_target=0.25,
            feet_air_time_min=0.2, feet_air_time_max=0.3)
        r.feet_indices = torch.arange(4)
        r.contact_forces = torch.zeros(1, 4, 3)
        r.contact_forces[:, :, 2] = 2.
        r.last_contacts = torch.zeros(1, 4, dtype=torch.bool)
        r.feet_air_time = torch.full((1, 4), 0.23)
        self.assertEqual(r._reward_feet_air_time().item(), 0.)
        self.assertTrue(torch.all(r.feet_air_time == 0))


if __name__ == '__main__':
    unittest.main()

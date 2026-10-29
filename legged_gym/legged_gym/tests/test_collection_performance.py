"""Check that collection optimizations preserve reset observations and materials."""
import isaacgym
import types
import unittest
from unittest.mock import Mock

import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
import test_observation_history_reset


class CollectionPerformanceTests(unittest.TestCase):
    def test_subset_heights_match_full_batch(self):
        robot = LeggedRobot.__new__(LeggedRobot)
        robot.num_envs, robot.num_height_points, robot.device = 3, 4, 'cpu'
        terrain = types.SimpleNamespace(mesh_type='trimesh', border_size=2.,
                                        horizontal_scale=.1, vertical_scale=.01)
        robot.cfg = types.SimpleNamespace(terrain=terrain)
        robot.terrain = types.SimpleNamespace(cfg=terrain)
        robot.root_states = torch.zeros(3, 13)
        robot.root_states[:, :2] = torch.tensor([[0., 0.], [.5, .3], [-1., .7]])
        yaw = torch.tensor([0., .8, -1.2])
        robot.base_quat = torch.zeros(3, 4)
        robot.base_quat[:, 2] = torch.sin(yaw / 2)
        robot.base_quat[:, 3] = torch.cos(yaw / 2)
        robot.height_points = torch.tensor([[-.2, -.1, 0.], [-.2, .1, 0.],
                                            [.2, -.1, 0.], [.2, .1, 0.]]).repeat(3, 1, 1)
        robot.height_samples = torch.arange(1600).reshape(40, 40)
        full = robot._get_heights()
        for ids in (torch.tensor([2, 0]), torch.tensor([1]), torch.arange(3),
                    torch.tensor([], dtype=torch.long)):
            torch.testing.assert_close(robot._get_heights(ids), full[ids])
        robot.cfg.terrain.mesh_type = 'plane'
        self.assertEqual(robot._get_heights(torch.tensor([1])).shape, (1, 4))

    def test_reset_updates_only_reset_heights(self):
        robot = test_observation_history_reset.TestObservationHistoryReset().make_robot()
        robot.cfg.terrain.measure_heights = True
        robot.measured_heights = torch.arange(12.).reshape(3, 4)
        before = robot.measured_heights.clone()
        robot._get_heights = Mock(return_value=torch.full((1, 4), 77.))
        robot.reset_idx(torch.tensor([1]))
        torch.testing.assert_close(robot._get_heights.call_args.args[0], torch.tensor([1]))
        torch.testing.assert_close(robot.measured_heights[[0, 2]], before[[0, 2]])
        torch.testing.assert_close(robot.measured_heights[1], torch.full((4,), 77.))

    def test_material_randomization_preserves_rng_and_per_shape_values(self):
        devices = ['cpu'] + (['cuda:0'] if torch.cuda.is_available() else [])
        for device in devices:
            for friction_enabled, restitution_enabled in ((True, False), (True, True), (False, False)):
                with self.subTest(device=device, friction=friction_enabled, restitution=restitution_enabled):
                    robot = LeggedRobot.__new__(LeggedRobot)
                    robot.device = device
                    robot.envs = [10, 20, 30]
                    robot.cfg = types.SimpleNamespace(domain_rand=types.SimpleNamespace(
                        randomize_friction=friction_enabled, friction_range=[.2, 1.5],
                        randomize_restitution=restitution_enabled, restitution_range=[0., .4]))
                    robot.friction_coeffs = torch.full((3, 1), .7, device=device)
                    robot.restitution_coeffs = torch.full((3, 1), .1, device=device)
                    ids = torch.tensor([2, 0], device=device)
                    shapes = {handle: [types.SimpleNamespace() for _ in range(5)] for handle in robot.envs}
                    robot.gym = types.SimpleNamespace(
                        get_actor_rigid_shape_properties=lambda handle, actor: shapes[handle],
                        set_actor_rigid_shape_properties=Mock())
                    torch.manual_seed(42)
                    expected_friction = robot.friction_coeffs.clone()
                    expected_restitution = robot.restitution_coeffs.clone()
                    if friction_enabled:
                        expected_friction[ids] = 1.3 * torch.rand(2, 1, device=device) + .2
                    if restitution_enabled:
                        expected_restitution[ids] = .4 * torch.rand(2, 1, device=device)
                    next_random = torch.rand(3, device=device)
                    torch.manual_seed(42)
                    robot.refresh_actor_rigid_shape_props(ids)
                    torch.testing.assert_close(robot.friction_coeffs, expected_friction)
                    torch.testing.assert_close(robot.restitution_coeffs, expected_restitution)
                    torch.testing.assert_close(torch.rand(3, device=device), next_random)
                    for i in (2, 0):
                        for shape in shapes[robot.envs[i]]:
                            self.assertEqual(shape.friction, expected_friction[i].item())
                            self.assertEqual(shape.restitution, expected_restitution[i].item())
                    self.assertEqual(robot.gym.set_actor_rigid_shape_properties.call_count, 2)


if __name__ == '__main__':
    unittest.main()

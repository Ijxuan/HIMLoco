"""Synthetic edge trajectories exercise support, wall contact and reset semantics."""
import isaacgym
import unittest
import types
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.terrain import Terrain
import torch
from legged_gym.envs.minich.minich_config import MiniChRoughCfg
from legged_gym.utils.step_clearance import StepClearance


class TestStepClearance(unittest.TestCase):
    def setUp(self):
        self.s = StepClearance(2, 4, 'cpu', MiniChRoughCfg.rewards)
        self.feet = torch.zeros(2, 4, 3)
        self.feet[..., 0] = .9
        self.feet[..., 2] = .05
        self.force = torch.zeros_like(self.feet)
        self.cmd = torch.tensor([[.5, 0.], [.5, 0.]])

    def height(self, xy):
        return (xy[..., 0] >= 1.).float() * .08

    def update(self, sampler=None):
        self.s.update(self.feet, self.force, self.cmd, sampler or self.height)

    def test_support_and_flat_and_downhill(self):
        self.feet[..., 2] = .02
        self.force[..., 2] = 20
        self.update()
        self.assertFalse(self.s.active.any())
        self.force.zero_(); self.feet[..., 2] = .05
        for sampler in (lambda xy: torch.zeros_like(xy[..., 0]), lambda xy: -self.height(xy)):
            self.update(sampler)
            self.assertFalse(self.s.active.any())

    def test_wall_latch_height_and_release(self):
        self.update()
        self.assertTrue(self.s.active.all())
        self.assertTrue((self.s.cost > 0).all())
        target = self.s.target_height.clone()
        self.feet[..., 0] = .98
        self.force[..., 0] = -50; self.force[..., 2] = 30
        self.update()
        self.assertTrue(self.s.active.all())  # Friction is not a landing.
        torch.testing.assert_close(self.s.target_height, target)
        self.feet[..., 2] = .13  # 8 cm tread + 3 cm sole margin + 2 cm radius
        self.update()
        self.assertTrue((self.s.cost < 1e-10).all())
        self.feet[..., 0] = 1.08
        self.update()
        self.assertFalse(self.s.active.any())
        self.feet[..., 2] = .10
        self.update()
        self.assertEqual(self.s.cost.sum(), 0)

    def test_cancel_and_partial_reset(self):
        self.update(); self.s.reset(torch.tensor([0]))
        self.assertFalse(self.s.active[0].any())
        self.assertTrue(self.s.active[1].all())
        self.cmd.zero_(); self.update()
        self.assertFalse(self.s.active.any())
        self.assertEqual(self.s.cost.sum(), 0)

    def test_reverse_heading_and_four_independent_feet(self):
        self.feet[..., 0] = 1.1; self.cmd[..., 0] = -.5
        self.feet[:, 0, 2] = .02; self.force[:, 0, 2] = 20
        self.update(lambda xy: (xy[..., 0] <= 1.).float() * .08)
        self.assertFalse(self.s.active[:, 0].any())
        self.assertTrue(self.s.active[:, 1:].all())
        self.cmd[..., 0] = .5; self.update()
        self.assertFalse(self.s.active.any())

    def test_next_stair_and_first_edge_only(self):
        def staircase(xy):
            return (xy[..., 0] >= 1.).float() * .08 + (xy[..., 0] >= 1.12).float() * .08
        self.feet[..., 0] = .99
        self.update(staircase)
        torch.testing.assert_close(self.s.target_height, torch.full((2, 4), .11))
        self.feet[..., 0] = 1.08; self.feet[..., 2] = .14
        self.update(staircase); self.update(staircase)
        self.assertTrue(self.s.active.all())
        torch.testing.assert_close(self.s.target_height, torch.full((2, 4), .19))

    def test_grid_sampler_ramp_and_edge(self):
        r = LeggedRobot.__new__(LeggedRobot)
        r.device = 'cpu'
        cfg = types.SimpleNamespace(mesh_type='trimesh', border_size=0.,
                                    horizontal_scale=.1, vertical_scale=.005, slope_treshold=.75)
        r.cfg = types.SimpleNamespace(terrain=cfg)
        r.terrain = types.SimpleNamespace(cfg=cfg)
        x = torch.arange(30) * .1
        r.height_samples = ((.4*x/.005).round()[:, None]).expand(-1, 4)
        points = torch.stack((torch.linspace(.8, 1.2, 41), torch.ones(41)*.15), -1)
        heights = r._sample_step_heights(points)
        self.assertLess((heights[1:]-heights[:-1]).abs().max(), .005)
        r.height_samples = ((x >= 1.).float() * 16)[:, None].expand(-1, 4)
        heights = r._sample_step_heights(torch.tensor([[.99, .15], [1.01, .15]]))
        torch.testing.assert_close(heights, torch.tensor([0., .08]))
        r.terrain = Terrain.__new__(Terrain)
        r.terrain.cfg = cfg
        r.height_samples = ((x < 1.).float() * 16)[:, None].expand(-1, 4)
        # Negative-X ascent: low vertex x=1 shifts to high vertex x=.9.
        heights = r._sample_step_heights(torch.tensor([[.89, .15], [.95, .15]]))
        torch.testing.assert_close(heights, torch.tensor([.08, 0.]))



if __name__ == '__main__':
    unittest.main()

"""Physics-substep cost semantics, independent of Isaac Gym installation."""
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch

path = Path(__file__).parents[1] / 'utils/foot_vertical_force.py'
spec = importlib.util.spec_from_file_location('foot_vertical_force', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def config(**overrides):
    values = dict(foot_vertical_force_flat_threshold=130.,
                  foot_vertical_force_stair_threshold=160.,
                  foot_vertical_force_flat_angle_deg=5.,
                  foot_vertical_force_stair_angle_deg=10.,
                  foot_vertical_force_min_forward_speed=.1)
    values.update(overrides)
    return SimpleNamespace(**values)


class TestFootVerticalForce(unittest.TestCase):
    def make(self, angles=(0.,), speeds=None, decimation=4, device='cpu'):
        self.s = module.FootVerticalForce(len(angles), device, config(), decimation)
        angle = torch.tensor(angles, device=device) * (math.pi/180)
        normal = torch.stack((-angle.sin(), torch.zeros_like(angle), angle.cos()), -1)
        quat = torch.zeros(len(angles), 4, device=device); quat[:, 3] = 1
        commands = torch.zeros(len(angles), 3, device=device)
        commands[:, 0] = .3 if speeds is None else torch.tensor(speeds, device=device)
        self.inputs = normal, quat, commands
        self.s.begin(*self.inputs)
        return self.s

    def test_angle_blend_and_downhill(self):
        s = self.make((-20., 0., 5., 7.5, 10., 25.))
        torch.testing.assert_close(s.threshold, torch.tensor([130.,130.,130.,145.,160.,160.]))

    def test_only_positive_world_vertical_force(self):
        s = self.make()
        force = torch.tensor([[[10000.,-10000.,130.],[0.,0.,-500.]]])
        for _ in range(4): s.accumulate(force)
        self.assertEqual(s.cost.item(), 0.)

    def test_midcycle_peak_is_not_lost_and_dt_once(self):
        s = self.make()
        for fz in [40.,220.,110.,60.]:
            s.accumulate(torch.tensor([[[0.,0.,fz]]]))
        self.assertEqual(s.cost.item(), 90.**2 / 4)
        # Hand-computed physical integral: coefficient * (220-130)^2 * .005.
        self.assertAlmostEqual(float(s.cost * -.0001 * .02), -.00405, places=8)

    def test_duration_and_sum_over_feet(self):
        s = self.make()
        force = torch.tensor([[[0.,0.,140.],[0.,0.,150.]]])
        for _ in range(4): s.accumulate(force)
        self.assertEqual(s.cost.item(), 100.+400.)

    def test_forward_command_gate(self):
        s = self.make((0.,)*5, (0.,-.3,.099,.1,.3))
        force = torch.zeros(5,4,3);force[...,2]=140.
        for _ in range(4):s.accumulate(force)
        torch.testing.assert_close(s.cost,torch.tensor([0.,0.,0.,400.,400.]))

    def test_threshold_does_not_follow_actual_body_pitch(self):
        s = self.make()
        normal,quat,commands=self.inputs
        quat[0]=torch.tensor([0.,math.sin(.2),0.,math.cos(.2)])
        s.begin(normal,quat,commands)
        self.assertEqual(s.threshold.item(),130.)

    def test_yaw_rotation_invariance(self):
        s=self.make((20.,));normal,quat,commands=self.inputs
        normal[0]=torch.tensor([0.,-math.sin(math.radians(20)),math.cos(math.radians(20))])
        quat[0]=torch.tensor([0.,0.,math.sin(math.pi/4),math.cos(math.pi/4)])
        s.begin(normal,quat,commands)
        self.assertAlmostEqual(s.target_uphill_deg.item(),20.,places=4)
        self.assertEqual(s.threshold.item(),160.)

    def test_partial_reset_and_next_action_clear_accumulation(self):
        s=self.make((0.,20.))
        force=torch.zeros(2,4,3);force[...,2]=200.
        s.accumulate(force);other=s.cost[1].item()
        s.reset(torch.tensor([0]))
        self.assertEqual(s.cost[0].item(),0.)
        self.assertEqual(s.cost[1].item(),other)
        s.begin(*self.inputs)
        self.assertFalse(s.cost.any())

    def test_threshold_and_gate_frozen_during_control_cycle(self):
        s=self.make();self.inputs[0][0]=torch.tensor([-1.,0.,0.]);self.inputs[2].zero_()
        for _ in range(4):s.accumulate(torch.tensor([[[0.,0.,140.]]]))
        self.assertEqual(s.cost.item(),100.)

    def test_invalid_config(self):
        for kw in [dict(foot_vertical_force_flat_threshold=-1),
                   dict(foot_vertical_force_stair_threshold=100),
                   dict(foot_vertical_force_stair_angle_deg=5),
                   dict(foot_vertical_force_min_forward_speed=0),
                   dict(foot_vertical_force_flat_threshold=float('nan'))]:
            with self.assertRaises(ValueError):module.FootVerticalForce(1,'cpu',config(**kw),4)

    def test_decimation_rate_independence(self):
        for dec in [1,4,8]:
            s=self.make(decimation=dec)
            for _ in range(dec):s.accumulate(torch.tensor([[[0.,0.,140.]]]))
            self.assertEqual(s.cost.item(),100.)


if __name__=='__main__':unittest.main()

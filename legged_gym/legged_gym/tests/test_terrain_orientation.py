"""Terrain reference geometry, yaw invariance, smoothing and orientation cost."""
import math
import types
import unittest
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.minich.minich_config import MiniChRoughCfg
from isaacgym.torch_utils import quat_from_euler_xyz
import torch


class TestTerrainOrientation(unittest.TestCase):
    def robot(self, yaw=0.):
        r = LeggedRobot.__new__(LeggedRobot)
        r.cfg = types.SimpleNamespace(rewards=types.SimpleNamespace(**{
            k: getattr(MiniChRoughCfg.rewards, k) for k in (
                'terrain_relative_orientation', 'orientation_window_half_length',
                'orientation_window_half_width', 'orientation_deadband_deg',
                'orientation_filter_tau', 'orientation_max_tilt_deg')}),
            terrain=types.SimpleNamespace(measure_heights=True))
        r.num_envs, r.device, r.dt = 1, 'cpu', .02
        x,y = torch.meshgrid(torch.arange(-8,9)*.1, torch.arange(-5,6)*.1, indexing='ij')
        r.height_points = torch.stack((x.flatten(),y.flatten(),torch.zeros(x.numel())),dim=1)[None]
        r.gravity_vec = torch.tensor([[0.,0.,-1.]])
        r.base_quat = quat_from_euler_xyz(torch.zeros(1),torch.zeros(1),torch.tensor([yaw]))
        r.measured_heights = torch.zeros(1,x.numel())
        r._init_orientation_reference()
        return r

    def test_plane_slope_height_offset_and_yaw(self):
        for yaw in (0.,.7,math.pi/2):
            r=self.robot(yaw)
            # Same WORLD slope z=.4*x for every heading.
            p=r.height_points[0]
            r.measured_heights[0]=.4*(p[:,0]*math.cos(yaw)-p[:,1]*math.sin(yaw))+10.
            r._update_orientation_reference()
            expected=torch.tensor([[-.4,0.,1.]])
            expected=expected/expected.norm(dim=1,keepdim=True)
            torch.testing.assert_close(r.orientation_target_normal,expected,atol=2e-6,rtol=1e-5)

    def test_flat_deadband_yaw_and_inversion(self):
        r=self.robot();r._update_orientation_reference()
        for pitch,expected in [(0.,0.),(4.,0.),(5.,0.),(10.,math.radians(5)**2),(180.,math.radians(175)**2)]:
            r.base_quat=quat_from_euler_xyz(torch.zeros(1),torch.tensor([math.radians(pitch)]),torch.tensor([1.]))
            self.assertAlmostEqual(r._reward_orientation().item(),expected,places=5)

    def test_aligned_uphill_and_downhill(self):
        for slope in (.4,-.4):
            r=self.robot();r.measured_heights[0]=slope*r.height_points[0,:,0]
            r._update_orientation_reference()
            # Standard right-handed pitch: nose up is NEGATIVE pitch.
            r.base_quat=quat_from_euler_xyz(torch.zeros(1),torch.tensor([-math.atan(slope)]),torch.zeros(1))
            self.assertLess(r._reward_orientation().item(),1e-10)

    def test_stair_midsection_and_platform(self):
        angles=[]
        for center in (2.01,2.06,2.11,2.16):
            r=self.robot();x=r.height_points[0,:,0]+center
            r.measured_heights[0]=(torch.floor((x-1.)/.2)+1).clamp(0,20)*.08
            r._update_orientation_reference()
            n=r.orientation_target_normal[0]
            angles.append(math.degrees(math.atan2(-n[0],n[2])))
        self.assertTrue(all(abs(a-math.degrees(math.atan(.4)))<3 for a in angles),angles)
        r=self.robot();r.measured_heights[:]=1.6;r._update_orientation_reference()
        torch.testing.assert_close(r.orientation_target_normal,torch.tensor([[0.,0.,1.]]),atol=1e-6,rtol=0)

    def test_filter_convergence_and_reinitialization(self):
        r=self.robot();r._update_orientation_reference()
        r.measured_heights[0]=.4*r.height_points[0,:,0]
        r._update_orientation_reference()
        first=math.degrees(math.atan2(-r.orientation_target_normal[0,0],r.orientation_target_normal[0,2]))
        self.assertTrue(0<first<math.degrees(math.atan(.4)))
        before=r.orientation_target_normal.clone()
        r._reward_orientation();r._reward_orientation()
        torch.testing.assert_close(before,r.orientation_target_normal)
        for _ in range(100):r._update_orientation_reference()
        self.assertAlmostEqual(float(r.orientation_target_normal[0,0]/r.orientation_target_normal[0,2]),-.4,places=5)
        r.orientation_reference_initialized[:]=False;r.measured_heights[:]=0;r._update_orientation_reference()
        torch.testing.assert_close(r.orientation_target_normal,torch.tensor([[0.,0.,1.]]))

    def test_reference_cap_and_legacy(self):
        r=self.robot();r.measured_heights[0]=10*r.height_points[0,:,0];r._update_orientation_reference()
        n=r.orientation_target_normal[0]
        self.assertAlmostEqual(math.degrees(math.atan2(-n[0],n[2])),35.,places=4)
        r.cfg.rewards.terrain_relative_orientation=False;r.projected_gravity=torch.tensor([[.3,.4,-.8]])
        self.assertAlmostEqual(r._reward_orientation().item(),.25,places=6)


if __name__ == '__main__':unittest.main()

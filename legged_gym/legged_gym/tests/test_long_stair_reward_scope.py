"""Exclude ordinary terrain and introductory flat rows from swing rewards."""
import isaacgym
import unittest
from types import SimpleNamespace
import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.minich.minich_config import MiniChRoughCfg
from legged_gym.utils.step_retraction import StepRetraction
from legged_gym.utils.swing_peak_height import SwingPeakHeight


class TestLongStairRewardScope(unittest.TestCase):
    def test_terrain_rows_and_command_gates(self):
        robot = LeggedRobot.__new__(LeggedRobot)
        robot.num_envs = 6; robot.device = 'cpu'; robot.cfg = MiniChRoughCfg()
        robot.long_stair_mask = torch.tensor([False, True, True, True, True, True])
        robot.terrain_levels = torch.tensor([2, 1, 2, 2, 2, 2])
        mask = robot._long_stair_reward_mask()
        torch.testing.assert_close(mask, torch.tensor([False, False, True, True, True, True]))
        cmd = torch.tensor([[.3,0,0], [.3,0,0], [.3,0,0], [-.3,0,0], [0,0,0], [.3,0,.05]])
        r = StepRetraction(6,4,'cpu',robot.cfg.rewards,.02)
        h = SwingPeakHeight(6,4,'cpu',robot.cfg.rewards,.02)
        feet = torch.zeros(6,4,3); base = torch.zeros(6,2); support = torch.ones(6,4,dtype=torch.bool)
        enabled = mask & (cmd[:,0].abs() >= .1) & (cmd[:,2].abs() < .05)
        def update(n):
            for _ in range(n):
                r.update(feet,cmd,cmd[:,:2],support,base,enabled_envs=mask)
                h.update(feet[...,2],support,cmd,enabled)
        update(4); support.zero_(); feet[...,2]=.05; update(2)
        feet[:,:,0] = -torch.sign(cmd[:,0,None])*.01
        update(4); support.fill_(True); update(2)
        included = torch.tensor([2,3]); excluded = torch.tensor([0,1,4,5])
        self.assertTrue((r.stats['positive_sum'][included] > 0).all())
        self.assertTrue((h.stats['positive_sum'][included] > 0).all())
        self.assertEqual(r.stats['count'][excluded].sum().item(),0.)
        self.assertEqual(h.stats['count'][excluded].sum().item(),0.)
        del robot.long_stair_mask
        self.assertFalse(robot._long_stair_reward_mask().any())

if __name__ == '__main__':unittest.main()

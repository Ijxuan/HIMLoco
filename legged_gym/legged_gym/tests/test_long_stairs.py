import isaacgym
import unittest
import copy
import types
import numpy as np
import torch
from legged_gym.envs.minich.minich_config import MiniChRoughCfg
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.terrain import Terrain, long_tread_stairs
from isaacgym import terrain_utils


class TestLongStairs(unittest.TestCase):
    def test_exact_geometry(self):
        t=terrain_utils.SubTerrain(width=80,length=80,horizontal_scale=.1,vertical_scale=.005)
        long_tread_stairs(t)
        h=t.height_field_raw[:,40]
        self.assertEqual(h[40],0)
        self.assertEqual(h[52],16)
        self.assertEqual(h[64],32)
        self.assertEqual(h[51],0)
        self.assertTrue(np.all(h[52:64]==16))
        self.assertTrue(np.all(h[64:]==32))
        np.testing.assert_array_equal(h[1:40],h[79:40:-1])

    def test_columns_and_levels(self):
        cfg=types.SimpleNamespace(**{k:getattr(MiniChRoughCfg.terrain,k) for k in dir(MiniChRoughCfg.terrain) if not k.startswith('_')})
        cfg.mesh_type='heightfield';cfg.border_size=0.;cfg.num_rows=4
        t=Terrain(cfg,20)
        self.assertEqual(t.long_stair_columns,3)
        for j in range(17,20):
            for i in range(4):
                tile=t.height_field_raw[i*80:(i+1)*80,j*80:(j+1)*80]
                self.assertEqual(tile.max(),0 if i<2 else 32)
                self.assertEqual(t.env_origins[i,j,2],0)
        cfg.long_stairs_enabled=False
        old=Terrain(cfg,20)
        self.assertEqual(old.long_stair_columns,0)

    def robot(self):
        r=LeggedRobot.__new__(LeggedRobot)
        r.cfg=MiniChRoughCfg();r.num_envs=3;r.device='cpu';r.dt=.02
        r.long_stair_mask=torch.tensor([True,True,False])
        r.terrain_levels=torch.tensor([2,2,2]);r.terrain_types=torch.zeros(3,dtype=torch.long)
        r.long_stair_success=torch.zeros(3,dtype=torch.bool);r.long_stair_hold=torch.zeros(3)
        r.env_origins=torch.zeros(3,3);r.feet_pos=torch.zeros(3,4,3)
        r.feet_pos[...,0]=2.8;r.feet_pos[...,2]=.18
        r.feet_indices=torch.arange(4);r.contact_forces=torch.zeros(3,4,3);r.contact_forces[...,2]=10
        r.projected_gravity=torch.tensor([[0.,0.,-1.]]*3)
        return r

    def test_success_requires_hold_all_feet_and_tile(self):
        r=self.robot();r.feet_pos[1,0,0]=1.8
        for _ in range(11):r._update_long_stair_success()
        self.assertEqual(r.long_stair_success.tolist(),[True,False,False])
        r.long_stair_success.zero_();r.long_stair_hold.zero_();r.feet_pos[...,1]=5
        for _ in range(11):r._update_long_stair_success()
        self.assertFalse(r.long_stair_success.any())

    def test_curriculum_no_distance_shortcut(self):
        r=self.robot();r.init_done=True
        r.terrain=types.SimpleNamespace(env_length=8.)
        r.root_states=torch.zeros(3,13);r.root_states[:,1]=5
        r.commands=torch.tensor([[.5,0.,0.]]*3);r.max_episode_length_s=20
        r.time_out_buf=torch.ones(3,dtype=torch.bool);r.max_terrain_level=10
        r.terrain_origins=torch.zeros(10,1,3)
        r.long_stair_success[1]=True
        r._update_terrain_curriculum(torch.arange(3))
        self.assertEqual(r.terrain_levels.tolist(),[1,3,3])

if __name__=='__main__':unittest.main()

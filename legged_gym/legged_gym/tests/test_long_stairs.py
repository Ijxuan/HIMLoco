import isaacgym
import unittest
import copy
import types
import numpy as np
import torch
from legged_gym.envs.minich.minich_config import MiniChRoughCfg
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.terrain import Terrain, long_tread_stairs, long_stair_height
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
        cfg.mesh_type='heightfield';cfg.border_size=0.
        # Explicitly exercise the retained short-tread course, not the legacy default.
        cfg.long_stairs_height_levels=[.08,.10,.12,.15,.18];cfg.long_stairs_depth=.20
        t=Terrain(cfg,20)
        self.assertEqual(t.long_stair_columns,3)
        for j in range(17,20):
            for i in range(cfg.num_rows):
                tile=t.height_field_raw[i*80:(i+1)*80,j*80:(j+1)*80]
                rise = int(round(long_stair_height(cfg, i) / cfg.vertical_scale))
                self.assertEqual(tile.max(), 2 * rise)
                self.assertEqual(tile[51,40], 0)
                np.testing.assert_array_equal(tile[52:54,40], [rise, rise])
                self.assertTrue(np.all(tile[54:,40] == 2 * rise))
                self.assertEqual(t.env_origins[i,j,2],0)
        r = LeggedRobot.__new__(LeggedRobot)
        r.cfg = types.SimpleNamespace(terrain=cfg)
        r.num_envs = 20; r.device = 'cpu'; r.terrain = t
        r._get_env_origins()
        self.assertTrue((r.terrain_levels[r.long_stair_mask] <= cfg.long_stairs_start_level).all())
        torch.testing.assert_close(r.long_stair_step_heights,
                                  torch.tensor([0.,0.,.08,.10,.12,.15,.18,.18,.18,.18]))
        cfg.long_stairs_enabled=False
        old=Terrain(cfg,20)
        self.assertEqual(old.long_stair_columns,0)

    def robot(self):
        r=LeggedRobot.__new__(LeggedRobot)
        r.cfg=MiniChRoughCfg();r.cfg.terrain.long_stairs_height_levels=[.08,.10,.12,.15,.18]
        r.cfg.terrain.long_stairs_depth=.20;r.num_envs=3;r.device='cpu';r.dt=.02
        r.long_stair_mask=torch.tensor([True,True,False])
        r.terrain_levels=torch.tensor([2,2,2]);r.terrain_types=torch.zeros(3,dtype=torch.long)
        r.long_stair_success=torch.zeros(3,dtype=torch.bool);r.long_stair_hold=torch.zeros(3)
        r.long_stair_step_heights=torch.tensor([long_stair_height(r.cfg.terrain,i) for i in range(10)])
        r.env_origins=torch.zeros(3,3);r.feet_pos=torch.zeros(3,4,3)
        r.feet_pos[...,0]=2.8;r.feet_pos[...,2]=.18
        r.feet_indices=torch.arange(4);r.contact_forces=torch.zeros(3,4,3);r.contact_forces[...,2]=10
        r.projected_gravity=torch.tensor([[0.,0.,-1.]]*3)
        return r

    def test_success_requires_hold_all_feet_and_tile(self):
        r=self.robot();r.feet_pos[1,0,0]=1.3
        for _ in range(11):r._update_long_stair_success()
        self.assertEqual(r.long_stair_success.tolist(),[True,False,False])
        r.long_stair_success.zero_();r.long_stair_hold.zero_();r.feet_pos[...,1]=5
        for _ in range(11):r._update_long_stair_success()
        self.assertFalse(r.long_stair_success.any())

    def test_success_uses_current_height_and_both_landings(self):
        r=self.robot();r.terrain_levels[:2]=torch.tensor([2,6])
        r.feet_pos[...,0]=1.6
        # Low-stage height must not count as success on the 18 cm stage.
        for _ in range(11):r._update_long_stair_success()
        self.assertEqual(r.long_stair_success.tolist(),[True,False,False])
        r.feet_pos[1,:,0]=-1.6;r.feet_pos[1,:,2]=.38
        for _ in range(11):r._update_long_stair_success()
        self.assertTrue(r.long_stair_success[1])

    def test_height_schedule_and_invalid_geometry(self):
        cfg=MiniChRoughCfg().terrain
        cfg.long_stairs_height_levels=[.08,.10,.12,.15,.18]
        self.assertEqual([long_stair_height(cfg,i) for i in range(2,7)], [.08,.10,.12,.15,.18])
        t=terrain_utils.SubTerrain(width=80,length=80,horizontal_scale=.1,vertical_scale=.005)
        for approach in [0., -.1, .25, 4.]:
            with self.assertRaises(ValueError):
                long_tread_stairs(t, .18, .2, 2, approach=approach)

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

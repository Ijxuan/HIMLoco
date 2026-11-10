"""Compare the reconstructed baseline with the pinned historical source."""
import isaacgym
import subprocess
import unittest
from types import SimpleNamespace
import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from pathlib import Path
from legged_gym.envs.minich.minich_config import MiniChRoughCfg, MiniChRoughCfgPPO
from legged_gym.utils import class_to_dict
from legged_gym.utils.terrain import Terrain
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
REV = '73cd811'


class TestHistoricalTrainingConfig(unittest.TestCase):
    def test_all_historical_config_values_except_documented_torque_override(self):
        source = subprocess.check_output(['git','show',REV+':legged_gym/legged_gym/envs/minich/minich_config.py'], cwd=ROOT, text=True)
        base_source = subprocess.check_output(['git','show',REV+':legged_gym/legged_gym/envs/base/legged_robot_config.py'], cwd=ROOT, text=True)
        scope = {'__name__': 'legged_gym.envs.base.historical_config', '__package__': 'legged_gym.envs.base'}
        exec(compile(base_source, '<historical-base-config>', 'exec'), scope)
        source = source.replace('from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO', '')
        exec(compile(source, '<historical-config>', 'exec'), scope)
        def compare(old, new, prefix=''):
            for key, value in old.items():
                path = prefix + key
                if isinstance(value, dict):
                    compare(value, new[key], path+'.')
                else:
                    expected = -2e-4 if path == 'rewards.scales.torques' else value
                    self.assertEqual(new[key], expected, path)
        compare(class_to_dict(scope['MiniChRoughCfg']()), class_to_dict(MiniChRoughCfg()))
        compare(class_to_dict(scope['MiniChRoughCfgPPO']()), class_to_dict(MiniChRoughCfgPPO()))
        cfg = MiniChRoughCfg()
        self.assertFalse(cfg.env.stair_gait_clock)
        self.assertFalse(cfg.commands.long_stairs_forward_only)
        self.assertEqual(cfg.commands.stair_step_stand_probability, 0.)
        self.assertIsNone(cfg.terrain.long_stairs_height_levels)
        for key in ('stair_step','stair_step_overdue','stair_gait_force','stair_gait_height',
                    'stair_gait_velocity','stair_gait_pair_deficit','foot_vertical_force','com_foot_center'):
            self.assertEqual(getattr(cfg.rewards.scales, key), 0., key)

    def test_generated_default_terrain_equals_historical_generator(self):
        source = subprocess.check_output(['git','show',REV+':legged_gym/legged_gym/utils/terrain.py'], cwd=ROOT, text=True)
        scope = {}; exec(compile(source, '<historical-terrain>', 'exec'), scope)
        old_cfg = MiniChRoughCfg().terrain; new_cfg = MiniChRoughCfg().terrain
        for cfg in (old_cfg,new_cfg): cfg.mesh_type='heightfield'; cfg.border_size=0.
        np.random.seed(17); old = scope['Terrain'](old_cfg,20)
        np.random.seed(17); new = Terrain(new_cfg,20)
        np.testing.assert_array_equal(new.height_field_raw, old.height_field_raw)
        np.testing.assert_array_equal(new.env_origins, old.env_origins)
        r = LeggedRobot.__new__(LeggedRobot)
        r.cfg = SimpleNamespace(terrain=new_cfg); r.terrain = new
        r.num_envs = 4096; r.device = 'cpu'
        torch.manual_seed(17)
        expected_levels = torch.randint(0, new_cfg.max_init_terrain_level+1, (4096,))
        torch.manual_seed(17); r._get_env_origins()
        torch.testing.assert_close(r.terrain_levels, expected_levels)
        self.assertTrue((r.terrain_levels[r.long_stair_mask] > 2).any())

if __name__ == '__main__': unittest.main()

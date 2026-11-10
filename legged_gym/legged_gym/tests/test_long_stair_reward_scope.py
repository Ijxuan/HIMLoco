"""Historical reward scope through the actual post-physics callback."""
import isaacgym
import unittest
from unittest.mock import Mock
import torch
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.minich.minich_config import MiniChRoughCfg
from legged_gym.utils.step_retraction import StepRetraction
from legged_gym.utils.swing_peak_height import SwingPeakHeight


class TestHistoricalRewardScope(unittest.TestCase):
    def test_actual_callback_rewards_ordinary_terrain_and_gates_commands(self):
        r = LeggedRobot.__new__(LeggedRobot)
        r.num_envs = 10; r.num_bodies = 4; r.device = 'cpu'; r.cfg = MiniChRoughCfg()
        r.gym = Mock(); r.sim = None
        r.episode_length_buf = torch.zeros(10, dtype=torch.long); r.common_step_counter = 0
        r.root_states = torch.zeros(10,13); r.root_states[:,6] = 1
        r.base_quat = torch.zeros(10,4)
        r.base_lin_vel = torch.zeros(10,3); r.base_ang_vel = torch.zeros(10,3)
        r.gravity_vec = torch.tensor([[0.,0.,-1.]]).repeat(10,1)
        r.projected_gravity = torch.zeros(10,3)
        r.feet_indices = torch.arange(4); r.rigid_body_states = torch.zeros(10,4,13)
        r.rigid_body_states[:,:,2] = .02
        r.contact_forces = torch.zeros(10,4,3); r.contact_forces[:,:,2] = 10
        r.commands = torch.tensor([[.3,0.,0.]]*10)
        r.commands[1,0] = -.3  # Backward gait-group swing remains valid.
        r.commands[4,0] = 0.; r.commands[5,2] = .05
        # Neither reward is restricted to the dedicated stair columns.
        r.long_stair_mask = torch.arange(10) >= 8
        r.terrain_levels = torch.zeros(10, dtype=torch.long)
        r._post_physics_step_callback = Mock(); r._update_orientation_reference = Mock()
        r._sample_step_heights = lambda xy: torch.zeros(10,4)
        r.step_retraction = StepRetraction(10,4,'cpu',r.cfg.rewards,.02)
        r.swing_peak_height = SwingPeakHeight(10,4,'cpu',r.cfg.rewards,.02)
        # Stop after the real reward update; no simulator/reset state is needed.
        r.check_termination = Mock(side_effect=StopIteration)
        def update(n):
            for _ in range(n):
                with self.assertRaises(StopIteration):
                    r.post_physics_step()
        update(4)
        r.contact_forces.zero_(); r.rigid_body_states[:,:,2] = .07
        update(2)
        r.rigid_body_states[:,:,0] = -torch.sign(r.commands[:,0,None])*.01
        update(4)
        r.rigid_body_states[:,:,2] = .02; r.contact_forces[:,:,2] = 10
        update(2)
        self.assertTrue((r.step_retraction.stats['positive_sum'][[0,1,2,3,6,7,8,9]] > 0).all())
        self.assertEqual(r.step_retraction.stats['count'][[4,5]].sum().item(), 0.)
        self.assertTrue((r.swing_peak_height.stats['positive_sum'][:2] > 0).all())
        self.assertEqual(r.swing_peak_height.stats['count'][2:].sum().item(), 0.)

if __name__ == '__main__': unittest.main()

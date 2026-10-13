"""Behavioral checks for minimum-clearance and rolling lift-frequency reward."""
import types
import unittest
from legged_gym.envs.base.legged_robot import LeggedRobot
import torch
import test_air_time_continuous_cost as legacy_tests
import test_observation_history_reset as reset_tests


class TestTurnLiftReward(unittest.TestCase):
    def robot(self):
        r=legacy_tests.TestAirTimeContinuousCost().make_robot()
        r.device='cpu';r.num_envs=1
        r.cfg.terrain=types.SimpleNamespace(mesh_type='plane')
        cfg=r.cfg.rewards
        cfg.turn_lift_min_clearance=.01;cfg.turn_lift_min_frequency=.5
        cfg.turn_lift_window_s=4.;cfg.turn_lift_foot_radius=.02
        cfg.turn_lift_frequency_cost=1.
        r.feet_pos=torch.zeros(1,4,3);r.feet_pos[:,:,2]=.02
        r.contact_forces[:,:,2]=2.;r.last_contacts[:]=True
        r._init_turn_lift_buffers()
        return r

    def lift(self,r,height=.01,feet=(0,),steps=10):
        r.contact_forces[:,feet,2]=0.;r.feet_pos[:,feet,2]=.02+height
        for _ in range(steps):r._reward_feet_air_time()
        r.contact_forces[:,feet,2]=2.;r.feet_pos[:,feet,2]=.02
        return r._reward_feet_air_time().item()

    def test_static_turn_gets_dense_deficit_and_stand_does_not(self):
        r=self.robot();early=r._reward_feet_air_time().item()
        self.assertLess(early,0)
        for _ in range(220):last=r._reward_feet_air_time().item()
        self.assertAlmostEqual(last,-1.,places=5)
        r.commands.zero_();self.assertEqual(r._reward_feet_air_time().item(),0.)
        self.assertEqual(r.turn_lift_elapsed.item(),0.)
        self.assertFalse((r.turn_lift_ages<4).any())

    def test_one_cm_inclusive_but_low_lift_does_not_count(self):
        for height,valid in ((.009,False),(.01,True),(.02,True)):
            r=self.robot();rew=self.lift(r,height)
            self.assertEqual(bool((r.turn_lift_ages[0,0]<4).any()),valid)
            self.assertEqual(rew>0,valid)

    def test_bonus_is_capped_and_each_foot_must_participate(self):
        r=self.robot()
        self.assertGreater(self.lift(r),0)
        self.assertGreater(self.lift(r),0)
        self.assertLessEqual(self.lift(r),0)
        r.turn_lift_elapsed[:]=4.
        self.assertAlmostEqual(r._reward_feet_air_time().item(),-.75,places=5)

    def test_two_lifts_each_foot_satisfy_quota_then_expire(self):
        r=self.robot()
        for _ in range(2):self.lift(r,feet=(0,1,2,3))
        r.turn_lift_elapsed[:]=4.
        self.assertAlmostEqual(r._reward_feet_air_time().item(),0.,places=5)
        for _ in range(205):last=r._reward_feet_air_time().item()
        self.assertAlmostEqual(last,-1.,places=5)

    def test_long_hang_pays_before_landing_and_is_not_counted(self):
        r=self.robot();r.contact_forces[:,0,2]=0;r.feet_pos[:,0,2]=.05
        for _ in range(60):rew=r._reward_feet_air_time().item()
        self.assertLess(rew,-.4)
        self.assertFalse((r.turn_lift_ages<4).any())

    def test_pre_turn_airborne_and_interrupted_flights_do_not_count(self):
        r=self.robot();r.commands.zero_();r.contact_forces[:,0,2]=0
        r.feet_pos[:,0,2]=.04
        for _ in range(5):r._reward_feet_air_time()
        r.commands[:,2]=.25
        r.contact_forces[:,0,2]=2;r._reward_feet_air_time()
        self.assertFalse((r.turn_lift_ages<4).any())
        r=self.robot();r.contact_forces[:,0,2]=0;r.feet_pos[:,0,2]=.04
        for _ in range(5):r._reward_feet_air_time()
        r.commands.zero_();r._reward_feet_air_time()
        r.commands[:,2]=.25;r.contact_forces[:,0,2]=2;r._reward_feet_air_time()
        self.assertFalse((r.turn_lift_ages<4).any())

    def test_command_gate_inclusive_and_both_directions(self):
        for yaw in (-.2,.2):
            r=self.robot();r.commands[:,2]=yaw
            self.assertGreater(self.lift(r),0)
        for command in ((0,0,.19),(.1,0,.25),(0,0,0)):
            r=self.robot();r.commands[:,:3]=torch.tensor(command)
            self.assertEqual(self.lift(r),0)

    def test_reset_clears_only_selected_environment(self):
        r=reset_tests.TestObservationHistoryReset().make_robot()
        r.device='cpu';r.cfg.rewards=self.robot().cfg.rewards
        r.last_contacts=torch.ones(3,4,dtype=torch.bool)
        r._init_turn_lift_buffers()
        r.turn_lift_ages[:]=0.;r.turn_lift_peak[:]=.04
        r.turn_lift_eligible[:]=True;r.turn_lift_elapsed[:]=4.
        r.reset_idx(torch.tensor([1]))
        self.assertFalse((r.turn_lift_ages[1]<4).any())
        self.assertTrue((r.turn_lift_ages[[0,2]]==0).all())
        self.assertFalse(r.turn_lift_eligible[1].any())
        self.assertTrue(r.turn_lift_eligible[[0,2]].all())
        self.assertFalse(r.last_contacts[1].any())
        self.assertEqual(r.turn_lift_peak[1].sum().item(),0.)
        self.assertEqual(r.turn_lift_elapsed[1].item(),0.)

    def test_local_terrain_height_is_subtracted(self):
        r=self.robot();r.cfg.terrain=types.SimpleNamespace(mesh_type='heightfield')
        r.terrain=types.SimpleNamespace(cfg=types.SimpleNamespace(border_size=0.,horizontal_scale=.1,vertical_scale=.01))
        r.height_samples=torch.full((3,3),10.)
        r.feet_pos[:,:,2]=.13
        self.assertTrue(torch.allclose(r._get_feet_heights()-.02,torch.full((1,4),.01)))

if __name__=='__main__':unittest.main()

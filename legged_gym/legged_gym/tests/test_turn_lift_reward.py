"""Continuous turn-lift coverage, independent overdue cost and exact windows."""
import types
import unittest
from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.envs.minich.minich_config import MiniChRoughCfg
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from legged_gym.utils.phase_reward_logger import TurnLiftLogger
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
        r.feet_pos=torch.zeros(1,4,3);r.feet_pos[:,:,2]=.02
        r.contact_forces[:,:,2]=2.;r.last_contacts[:]=True
        r._init_turn_lift_buffers()
        return r

    def tick(self,r):
        r._update_turn_lift_state()
        return r._reward_turn_lift_frequency().item()

    def lift(self,r,height=.01,feet=(0,),steps=10):
        r.contact_forces[:,feet,2]=0.;r.feet_pos[:,feet,2]=.02+height
        for _ in range(steps):self.tick(r)
        r.contact_forces[:,feet,2]=2.;r.feet_pos[:,feet,2]=.02
        return self.tick(r)

    def test_static_turn_ramps_to_minus_one_and_stand_is_zero(self):
        r=self.robot()
        self.assertAlmostEqual(self.tick(r),-.005,places=6)
        for _ in range(199):last=self.tick(r)
        self.assertEqual(last,-1.)
        r.commands.zero_();self.assertEqual(self.tick(r),0.)
        self.assertEqual(r.turn_lift_elapsed.item(),0)
        self.assertFalse((r.turn_lift_ages<200).any())

    def test_one_cm_inclusive_and_only_completed_flights_count(self):
        for height,valid in ((.009,False),(.01,True),(.02,True)):
            r=self.robot();self.lift(r,height)
            self.assertEqual(bool((r.turn_lift_ages[0,0]<200).any()),valid)

    def test_per_foot_cap_no_compensation_and_score_levels(self):
        r=self.robot();r.turn_lift_elapsed[:]=200
        for n,expected in ((0,-1.),(1,0.),(2,1.)):
            r.turn_lift_ages[:]=200
            r.turn_lift_ages[:,:,:n]=0
            self.assertEqual(self.tick(r),expected)
        r.turn_lift_ages[:]=200;r.turn_lift_ages[:,0,:]=0
        self.assertEqual(self.tick(r),-.5) # one foot complete, three missing
        r.turn_lift_ages[:]=0;r.turn_lift_ages[:,3,:]=200
        self.assertEqual(self.tick(r),.5)

    def test_full_window_stays_positive_without_touchdown_then_expires_exactly(self):
        r=self.robot();r.turn_lift_elapsed[:]=200;r.turn_lift_ages[:]=198
        self.assertEqual(self.tick(r),1.) # age199 included
        self.assertEqual(self.tick(r),-1.) # age200 excluded, no accumulated float error

    def test_two_lifts_each_foot_fill_quota(self):
        r=self.robot()
        for _ in range(2):self.lift(r,feet=(0,1,2,3))
        r.turn_lift_elapsed[:]=200
        self.assertEqual(self.tick(r),1.)
        self.assertEqual(r._reward_turn_air_time_overdue().item(),0.)

    def test_frequency_above_threshold_is_not_worse_than_below(self):
        means=[]
        for period in (100,101,50,25):
            r=self.robot();scores=[]
            for step in range(1200):
                for foot in range(4):
                    phase=(step-(0 if foot in (0,3) else period//2))%period
                    air=phase<10
                    r.contact_forces[0,foot,2]=0 if air else 2
                    r.feet_pos[0,foot,2]=.04 if air else .02
                value=self.tick(r)
                if step>=400:scores.append(value)
                self.assertEqual(r._reward_turn_air_time_overdue().item(),0.)
            means.append(sum(scores)/len(scores))
        self.assertEqual(means[0],1.)
        self.assertLess(means[1],means[0])
        self.assertEqual(means[2:], [1.,1.])

    def test_long_hang_cost_survives_command_change_until_landing(self):
        r=self.robot();r.contact_forces[:,0,2]=0;r.feet_pos[:,0,2]=.05
        self.tick(r) # contact filter consumes first sample
        cost=0.
        for step in range(50):
            if step==30:r.commands.zero_()
            if step==49:r.contact_forces[:,0,2]=2.
            self.tick(r);cost+=r._reward_turn_air_time_overdue().item()
        self.assertAlmostEqual(cost,(1.-.3)/.05,places=4)
        self.assertEqual(r._reward_turn_lift_frequency().item(),0.)
        self.tick(r);self.assertEqual(r._reward_turn_air_time_overdue().item(),0.)
        self.assertFalse(r.feet_air_time_penalty_active.any())

    def test_stand_only_long_hang_is_not_penalized(self):
        r=self.robot();r.commands.zero_();r.contact_forces[:,0,2]=0
        for _ in range(60):
            self.assertEqual(self.tick(r),0.)
            self.assertEqual(r._reward_turn_air_time_overdue().item(),0.)

    def test_pre_turn_airborne_and_interrupted_flights_do_not_count(self):
        for interrupt in (False,True):
            r=self.robot()
            if not interrupt:r.commands.zero_()
            r.contact_forces[:,0,2]=0;r.feet_pos[:,0,2]=.04
            for _ in range(5):self.tick(r)
            if interrupt:r.commands.zero_();self.tick(r)
            r.commands[:,2]=.25;r.contact_forces[:,0,2]=2;self.tick(r)
            self.assertFalse((r.turn_lift_ages<200).any())

    def test_command_gate_boundaries(self):
        for yaw in (-.2,.2):
            r=self.robot();r.commands[:,2]=yaw;self.lift(r)
            self.assertTrue((r.turn_lift_ages<200).any())
        for command in ((0,0,.199),(.1,0,.25),(.08,.08,.25),(0,0,0)):
            r=self.robot();r.commands[:,:3]=torch.tensor(command);self.lift(r)
            self.assertFalse((r.turn_lift_ages<200).any())
            self.assertEqual(r._reward_turn_lift_frequency().item(),0.)

    def test_reward_order_disabled_terms_and_repeated_reads_do_not_tick_state(self):
        for names in (('turn_lift_frequency','turn_air_time_overdue'),
                      ('turn_air_time_overdue','turn_lift_frequency'),
                      ('turn_air_time_overdue',),('turn_lift_frequency',)):
            r=self.robot();r.cfg.rewards.only_positive_rewards=False
            r.reward_scales={n:getattr(MiniChRoughCfg.rewards.scales,n) for n in names}
            r._prepare_reward_function();r.rew_buf=torch.zeros(1)
            r.compute_reward()
            self.assertEqual(r.turn_lift_elapsed.item(),1)
            for _ in range(3):r._reward_turn_lift_frequency();r._reward_turn_air_time_overdue()
            self.assertEqual(r.turn_lift_elapsed.item(),1)
            expected=-.005*.25*.02 if 'turn_lift_frequency' in names else 0.
            self.assertAlmostEqual(r.rew_buf.item(),expected,places=7)
        self.assertEqual(MiniChRoughCfg.rewards.scales.feet_air_time,0.)
        self.assertEqual(LeggedRobotCfg.rewards.scales.turn_lift_frequency,0.)
        r=self.robot();r.reward_scales={'feet_air_time':1.}
        with self.assertRaises(ValueError):r._prepare_reward_function()

    def test_reset_clears_only_selected_environment(self):
        r=reset_tests.TestObservationHistoryReset().make_robot()
        r.device='cpu';r.dt=.02;r.cfg.rewards=self.robot().cfg.rewards
        r.last_contacts=torch.ones(3,4,dtype=torch.bool)
        r._init_turn_lift_buffers()
        r.turn_lift_ages[:]=0;r.turn_lift_peak[:]=.04
        r.turn_lift_eligible[:]=True;r.turn_lift_elapsed[:]=100;r.turn_air_steps[:]=25
        r.turn_air_overdue_cost[:]=.4;r.turn_lift_frequency_reward[:]=1.
        r.reset_idx(torch.tensor([1]))
        self.assertTrue((r.turn_lift_ages[1]==200).all())
        self.assertTrue((r.turn_lift_ages[[0,2]]==0).all())
        self.assertFalse(r.turn_lift_eligible[1].any())
        self.assertEqual(r.turn_air_steps[1].sum().item(),0)
        self.assertEqual(r.turn_air_steps[0].sum().item(),100)
        self.assertEqual(r.turn_air_overdue_cost[1].item(),0.)
        self.assertEqual(r.turn_lift_frequency_reward[1].item(),0.)
        self.assertFalse(r.last_contacts[1].any())

    def test_local_terrain_height(self):
        r=self.robot();r.cfg.terrain=types.SimpleNamespace(mesh_type='heightfield')
        r.terrain=types.SimpleNamespace(cfg=types.SimpleNamespace(border_size=0.,horizontal_scale=.1,vertical_scale=.01))
        r.height_samples=torch.full((3,3),10.);r.feet_pos[:,:,2]=.13
        self.assertTrue(torch.allclose(r._get_feet_heights()-.02,torch.full((1,4),.01)))

    def test_logger_rates_full_window_denominator_and_direction(self):
        log=TurnLiftLogger(.02,'cpu')
        commands=torch.tensor([[0,0,.25],[0,0,-.25]])
        active=torch.tensor([True,True]);mature=torch.tensor([False,True])
        events=torch.tensor([[True,False,False,False],[False,True,False,False]])
        passes=torch.tensor([[False]*4,[True,True,True,False]])
        overdue=torch.tensor([[0.]*4,[0.,0.,.02,0.]])
        log.update(commands,active,mature,events,passes,overdue)
        x=log.pop()
        self.assertEqual(x['TurnLift/turn/FL/lift_rate_hz'],25.)
        self.assertEqual(x['TurnLift/turn/FL/window_pass_fraction'],1.)
        self.assertEqual(x['TurnLift/turn/all_feet_window_pass_fraction'],0.)
        self.assertAlmostEqual(x['TurnLift/turn_negative/RL/overdue_fraction'],1.)
        self.assertNotIn('TurnLift/startup/all_feet_window_pass_fraction',x)
        self.assertEqual(log.pop()['TurnLift/turn/samples'],0.)


if __name__=='__main__':unittest.main()

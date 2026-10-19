"""Per-swing world displacement, command gating, debouncing and non-exploitable credit."""
import isaacgym
import unittest
import torch
from legged_gym.envs.minich.minich_config import MiniChRoughCfg
from legged_gym.utils.step_retraction import StepRetraction


class TestStepRetraction(unittest.TestCase):
    def setUp(self):
        self.r = StepRetraction(2, 4, 'cpu', MiniChRoughCfg.rewards, .02)
        self.feet = torch.zeros(2, 4, 3)
        self.cmd = torch.tensor([[.5, 0., 0.], [.5, 0., 0.]])
        self.world = self.cmd[:, :2].clone()
        self.support = torch.ones(2, 4, dtype=torch.bool)

    def update(self, n=1):
        for _ in range(n):self.r.update(self.feet, self.cmd, self.world, self.support)

    def arm(self):
        self.support.fill_(True);self.update(3)
        self.assertTrue(self.r.armed.all())

    def trigger(self):
        self.arm();self.support.zero_();self.update(2)
        self.assertTrue((self.r.stage == self.r.RETRACT).all())

    def test_all_terrain_swing_without_collision_or_height_gate(self):
        self.feet[..., 2] = 1.5
        self.trigger()
        self.assertEqual(self.r.best.sum(), 0)
        self.update(30)
        self.assertFalse((self.r.stage == self.r.COMPLETE).any())
        self.assertTrue((self.r.stage == self.r.TIMED_OUT).all())
        torch.testing.assert_close(self.r.stall_cost, torch.tensor([4.,4.]))

    def test_reward_budget_no_repeated_credit_and_second_swing(self):
        self.trigger();total=torch.zeros(2)
        for x in [-.01,0.,-.01,-.02,-.03,-.05]:
            self.feet[...,0]=x;self.update();total += self.r.progress_reward*.02
            if x==0.:self.assertEqual(self.r.progress_reward.sum(),0)
        torch.testing.assert_close(total,torch.tensor([4.,4.]))
        self.assertTrue((self.r.stage==self.r.COMPLETE).all())
        self.support.fill_(True);self.update(2)
        self.assertFalse(self.r.failed_landing.any())
        self.arm();self.support.zero_();self.update(2)
        self.assertTrue(self.r.started.all())
        self.assertEqual(self.r.best.sum(),0)

    def test_yaw_zero_gate_and_lateral_only_disabled(self):
        for vx,vy,wz in [(0.,0.,0.),(0.,0.,.5),(.5,0.,.05),(.5,0.,-.2),(0.,.5,0.)]:
            self.setUp();self.cmd[:]=torch.tensor([vx,vy,wz]);self.world[:]=self.cmd[:,:2]
            self.update(4);self.support.zero_();self.update(5)
            self.assertEqual(self.r.stage.sum(),0)
            self.assertEqual(self.r.progress_reward.sum(),0)
            self.assertEqual(self.r.stall_cost.sum(),0)

    def test_world_direction_not_body_x_and_backward_command(self):
        for vx,world in [(.5,[0.,.5]),(-.5,[-.5,0.])]:
            self.setUp();self.cmd[:,0]=vx;self.world[:]=torch.tensor(world)
            self.trigger()
            self.feet[...,:2]=-self.world[:,None,:]/self.world.norm(dim=1)[:,None,None]*.03
            self.update()
            self.assertTrue(self.r.completed.all())
            torch.testing.assert_close(self.r.progress_reward*.02,torch.tensor([4.,4.]))

    def test_body_yaw_cannot_cancel_or_rotate_latched_axis(self):
        self.trigger()
        self.world[:]=torch.tensor([0.,.5])  # Same input command; body yaw changed 90 deg.
        self.update()
        self.assertTrue((self.r.stage==self.r.RETRACT).all())
        self.feet[...,1]=-.03;self.update()
        self.assertEqual(self.r.progress_reward.sum(),0)
        self.feet[...,0]=-.03;self.update()
        self.assertTrue(self.r.completed.all())

    def test_positive_world_motion_cannot_count_as_relative_body_retraction(self):
        self.trigger();self.feet[...,0]=.02;self.update()
        self.assertEqual(self.r.progress_reward.sum(),0)
        self.assertEqual(self.r.best.sum(),0)
        self.assertTrue((self.r.displacement<0).all())

    def test_continuous_lag_and_no_threshold_escape(self):
        self.trigger()
        # Same elapsed time, different sub-mm displacement: smooth penalty change.
        costs=[]
        for x in [0.,-.00019,-.00021,-.001]:
            self.setUp();self.trigger();self.feet[...,0]=x;self.update()
            costs.append(self.r.stall_cost[0].item())
        self.assertTrue(all(a>b for a,b in zip(costs,costs[1:])))
        self.assertLess(abs(costs[1]-costs[2]),.01)

    def test_failed_early_landing_penalty_and_no_stance_drag_credit(self):
        self.trigger();self.support.fill_(True);self.feet[...,0]=-.03
        self.update(2)
        self.assertTrue(self.r.failed_landing.all())
        self.assertEqual(self.r.progress_reward.sum(),0)
        self.assertTrue((self.r.stall_cost>=60).all())
        self.assertEqual(self.r.stage.sum(),0)

    def test_debounce_origin_preserves_first_air_displacement(self):
        self.arm();self.support.zero_();self.feet[...,0]=-.01;self.update()
        self.assertEqual(self.r.progress_reward.sum(),0)
        self.feet[...,0]=-.02;self.update()
        torch.testing.assert_close(self.r.best,torch.full((2,4),.02))
        self.assertTrue(self.r.started.all())
        self.support.fill_(True);self.update()  # One-frame touchdown cannot rearm.
        self.support.zero_();self.update()
        self.assertFalse(self.r.started.any())

    def test_command_cancel_partial_reset_no_midair_rearm(self):
        self.trigger();self.cmd[:,0]=-.5;self.world[:,0]=-.5;self.update()
        self.assertEqual(self.r.stage.sum(),0)
        self.update(3)
        self.assertEqual(self.r.stage.sum(),0)
        self.setUp();self.trigger();self.r.reset(torch.tensor([0]))
        self.assertFalse(self.r.initialized[0].any())
        self.assertTrue((self.r.stage[1]==self.r.RETRACT).all())

    def test_late_completion_and_initial_airborne_no_false_swing(self):
        self.support.zero_();self.update(10)
        self.assertEqual(self.r.stage.sum(),0)
        self.trigger();self.update(25)
        self.assertTrue((self.r.stage==self.r.TIMED_OUT).all())
        self.feet[...,0]=-.03;self.update()
        self.assertTrue(self.r.completed.all())
        self.assertEqual(self.r.stall_cost.sum(),0)

    def test_contact_chatter_and_independent_feet(self):
        self.arm();self.support[:,0]=False;self.update()
        self.support[:,0]=True;self.update()
        self.assertEqual(self.r.stage.sum(),0)
        self.support[:,1]=False;self.update(2)
        self.assertTrue((self.r.stage[:,1]==self.r.RETRACT).all())
        self.assertEqual(self.r.stage[:,[0,2,3]].sum(),0)

if __name__=='__main__':unittest.main()

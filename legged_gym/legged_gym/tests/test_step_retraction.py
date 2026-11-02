"""Per-swing world displacement, command gating, debouncing and non-exploitable credit."""
import isaacgym
import unittest
import torch
from legged_gym.envs.minich.minich_config import MiniChRoughCfg
from legged_gym.utils.step_retraction import StepRetraction


class TestStepRetraction(unittest.TestCase):
    def setUp(self):
        # Pin the original nonzero-bound regression cases independently of defaults.
        class LegacyBounds(MiniChRoughCfg.rewards):
            step_retraction_distance = .005
        self.r = StepRetraction(2, 4, 'cpu', LegacyBounds, .02)
        self.feet = torch.zeros(2, 4, 3)
        self.base = torch.zeros(2, 2)
        self.cmd = torch.tensor([[.5, 0., 0.], [.5, 0., 0.]])
        self.world = self.cmd[:, :2].clone()
        self.support = torch.ones(2, 4, dtype=torch.bool)
        self.enabled_envs = None

    def update(self, n=1):
        for _ in range(n):self.r.update(self.feet, self.cmd, self.world, self.support, self.base,
                                      enabled_envs=self.enabled_envs)

    def test_environment_mask_excludes_positive_and_negative_scores(self):
        self.enabled_envs = torch.tensor([True, False])
        for displacement in (-.01, 0.):
            self.support.fill_(True);self.update(3)
            self.support.zero_();self.update(2)
            self.feet[..., 0] += displacement;self.update()
            self.support.fill_(True);self.update(2)
            self.assertEqual(self.r.progress_reward[1].item(), 0.)
            self.assertEqual(self.r.stall_cost[1].item(), 0.)
        self.assertTrue((self.r.stats['count'][0] == 2).all())
        self.assertEqual(self.r.stats['count'][1].sum().item(), 0.)
        self.assertTrue((self.r.stats['positive_sum'][0] > 0).all())
        self.assertTrue((self.r.stats['negative_sum'][0] < 0).all())
        self.assertEqual(self.r.stats['enabled_seconds'][1].sum().item(), 0.)

    def test_disabling_environment_cancels_inflight_event(self):
        self.trigger()
        self.enabled_envs = torch.tensor([True, False])
        self.feet[..., 0] = -.01;self.update()
        self.support.fill_(True);self.update(2)
        self.assertGreater(self.r.progress_reward[0].item(), 0.)
        self.assertEqual(self.r.progress_reward[1].item(), 0.)
        self.assertEqual(self.r.stall_cost[1].item(), 0.)
        self.assertTrue((self.r.stage[1] == self.r.IDLE).all())
        self.assertTrue((self.r.stats['cancelled'][1] == 1).all())
        self.assertEqual(self.r.stats['count'][1].sum().item(), 0.)

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
        for x in [-.005,0.,-.005,-.01]:
            self.feet[...,0]=x;self.update();total += self.r.progress_reward*.02
        self.assertEqual(total.sum(),0)
        self.support.fill_(True);self.update(2)
        torch.testing.assert_close(self.r.progress_reward*.02,torch.tensor([4.,4.]))
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
            self.feet[...,:2]=-self.world[:,None,:]/self.world.norm(dim=1)[:,None,None]*.01
            self.update()
            self.assertTrue(self.r.completed.all())
            self.assertEqual(self.r.progress_reward.sum(),0)
            self.support.fill_(True);self.update(2)
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
        torch.testing.assert_close(self.r.progress_reward*.02,torch.full((2,),-4.))
        self.assertTrue((self.r.stall_cost>=60).all())
        self.assertEqual(self.r.stage.sum(),0)

    def test_debounce_origin_preserves_first_air_displacement(self):
        self.arm();self.support.zero_();self.feet[...,0]=-.01;self.update()
        self.assertEqual(self.r.progress_reward.sum(),0)
        self.feet[...,0]=-.02;self.update()
        torch.testing.assert_close(self.r.best,torch.full((2,4),.005))
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
        self.feet[...,0]=-.01;self.update()
        self.assertTrue(self.r.completed.all())
        self.assertEqual(self.r.stall_cost.sum(),0)

    def test_contact_chatter_and_independent_feet(self):
        self.arm();self.support[:,0]=False;self.update()
        self.support[:,0]=True;self.update()
        self.assertEqual(self.r.stage.sum(),0)
        self.support[:,1]=False;self.update(2)
        self.assertTrue((self.r.stage[:,1]==self.r.RETRACT).all())
        self.assertEqual(self.r.stage[:,[0,2,3]].sum(),0)

    def test_band_boundaries_and_landing_penalty(self):
        for distance, valid in [(0.,False),(.004,False),(.005,True),(.01,True),(.015,True),(.016,False),(.03,False)]:
            with self.subTest(distance=distance):
                self.setUp();self.trigger();self.feet[...,0]=-distance;self.update(20)
                if valid:self.assertEqual(self.r.stall_cost.sum(),0)
                else:self.assertGreater(self.r.stall_cost.sum(),0)
                self.support.fill_(True);self.update(2)
                self.assertEqual(bool(self.r.failed_landing.all()),not valid)

    def test_overshoot_after_completion_survives_forward_recovery(self):
        self.trigger();self.feet[...,0]=-.01;self.update()
        self.assertTrue(self.r.completed.all())
        self.feet[...,0]=-.04;self.update()
        self.assertGreater(self.r.over_cost.sum(),0)
        self.feet[...,0]=.1;self.update()
        self.assertGreater(self.r.stall_cost.sum(),0)
        self.support.fill_(True);self.update(2)
        self.assertTrue(self.r.failed_landing.all())

    def test_valid_retraction_can_then_swing_forward_without_under_penalty(self):
        self.trigger();self.feet[...,0]=-.01;self.update()
        self.feet[...,0]=.1;self.update(20)
        self.assertEqual(self.r.stall_cost.sum(),0)
        self.support.fill_(True);self.update(2)
        self.assertFalse(self.r.failed_landing.any())

    def test_body_retreat_alone_never_counts_in_any_command_direction(self):
        for vx,d in [(.5,[1.,0.]),(-.5,[-1.,0.]),(.5,[0.,1.])]:
            self.setUp();self.cmd[:,0]=vx;direction=torch.tensor(d)
            self.world[:]=direction*.5;self.trigger()
            for distance in [.01,.02,.03,.04]:
                self.base[:]=-distance*direction
                self.feet[...,:2]=self.base[:,None,:];self.update()
                self.assertEqual(self.r.best.sum(),0)
                self.assertFalse(self.r.completed.any())
            self.assertGreater(self.r.stall_cost.sum(),0)

    def test_forward_body_motion_cannot_refund_reverse_travel(self):
        self.trigger();self.base[:,0]=-.02;self.feet[...,0]=-.02;self.update()
        self.base.zero_();self.update()  # Foot stays back while trunk recovers.
        self.assertEqual(self.r.best.sum(),0)
        torch.testing.assert_close(self.r.body_backward,torch.full((2,4),.02))
        self.assertFalse(self.r.completed.any())

    def test_body_compensation_includes_debounce_and_resets_next_swing(self):
        self.arm();self.support.zero_();self.base[:,0]=-.02;self.feet[...,0]=-.02;self.update()
        self.base[:,0]=-.025;self.feet[...,0]=-.025;self.update()
        self.assertEqual(self.r.best.sum(),0)
        torch.testing.assert_close(self.r.body_backward,torch.full((2,4),.025))
        self.support.fill_(True);self.update(2);self.arm()
        self.support.zero_();self.update(2)
        self.assertEqual(self.r.body_backward.sum(),0)
        self.assertEqual(self.r.world_best.sum(),0)
        self.r.reset(torch.tensor([0]))
        self.assertEqual(self.r.previous_base_xy[0].sum(),0)

    def test_body_forward_does_not_create_world_retraction_credit(self):
        self.trigger();self.base[:,0]=.05;self.feet[...,0]=.02;self.update()
        self.assertEqual(self.r.best.sum(),0)
        self.feet[...,0]=-.01;self.update()
        self.assertTrue(self.r.completed.all())
        self.assertEqual(self.r.stall_cost.sum(),0)

    def test_body_retreat_cannot_hide_world_overshoot(self):
        self.trigger();self.base[:,0]=-.02;self.feet[...,0]=-.04;self.update()
        torch.testing.assert_close(self.r.displacement,torch.full((2,4),.02))
        self.assertTrue(self.r.completed.all())  # Minimum reached, not a band-valid landing.
        self.assertGreater(self.r.over_cost.sum(),0)

    def test_landing_peak_bonus_and_overshoot_not_refunded(self):
        for distance, expected in [(.004,-.04),(.005,0.),(.0075,.75),(.01,1.),(.0125,.75),(.015,0.),(.02,-1.)]:
            with self.subTest(distance=distance):
                self.setUp();self.trigger()
                self.feet[...,0]=-distance;self.update(5)
                self.assertEqual(self.r.progress_reward.sum(),0)
                self.feet[...,0]=.1;self.update()
                self.support.fill_(True);self.update(2)
                torch.testing.assert_close(self.r.progress_reward*.02,torch.full((2,),4*expected),atol=1e-5,rtol=1e-5)
                torch.testing.assert_close(self.r.stats['progress_sum'],torch.full((2,4),expected),atol=1e-5,rtol=1e-5)
                self.update(5);self.assertEqual(self.r.progress_reward.sum(),0)

    def test_timeout_settles_once_even_after_optimal_retraction(self):
        for distance in [0.,.01,.02]:
            self.setUp();self.trigger();self.feet[...,0]=-distance
            total=torch.zeros(2)
            for _ in range(30):
                self.update();total+=self.r.progress_reward*.02
            torch.testing.assert_close(total,torch.full((2,),-4.))
            self.assertTrue((self.r.stats['count']==1).all())
            self.assertTrue((self.r.stats['timeout']==1).all())
            self.feet[...,0]=-.01;self.support.fill_(True);self.update(2)
            self.assertEqual(self.r.progress_reward.sum(),0)
            self.assertTrue((self.r.stats['count']==1).all())

    def test_world_overshoot_cannot_receive_positive_event_score(self):
        self.trigger();self.base[:,0]=-.01;self.feet[...,0]=-.02;self.update(4)
        self.support.fill_(True);self.update(2)
        torch.testing.assert_close(self.r.progress_reward*.02,torch.full((2,),-4.))
        torch.testing.assert_close(self.r.stats['positive_sum'],torch.zeros(2,4))
        torch.testing.assert_close(self.r.stats['score_sum'],self.r.stats['positive_sum']+self.r.stats['negative_sum'])

    def test_landing_bonus_effective_config(self):
        self.assertGreater(MiniChRoughCfg.rewards.scales.step_retraction,0.)
        self.assertEqual(MiniChRoughCfg.rewards.scales.step_retraction_stall,0.)

    def test_zero_lower_bound_landing_scores_and_finite_state(self):
        self.assertEqual(MiniChRoughCfg.rewards.step_retraction_distance, 0.)
        for distance, expected in [(0., 0.), (.005, .75), (.01, 1.), (.015, 0.), (.02, -1.)]:
            with self.subTest(distance=distance):
                self.setUp()
                self.r = StepRetraction(2, 4, 'cpu', MiniChRoughCfg.rewards, .02)
                self.arm(); self.support.zero_(); self.update(2)
                self.feet[..., 0] = -distance; self.update(5)
                self.assertEqual(self.r.progress_reward.sum(), 0.)
                self.support.fill_(True); self.update(2)
                torch.testing.assert_close(self.r.progress_reward * .02, torch.full((2,), 4 * expected), atol=1e-5, rtol=1e-5)
                for value in vars(self.r).values():
                    if isinstance(value, torch.Tensor):
                        self.assertTrue(torch.isfinite(value).all())
                self.update(5)
                self.assertEqual(self.r.progress_reward.sum(), 0.)

    def test_zero_lower_bound_timeout_remains_negative_once(self):
        self.r = StepRetraction(2, 4, 'cpu', MiniChRoughCfg.rewards, .02)
        self.arm(); self.support.zero_(); self.update(2)
        self.feet[..., 0] = -.01
        total = torch.zeros(2)
        for _ in range(30):
            self.update(); total += self.r.progress_reward * .02
        torch.testing.assert_close(total, torch.full((2,), -4.))
        self.support.fill_(True); self.update(2)
        self.assertEqual(self.r.progress_reward.sum(), 0.)
        self.assertTrue((self.r.stats['count'] == 1).all())

if __name__=='__main__':unittest.main()

import isaacgym
import unittest
import types
import torch
from legged_gym.envs.minich.minich_config import MiniChRoughCfg
from legged_gym.utils.step_retraction import StepRetraction


class TestStepRetraction(unittest.TestCase):
    def setUp(self):
        self.r = StepRetraction(2, 4, 'cpu', MiniChRoughCfg.rewards, .02)
        self.feet = torch.zeros(2, 4, 3); self.feet[..., 2] = .05
        self.force = torch.zeros_like(self.feet); self.force[..., 0] = -10
        self.c = types.SimpleNamespace(active=torch.ones(2,4,dtype=torch.bool),
            target_height=torch.full((2,4),.11), edge=torch.zeros(2,4,2),
            direction=torch.zeros(2,4,2), per_foot_cost=torch.ones(2,4),cost=torch.ones(2))
        self.c.edge[..., 0] = .02; self.c.direction[..., 0] = 1.

    def update(self):
        self.c.per_foot_cost.fill_(1)
        self.r.update(self.feet,self.force,self.c)

    def trigger(self):
        self.update();self.update()
        self.assertTrue((self.r.stage == self.r.RETRACT).all())

    def test_stationary_penalty_and_no_trigger_sample_penalty(self):
        self.trigger()
        self.assertEqual(self.r.stall_cost.sum(),0)
        self.assertEqual(self.c.cost.sum(),0)
        self.update()
        torch.testing.assert_close(self.r.stall_cost,torch.tensor([4.,4.]))
        self.assertEqual(self.r.progress_reward.sum(),0)

    def test_displacement_budget_and_no_oscillation_credit(self):
        self.trigger()
        total=torch.zeros(2)
        for x in [-.01,0.,-.01,-.02,-.03,-.05]:
            self.feet[...,0]=x;self.update()
            total += self.r.progress_reward*.02
            if x==0.:self.assertEqual(self.r.progress_reward.sum(),0)
        torch.testing.assert_close(total,torch.tensor([4.,4.]))
        self.assertTrue((self.r.stage == self.r.COMPLETE).all())
        self.assertTrue((self.c.cost == 4).all())
        self.feet[...,0]=0.;self.update();self.update()
        self.assertEqual(self.r.progress_reward.sum(),0)

    def test_sideways_target_direction(self):
        self.c.direction.zero_();self.c.direction[...,1]=1
        self.c.edge.zero_();self.c.edge[...,1]=.02
        self.force.zero_();self.force[...,1]=-10
        self.trigger()
        self.feet[...,0]=-.03;self.update()
        self.assertEqual(self.r.progress_reward.sum(),0)
        self.feet[...,1]=-.03;self.update()
        torch.testing.assert_close(self.r.progress_reward*.02,torch.tensor([4.,4.]))

    def test_timeout_does_not_rearm_and_keeps_cost(self):
        self.trigger()
        for _ in range(30):self.update()
        self.assertTrue((self.r.stage == self.r.TIMED_OUT).all())
        self.assertTrue((self.r.stall_cost == 4).all())
        self.assertTrue((self.c.cost == 4).all())
        self.feet[...,0]=-.03;self.update()
        self.assertEqual(self.r.progress_reward.sum(),0)
        self.c.active.zero_();self.update()
        self.assertEqual(self.r.stage.sum(),0)
        self.assertEqual(self.r.stall_cost.sum(),0)

    def test_reset_and_already_high(self):
        self.feet[...,2]=.14;self.update();self.update()
        self.assertEqual(self.r.stage.sum(),0)
        self.feet[...,2]=.05;self.trigger()
        self.r.reset(torch.tensor([0]))
        self.assertEqual(self.r.stage[0].sum(),0)
        self.assertTrue((self.r.stage[1]==self.r.RETRACT).all())

    def test_force_confirmation_and_contact_direction(self):
        self.force[...,0]=10;self.update();self.update()
        self.assertEqual(self.r.stage.sum(),0)
        self.force[...,0]=-10;self.update()
        self.force.zero_();self.update()
        self.assertEqual(self.r.stage.sum(),0)
        self.assertEqual(self.r.contact_time.sum(),0)

if __name__=='__main__':unittest.main()

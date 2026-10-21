import isaacgym
import unittest
import torch
from legged_gym.envs.minich.minich_config import MiniChRoughCfg
from legged_gym.utils.swing_peak_height import SwingPeakHeight, peak_height_score
import test_command_stages


class TestPeakHeight(unittest.TestCase):
    def setUp(self):
        self.r = SwingPeakHeight(2,4,'cpu',MiniChRoughCfg.rewards,.02)
        self.h = torch.zeros(2,4)
        self.contact = torch.ones(2,4,dtype=torch.bool)
        self.cmd = torch.tensor([[.3,0.,0.],[-.3,0.,0.]])
        self.enabled = torch.tensor([True,False])

    def step(self,n=1):
        total=torch.zeros(2)
        for _ in range(n):
            self.r.update(self.h,self.contact,self.cmd,self.enabled)
            total += self.r.score*.02
        return total

    def swing(self,height):
        self.step(3);self.contact.zero_();self.h[:]=height
        credit=self.step(5)
        self.contact.fill_(True);credit+=self.step(2)
        return credit

    def test_curve_sign_peak_continuity(self):
        h=torch.tensor([0.,.0125,.025,.0375,.05,.065,.08,.095,.11])
        expected=torch.tensor([-1.,-.25,0.,.75,1.,.75,0.,-.25,-1.])
        torch.testing.assert_close(peak_height_score(h),expected,atol=1e-6,rtol=1e-6)
        for edge in [.025,.05,.08]:
            scores=peak_height_score(torch.tensor([edge-1e-7,edge,edge+1e-7]))
            self.assertLess((scores.max()-scores.min()).item(),1e-4)

    def test_once_per_swing_and_only_selected_envs(self):
        total=self.swing(.05)
        torch.testing.assert_close(total,torch.tensor([4.,0.]))
        self.assertEqual(self.step(20).sum(),0)
        self.assertEqual(self.r.stats['count'][0].sum(),4)
        self.assertEqual(self.r.stats['count'][1].sum(),0)

    def test_peak_not_landing_height(self):
        self.step(3);self.contact.zero_();self.h[:]=.05;self.step(4)
        self.h.zero_();self.step();self.contact.fill_(True)
        torch.testing.assert_close(self.step(2),torch.tensor([4.,0.]))

    def test_overshoot_not_erased_by_lowering(self):
        self.step(3);self.contact.zero_();self.h[:]=.11;self.step(4)
        self.h[:]=.05;self.step();self.contact.fill_(True)
        torch.testing.assert_close(self.step(2),torch.tensor([-4.,0.]))

    def test_underheight_negative(self):
        torch.testing.assert_close(self.swing(.0125),torch.tensor([-1.,0.]))

    def test_timeout_once_no_hanging_credit(self):
        self.step(3);self.contact.zero_();self.h[:]=.05
        torch.testing.assert_close(self.step(30),torch.tensor([-4.,0.]))
        self.assertEqual(self.step(50).sum(),0)
        self.assertEqual(self.r.stats['timeout'][0].sum(),4)

    def test_no_initial_airborne_or_stance_credit(self):
        self.contact.zero_();self.h[:]=.05
        self.assertEqual(self.step(50).sum(),0)
        self.contact.fill_(True)
        self.assertEqual(self.step(50).sum(),0)

    def test_short_dropout_cannot_earn_positive_reward(self):
        self.step(3);self.contact.zero_();self.h[:]=.05;self.step()
        self.contact.fill_(True)
        self.assertEqual(self.step(2).sum(),0)

    def test_cancel_and_reset_do_not_carry_old_peak(self):
        self.step(3);self.contact.zero_();self.h[:]=.1;self.step(4)
        self.enabled.zero_();self.step()
        self.assertFalse(self.r.active.any())
        self.enabled[0]=True
        self.assertEqual(self.step(30).sum(),0)
        self.r.reset(torch.tensor([0]))
        self.assertEqual(self.r.peak.sum(),0)

    def test_reverse_command_cancels_old_cycle(self):
        self.step(3);self.contact.zero_();self.h[:]=.05;self.step(4)
        self.cmd*=-1;self.step()
        self.assertFalse(self.r.active.any())
        self.assertEqual(self.step(30).sum(),0)

    def test_duration_invariant_event_budget(self):
        for n in [4,10,20]:
            self.setUp();self.step(3);self.contact.zero_();self.h[:]=.05
            total=self.step(n);self.contact.fill_(True);total+=self.step(2)
            torch.testing.assert_close(total,torch.tensor([4.,0.]))

    def test_independent_feet(self):
        self.step(3);self.contact[:,0]=False;self.h[:,0]=.05;self.step(5)
        self.contact.fill_(True)
        torch.testing.assert_close(self.step(2),torch.tensor([1.,0.]))


class TestGaitSampling(unittest.TestCase):
    def robot(self):
        r=test_command_stages.TestCommandStages().make_robot(n=100)
        r.cfg.commands.gait_training_fraction=.2
        r.cfg.commands.gait_training_speed_range=[.2,.5]
        return r

    def test_low_speed_group_no_turns_even_with_high_curriculum(self):
        r=self.robot();r.command_ranges['lin_vel_x']=[-3.,3.]
        r.cfg.commands.in_place_turn_probability=1.
        r.cfg.commands.turn_to_stand_probability=0.
        for _ in range(20):
            r._resample_commands(torch.arange(100))
            self.assertTrue((r.commands[:20,0].abs()>=.2).all())
            self.assertTrue((r.commands[:20,0].abs()<=.5).all())
            self.assertFalse(r.commands[:20,1:3].any())
            self.assertFalse(r.in_place_turn_buf[:20].any())
            self.assertTrue(r.in_place_turn_buf[20:].all())
            self.assertTrue((r.command_stage[:20]==r.COMMAND_NORMAL).all())
        r._post_physics_step_callback()
        self.assertFalse(r.commands[:20,2].any())
        self.assertTrue((r.commands[20:,2].abs()>0).any())

    def test_explicit_zero_range_and_subset(self):
        r=self.robot();r.command_ranges['lin_vel_x']=[0.,0.]
        r.commands.fill_(.123)
        r._resample_commands(torch.tensor([3,7]))
        self.assertFalse(r.commands[[3,7],:3].any())
        self.assertTrue((r.commands[0]==.123).all())

    def test_exact_minimum_speed_not_removed_by_deadband(self):
        r=self.robot();r.cfg.commands.gait_training_speed_range=[.2,.2]
        r._resample_commands(torch.arange(100))
        torch.testing.assert_close(r.commands[:20,0].abs(),torch.full((20,),.2))

    def test_curriculum_cannot_expand_gait_group(self):
        r=self.robot();before=list(r.command_ranges['lin_vel_x'])
        r.update_command_curriculum(torch.arange(100))
        self.assertEqual(r.command_ranges['lin_vel_x'],before)

if __name__=='__main__':unittest.main()

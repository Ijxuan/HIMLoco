import isaacgym
import unittest
import torch
from legged_gym.envs.minich.minich_config import MiniChRoughCfg
from rsl_rl.utils.gait_metrics import summarize_gait_episode_metrics
import test_step_retraction
import test_swing_peak_height


class TestGaitMetrics(unittest.TestCase):
    def test_event_weighted_not_mean_of_ratios_and_no_empty_nan(self):
        prefix='gait_height/FL_foot/'
        infos=[{prefix+'count':torch.tensor(1.),prefix+'peak_sum_cm':torch.tensor(6.),prefix+'in_band':1.},
               {prefix+'count':9.,prefix+'peak_sum_cm':36.,prefix+'in_band':0.},
               {'rew_tracking_lin_vel':1.}]
        m=summarize_gait_episode_metrics(infos);p='Gait/height/FL_foot/'
        self.assertAlmostEqual(m[p+'mean_peak_cm'],4.2)
        self.assertAlmostEqual(m[p+'in_band_rate'],.1)
        m=summarize_gait_episode_metrics([{prefix+'count':0.,prefix+'started':0.,prefix+'enabled_seconds':0.}])
        self.assertNotIn(p+'mean_peak_cm',m)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in m.values()))

    def test_retraction_costs_reconstruct_reward_and_peaks_not_capped(self):
        t=test_step_retraction.TestStepRetraction();t.setUp();t.trigger()
        total=(t.r.stats['lag_integral']+t.r.stats['over_integral']+t.r.stats['landing_integral']).sum(-1).clone()
        for x in [-.01,-.02,-.04]:
            t.feet[...,0]=x;t.update();total+=t.r.stall_cost*.02
        t.support.fill_(True)
        for _ in range(2):t.update();total+=t.r.stall_cost*.02
        s=t.r.stats
        costs=(s['lag_integral']+s['over_integral']+s['landing_integral']).sum(-1)
        torch.testing.assert_close(costs,total)
        self.assertTrue((s['count']==1).all());self.assertTrue((s['over']==1).all())
        self.assertFalse(s['in_band'].any())
        torch.testing.assert_close(s['effective_peak_sum_cm'],torch.full((2,4),4.))
        torch.testing.assert_close(s['world_peak_sum_cm'],torch.full((2,4),4.))
        before={k:v.clone() for k,v in s.items()}
        t.r.reset(torch.tensor([0]))
        for k,v in s.items():torch.testing.assert_close(v,before[k])

    def test_pure_body_retreat_logged_not_success(self):
        t=test_step_retraction.TestStepRetraction();t.setUp();t.trigger()
        t.base[:,0]=-.02;t.feet[...,0]=-.02;t.update()
        t.support.fill_(True);t.update(2)
        s=t.r.stats
        self.assertTrue((s['under']==1).all());self.assertFalse(s['in_band'].any())
        self.assertFalse(s['effective_peak_sum_cm'].any())
        torch.testing.assert_close(s['body_backward_sum_cm'],torch.full((2,4),2.))

    def test_timeout_counts_once_and_cancel_on_reset(self):
        t=test_step_retraction.TestStepRetraction();t.setUp();t.trigger();t.update(50)
        self.assertTrue((t.r.stats['timeout']==1).all())
        t.r.reset(torch.tensor([0]));t.r.reset(torch.tensor([0]))
        self.assertTrue((t.r.stats['cancelled'][0]==1).all())
        self.assertFalse(t.r.stats['cancelled'][1].any())

    def test_height_score_splits_and_valid_landing(self):
        t=test_swing_peak_height.TestPeakHeight();t.setUp();total=t.swing(.065)
        s=t.r.stats
        torch.testing.assert_close(s['score_sum'].sum(-1),total)
        torch.testing.assert_close(s['positive_sum']+s['negative_sum'],s['score_sum'])
        self.assertTrue((s['valid_landing'][0]==1).all())
        t.setUp();t.step(3);t.contact.zero_();t.h[:]=.065;t.step(30)
        self.assertTrue((t.r.stats['in_band'][0]==1).all())
        self.assertFalse(t.r.stats['valid_landing'].any())
        self.assertTrue((t.r.stats['timeout'][0]==1).all())

    def test_lateral_zero_after_update_other_envs_unchanged(self):
        t=test_swing_peak_height.TestGaitSampling();r=t.robot()
        r._resample_commands(torch.arange(100));r.commands[:,1]=.4
        r.cfg.commands.heading_command=False
        r._post_physics_step_callback()
        self.assertFalse(r.commands[:20,1].any())
        torch.testing.assert_close(r.commands[20:,1],torch.full((80,),.4))

    def test_group_and_reward_denominators(self):
        p='gait_retraction/FL_foot/';q='retraction_other/FL_foot/'
        m=summarize_gait_episode_metrics([{p+'count':2.,p+'in_band':1.,p+'enabled_seconds':4.,p+'weighted_score_sum':-.8,p+'lag_reward_sum':-.4,q+'count':10.,q+'in_band':10.}])
        self.assertAlmostEqual(m['Gait/retraction/FL_foot/in_band_rate'],.5)
        self.assertAlmostEqual(m['Gait/retraction/FL_foot/reward_per_enabled_second'],-.2,places=6)
        self.assertAlmostEqual(m['Gait/retraction/FL_foot/lag_reward_per_enabled_second'],-.1,places=6)
        self.assertAlmostEqual(m['Gait/retraction_other/FL_foot/in_band_rate'],1.)

if __name__=='__main__':unittest.main()

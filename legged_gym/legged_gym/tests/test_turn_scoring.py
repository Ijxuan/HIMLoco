"""Independent scoring tests use synthetic physical outcomes, without Isaac Gym."""
import copy
import importlib.util
from pathlib import Path
import unittest
import numpy as np
spec=importlib.util.spec_from_file_location('turn_score',Path(__file__).resolve().parents[1]/'scripts/score_minich_turn.py')
score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)


class TestTurnScoring(unittest.TestCase):
    def fixture(self, stepping=True, turn=10.):
        cols=['x','y','z','gravity_z','body_vx','body_vy','body_wz']+[f'{p}{i}' for p in ('foot_fz_','foot_xy_speed_','foot_world_z_') for i in range(4)]
        c={k:i for i,k in enumerate(cols)};n=round((3+turn+6)/.02);end=150+round(turn/.02)
        a=np.zeros((n,1,len(cols)));a[:,:,c['z']]=.26;a[:,:,c['gravity_z']]=-1
        a[150:end,:,c['body_wz']]=.25
        for f in range(4):a[:,:,c[f'foot_fz_{f}']]=10;a[:,:,c[f'foot_world_z_{f}']]=.02
        if stepping:
            for start in range(150,end-12,20):
                feet=(0,3) if ((start-150)//20)%2==0 else (1,2)
                for f in feet:
                    a[start:start+11,:,c[f'foot_fz_{f}']]=0
                    a[start:start+11,:,c[f'foot_world_z_{f}']]=.05
        m={'dt':.02,'schedule_s':[3,0,turn,6],'yaw_command_rad_s':.25,'columns':cols,'policies':[{'path':'test/model_1.pt','sha256':'test'}]}
        return m,{'states':a},c

    def evaluate(self,m,d,cfg=None):return score.evaluate(m,d,0,cfg or copy.deepcopy(score.DEFAULT))

    def test_normal_alternating_turn_then_stop_gets_full_score(self):
        m,d,_=self.fixture();r=self.evaluate(m,d)
        self.assertAlmostEqual(r['total'],100)
        self.assertGreater(r['metrics']['alternating_touchdown_fraction'],.99)

    def test_ground_spinning_is_deducted_even_with_perfect_tracking(self):
        m,d,_=self.fixture(False);r=self.evaluate(m,d)
        self.assertAlmostEqual(r['components']['turn'],40)
        self.assertLess(r['components']['feet'],20)
        cfg=copy.deepcopy(score.DEFAULT);cfg['require_steps']=False
        self.assertAlmostEqual(self.evaluate(m,d,cfg)['total'],100)

    def test_long_hanging_and_contact_slip_are_separately_penalized(self):
        m,d,c=self.fixture();d['states'][150:650,0,c['foot_fz_0']]=0
        r=self.evaluate(m,d);self.assertEqual(r['metric_scores']['longest_flight_s'],0)
        m,d,c=self.fixture();d['states'][150:650,0,c['foot_xy_speed_0']]=.2
        r=self.evaluate(m,d);self.assertEqual(r['metric_scores']['stance_slip_p95_m_s'],0)

    def test_stop_oscillation_cannot_cancel_absolute_angle(self):
        m,d,c=self.fixture();a=d['states'];a[700:825,0,c['body_wz']]=.1;a[825:950,0,c['body_wz']]=-.1
        r=self.evaluate(m,d)
        self.assertAlmostEqual(a[700:950,0,c['body_wz']].sum(),0)
        self.assertGreater(r['metrics']['stop_abs_angle_after_grace_deg'],28)
        self.assertTrue(r['diagnostics']['never_settled'])

    def test_short_turn_is_supported(self):
        m,d,_=self.fixture(turn=2);r=self.evaluate(m,d)
        self.assertTrue(np.isfinite(r['total']))

    def test_fall_and_nonfinite_return_zero(self):
        m,d,c=self.fixture();d['states'][300,0,c['z']]=.1
        self.assertEqual(self.evaluate(m,d)['total'],0)
        d['states'][300,0,c['z']]=np.nan
        self.assertTrue(self.evaluate(m,d)['invalid'])

    def test_stop_only_settles_when_remaining_capture_stays_stable(self):
        m,d,c=self.fixture();d['states'][800,0,c['body_wz']]=.05
        r=self.evaluate(m,d)
        self.assertGreater(r['metrics']['settling_time_s'],3)

    def test_tiny_clearance_does_not_count_as_normal_step(self):
        m,d,c=self.fixture()
        for f in range(4):d['states'][:,:,c[f'foot_world_z_{f}']]=.021
        r=self.evaluate(m,d)
        self.assertEqual(r['metrics']['normal_touchdown_fraction'],0)
        self.assertEqual(r['metrics']['touchdowns_per_foot_per_s'],0)
        self.assertFalse(r['minimum_lift_pass'])
        self.assertLess(r['components']['feet'],30)

    def test_one_cm_boundary_and_half_hz_are_inclusive(self):
        m,d,c=self.fixture(False,turn=4.)
        for start in (160,260):
            for f in range(4):
                d['states'][start:start+11,0,c[f'foot_fz_{f}']]=0
                d['states'][start:start+11,0,c[f'foot_world_z_{f}']]=.03
        r=self.evaluate(m,d)
        self.assertTrue(r['minimum_lift_pass'])
        self.assertAlmostEqual(r['metrics']['touchdowns_per_foot_per_s'],.5)
        for f in range(4):
            d['states'][:,0,c[f'foot_world_z_{f}']]=.029
        r=self.evaluate(m,d)
        self.assertFalse(r['minimum_lift_pass'])
        self.assertEqual(r['metrics']['touchdowns_per_foot_per_s'],0)

    def test_one_inactive_foot_fails_minimum(self):
        m,d,c=self.fixture()
        d['states'][:,0,c['foot_fz_0']]=10
        self.assertFalse(self.evaluate(m,d)['minimum_lift_pass'])

    def test_early_burst_cannot_hide_late_no_lift_window(self):
        m,d,c=self.fixture()
        for f in range(4):d['states'][400:650,0,c[f'foot_fz_{f}']]=10
        r=self.evaluate(m,d)
        self.assertTrue(all(x >= .5 for x in r['diagnostics']['qualified_lift_rates_per_foot_hz']))
        self.assertEqual(r['metrics']['touchdowns_per_foot_per_s'],0)
        self.assertFalse(r['minimum_lift_pass'])

    def test_invalid_standards_are_rejected(self):
        cfg=copy.deepcopy(score.DEFAULT);cfg['weights']['feet']=50
        with self.assertRaises(ValueError):score.validate(cfg)


if __name__=='__main__':unittest.main()

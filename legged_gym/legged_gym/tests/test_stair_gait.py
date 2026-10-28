"""Periodic gait feedback, temporal alignment, history and inference contracts."""
import isaacgym
import copy
import math
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
import torch
from legged_gym.envs.minich.minich_config import MiniChRoughCfg
from legged_gym.utils.stair_gait import StairGait
from legged_gym.utils.helpers import PolicyExporterHIM, export_policy_as_onnx
from rsl_rl.modules import HIMActorCritic
import test_observation_history_reset as history_tests


class TestStairGait(unittest.TestCase):
    def make(self, dt=.02):
        return StairGait(3, ['FR_foot', 'RL_foot', 'RR_foot', 'FL_foot'], 'cpu', MiniChRoughCfg.rewards, dt)

    def inputs(self):
        return (torch.tensor([[.2, 0., 1.]]).repeat(3, 1), torch.zeros(3, 3),
                torch.zeros(3, 4), torch.zeros(3, 4, 3), torch.zeros(3, 4, 3))

    def test_invalid_clock_settings_are_rejected(self):
        cfg=SimpleNamespace(**{k:getattr(MiniChRoughCfg.rewards,k) for k in dir(MiniChRoughCfg.rewards) if k.startswith('stair_')})
        for name,value in [('stair_gait_frequency',0.),('stair_gait_duty_factor',1.),
                           ('stair_gait_height',float('nan')),('stair_step_linear_threshold',float('inf'))]:
            bad=copy.copy(cfg);setattr(bad,name,value)
            with self.assertRaises(ValueError):
                StairGait(3,['FL_foot','FR_foot','RL_foot','RR_foot'],'cpu',bad,.02)

    def test_activation_phase_then_independent_clock(self):
        g=self.make();x=self.inputs();g.update(*x)
        torch.testing.assert_close(g.clock, torch.tensor([[0., 1.]]).repeat(3, 1))
        self.assertEqual(g.height_cost.sum(), 0)
        for n in range(1, 101):
            g.update(*x)
            expected=(n*.02*1.5)%1
            torch.testing.assert_close(g.clock[:,0],torch.full((3,),math.sin(2*math.pi*expected)),atol=2e-5,rtol=0)
        self.assertTrue(g.enabled.all())

    def test_hysteresis_and_command_gates(self):
        g=self.make();x=list(self.inputs())
        for deg, enabled in [(4,False),(6,True),(4,True),(2,False),(4,False)]:
            x[0][:]=torch.tensor([math.sin(math.radians(deg)),0,math.cos(math.radians(deg))])
            g.update(*x);self.assertEqual(g.enabled.all().item(),enabled)
            if not enabled:self.assertEqual(g.clock.abs().sum(),0)
        x[0][:]=torch.tensor([.2,0.,1.]);g.update(*x)
        for axis in range(3):
            x[1].zero_();x[1][:,axis]=.11;g.update(*x)
            self.assertFalse(g.enabled.any());self.assertEqual(g.height_cost.sum(),0)
        x[1].zero_();g.update(*x)
        self.assertEqual(g.phase.sum(),0)

    def test_named_diagonals_and_double_support(self):
        g=self.make();g.phase[:]=.3
        swing,height=g.targets()
        torch.testing.assert_close(swing[:,0],swing[:,1])
        torch.testing.assert_close(swing[:,2],swing[:,3])
        self.assertTrue((swing[:,0]>.99).all());self.assertEqual(swing[:,2:].sum(),0)
        g.phase[:]=.8;swing,_=g.targets()
        self.assertTrue((swing[:,2:]>.99).all());self.assertEqual(swing[:,:2].sum(),0)
        g.phase[:]=.05;swing,_=g.targets();self.assertEqual(swing.sum(),0)

    def test_dense_signal_before_landing_and_phase_alignment(self):
        g=self.make();x=list(self.inputs());g.update(*x);g.phase[:]=.3
        swing,target=g.targets();x[3][:,:,2]=50
        g.update(*x)
        # Reward belongs to .3 that was observed, not the next .33 clock.
        torch.testing.assert_close(g.target_height,target)
        self.assertTrue((g.height_cost>.49).all());self.assertTrue((g.force_cost>.49).all())
        self.assertTrue((g.phase>.32).all())
        g.phase[:]=.3;x[2][:]=target;x[3][:,:2]=0
        g.update(*x)
        self.assertLess(g.height_cost.max().item(),1e-10)
        self.assertLess(g.force_cost.max().item(),1e-10)
        x[4].fill_(1.);g.update(*x)
        self.assertTrue((g.velocity_cost>0).all())

    def test_partial_reset_and_read_only_queries(self):
        g=self.make();x=self.inputs()
        for _ in range(12):g.update(*x)
        before=g.phase.clone();g.targets();g.targets();torch.testing.assert_close(g.phase,before)
        g.reset(torch.tensor([1]));self.assertEqual(g.clock[1].abs().sum(),0)
        torch.testing.assert_close(g.phase[[0,2]],before[[0,2]])
        g.update(*x);self.assertEqual(g.phase[1],0)

    def test_period_integral_not_dt_dependent(self):
        totals=[]
        for dt in [.01,.02]:
            g=self.make(dt);x=list(self.inputs());x[3][:,:,2]=50;g.update(*x)
            total=0.
            for _ in range(round(2/dt)):
                g.update(*x);total+=g.height_cost[0].item()*dt
            totals.append(total)
        self.assertAlmostEqual(*totals,places=4)

    def test_history_privileged_velocity_and_no_clock_noise(self):
        r=history_tests.TestObservationHistoryReset().make_robot()
        r.num_one_step_obs=47;r.num_one_step_privileged_obs=53;r.num_privileged_obs=53
        r.obs_buf=torch.zeros(3,282);r.privileged_obs_buf=torch.zeros(3,53)
        r.stair_gait=self.make();r.stair_gait.update(*self.inputs())
        r.compute_observations()
        torch.testing.assert_close(r.obs_buf[:,45:47],r.stair_gait.clock)
        torch.testing.assert_close(r.privileged_obs_buf[:,47:50],r.base_lin_vel)
        prior=r.obs_buf.clone();phase=r.stair_gait.phase.clone()
        r.compute_observations();r.get_current_obs();terminal=r.compute_termination_observations(torch.tensor([1]))
        torch.testing.assert_close(r.obs_buf[:,47:],prior[:,:-47])
        torch.testing.assert_close(terminal[:,45:47],r.stair_gait.clock[1:2])
        torch.testing.assert_close(r.stair_gait.phase,phase)
        r.reset_idx(torch.tensor([1]));r.compute_observations()
        frames=r.obs_buf.view(3,6,47)
        torch.testing.assert_close(frames[1],frames[1,:1].repeat(6,1))
        self.assertEqual(frames[1,:,45:].abs().sum(),0)

    def test_jit_and_onnx_legacy_and_clock_dimensions(self):
        import onnx
        from onnx.reference import ReferenceEvaluator
        for frame in (45,47):
            policy=HIMActorCritic(6*frame,frame+193,frame,12).eval()
            if frame==47:policy.observation_contract=self.make().contract()
            model=PolicyExporterHIM(policy).eval();scripted=torch.jit.script(model)
            obs=torch.randn(2,6*frame)
            with torch.no_grad():
                expected=policy.act_inference(obs)
                torch.testing.assert_close(scripted(obs),expected)
            with tempfile.TemporaryDirectory() as folder:
                export_policy_as_onnx(policy,folder)
                graph=onnx.load(str(Path(folder)/'policy.onnx'))
                self.assertEqual(graph.graph.input[0].type.tensor_type.shape.dim[1].dim_value,6*frame)
                actual=ReferenceEvaluator(graph).run(None,{'obs':obs[:1].numpy()})[0]
                torch.testing.assert_close(torch.from_numpy(actual),expected[:1],atol=1e-5,rtol=1e-5)

    def test_checkpoint_contract_rejects_missing_or_changed_clock(self):
        from rsl_rl.runners import HIMOnPolicyRunner
        policy=HIMActorCritic(282,240,47,12)
        policy.observation_contract=self.make().contract()
        runner=HIMOnPolicyRunner.__new__(HIMOnPolicyRunner)
        runner.env=SimpleNamespace(num_one_step_obs=47)
        runner.alg=SimpleNamespace(actor_critic=policy)
        saved={'model_state_dict':policy.state_dict(),'iter':0,'infos':None}
        with tempfile.TemporaryDirectory() as folder:
            path=str(Path(folder)/'model.pt')
            torch.save(saved,path)
            with self.assertRaisesRegex(ValueError,'contract'):runner.load(path,load_optimizer=False)
            saved['observation_contract']=dict(policy.observation_contract,frequency_hz=2.)
            torch.save(saved,path)
            with self.assertRaisesRegex(ValueError,'contract'):runner.load(path,load_optimizer=False)
            saved['observation_contract']=policy.observation_contract
            torch.save(saved,path);runner.load(path,load_optimizer=False)
            runner.env.num_one_step_obs=45
            with self.assertRaisesRegex(ValueError,'47-D'):runner.load(path,load_optimizer=False)

    def test_estimator_velocity_slice_and_forward_dimensions(self):
        policy=HIMActorCritic(282,240,47,12)
        obs=torch.zeros(3,282);critic=torch.zeros(3,240);critic[:,47:50]=.2
        vel,latent=policy.estimator(obs)
        self.assertEqual(vel.shape,(3,3));self.assertEqual(latent.shape,(3,16))
        self.assertEqual(policy.act_inference(obs).shape,(3,12))
        self.assertEqual(policy.evaluate(critic).shape,(3,1))


if __name__=='__main__':unittest.main()

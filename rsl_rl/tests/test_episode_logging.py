"""Exercise actual log methods without creating an env, policy or training loop."""
import contextlib
import io
import types
import unittest
import torch
from rsl_rl.runners import HIMOnPolicyRunner, OnPolicyRunner


class TestEpisodeLogging(unittest.TestCase):
    def log(self, runner_type, infos):
        runner = runner_type.__new__(runner_type)
        runner.device = 'cpu'
        runner.tot_timesteps = 0
        runner.tot_time = 0.
        runner.num_steps_per_env = 1
        runner.env = types.SimpleNamespace(num_envs=1)
        runner.current_learning_iteration = 0
        runner.alg = types.SimpleNamespace(learning_rate=.001,
            actor_critic=types.SimpleNamespace(std=torch.ones(1)))
        result = {}
        runner.writer = types.SimpleNamespace(add_scalar=lambda key,value,step: result.update({key:float(value)}))
        locs = dict(ep_infos=infos, collection_time=.1, learn_time=.1, it=0,
                    num_learning_iterations=1, rewbuffer=[], lenbuffer=[],
                    mean_value_loss=0.,mean_surrogate_loss=0.,mean_estimation_loss=0.,mean_swap_loss=0.)
        with contextlib.redirect_stdout(io.StringIO()):
            runner.log(locs)
        return result

    def test_missing_first_or_later_and_no_mutation(self):
        for runner in (HIMOnPolicyRunner,OnPolicyRunner):
            for infos in ([{'long_stairs_success':1.}, {}],
                          [{}, {'long_stairs_success':1.}]):
                with self.subTest(runner=runner.__name__,infos=infos):
                    before=[dict(info) for info in infos]
                    result=self.log(runner,infos)
                    self.assertEqual(result['Episode/long_stairs_success'],1.)
                    self.assertEqual(infos,before)

    def test_mixed_shapes_and_missing_not_zero(self):
        for runner in (HIMOnPolicyRunner,OnPolicyRunner):
            result=self.log(runner,[{'metric':1}, {}, {'metric':torch.tensor(0.)},
                                    {'metric':torch.tensor([.5,.5])}, {'empty':torch.empty(0)}])
            self.assertEqual(result['Episode/metric'],.5)
            self.assertNotIn('Episode/empty',result)

    def test_no_episodes(self):
        for runner in (HIMOnPolicyRunner,OnPolicyRunner):
            for infos in ([],[{},{}]):
                result=self.log(runner,infos)
                self.assertFalse(any(key.startswith('Episode/') for key in result))

if __name__=='__main__':unittest.main()

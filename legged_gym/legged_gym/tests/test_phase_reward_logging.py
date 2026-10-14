"""Check stage denominators, command transitions and reward transparency."""
import unittest
from types import SimpleNamespace

import isaacgym
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils.phase_reward_logger import PhaseRewardLogger


class TestPhaseRewardLogging(unittest.TestCase):
    def test_phase_rates_and_direction_split(self):
        logger = PhaseRewardLogger(('feet_air_time', 'tracking_ang_vel'), .02, 'cpu')
        commands = torch.tensor([[0., 0., .25], [1., 0., .25],
                                 [0., 0., 0.], [0., 0., -.25]])
        terms = {'feet_air_time': torch.tensor([.02, 0., 0., -.04]),
                 'tracking_ang_vel': torch.tensor([.02, .01, .02, .02])}
        raw = sum(terms.values())
        logger.update(commands, terms, raw.clamp(min=0), (raw < 0).float())
        values = logger.pop()
        self.assertAlmostEqual(values['PhaseReward/turn/rew_feet_air_time'], -.5, places=5)
        self.assertAlmostEqual(values['PhaseReward/turn/total_unclipped'], .5, places=5)
        self.assertAlmostEqual(values['PhaseReward/turn/total_policy'], 1., places=5)
        self.assertEqual(values['PhaseReward/turn/clipped_fraction'], .5)
        self.assertAlmostEqual(values['PhaseReward/turn_positive/rew_feet_air_time'], 1., places=5)
        self.assertAlmostEqual(values['PhaseReward/turn_negative/rew_feet_air_time'], -2., places=5)
        self.assertEqual(values['PhaseReward/translate/samples'], 1)
        self.assertEqual(values['PhaseReward/stand/samples'], 1)

    def test_transition_weights_by_samples_not_episodes(self):
        logger = PhaseRewardLogger(('tracking_ang_vel',), .02, 'cpu')
        def update(commands, rewards):
            rewards = torch.tensor(rewards)
            logger.update(torch.tensor(commands), {'tracking_ang_vel': rewards}, rewards,
                          torch.zeros_like(rewards))
        update([[0., 0., .25], [0., 0., -.25]], [.02, .02])
        # The same robots switch commands without ending either episode.
        update([[0., 0., 0.], [0., 0., -.25]], [.04, .08])
        result = logger.pop()
        self.assertEqual(result['PhaseReward/turn/samples'], 3)
        self.assertAlmostEqual(result['PhaseReward/turn/rew_tracking_ang_vel'], 2., places=5)
        self.assertAlmostEqual(result['PhaseReward/stand/rew_tracking_ang_vel'], 2., places=5)
        self.assertEqual(result['PhaseReward/stand/sample_fraction'], .25)

    def test_missing_phase_omits_reward_and_pop_clears(self):
        logger = PhaseRewardLogger(('tracking_ang_vel',), .02, 'cpu')
        rewards = torch.tensor([.02])
        logger.update(torch.zeros(1, 3), {'tracking_ang_vel': rewards}, rewards, rewards * 0)
        result = logger.pop()
        self.assertNotIn('PhaseReward/turn/rew_tracking_ang_vel', result)
        self.assertEqual(result['PhaseReward/turn/samples'], 0)
        self.assertEqual(logger.pop()['PhaseReward/stand/samples'], 0)

    def test_logging_preserves_reward_and_calls_stateful_terms_once(self):
        def robot(logging):
            calls = []
            def reward(name, value):
                def calculate():
                    calls.append(name)
                    return torch.tensor([value])
                return calculate
            names = ('feet_air_time', 'tracking_ang_vel')
            r = SimpleNamespace(commands=torch.tensor([[0., 0., .25]]),
                reward_names=names, reward_scales={n: .02 for n in names},
                reward_functions=[reward(names[0], -4.), reward(names[1], 1.)],
                rew_buf=torch.zeros(1), episode_sums={n: torch.zeros(1) for n in names},
                cfg=SimpleNamespace(rewards=SimpleNamespace(only_positive_rewards=True)))
            if logging:
                r.phase_reward_logger = PhaseRewardLogger(names, .02, 'cpu')
            LeggedRobot.compute_reward(r)
            return r, calls
        plain, plain_calls = robot(False)
        logged, logged_calls = robot(True)
        torch.testing.assert_close(plain.rew_buf, logged.rew_buf)
        for name in plain.episode_sums:
            torch.testing.assert_close(plain.episode_sums[name], logged.episode_sums[name])
        self.assertEqual(plain_calls, logged_calls)
        self.assertEqual(logged_calls, ['feet_air_time', 'tracking_ang_vel'])
        result = logged.phase_reward_logger.pop()
        self.assertAlmostEqual(result['PhaseReward/turn/rew_feet_air_time'], -4., places=5)
        self.assertEqual(result['PhaseReward/turn/total_policy'], 0.)
        self.assertEqual(result['PhaseReward/turn/clipped_fraction'], 1.)

    def test_termination_stays_after_clipping_and_disabled_clip_is_reported(self):
        for clip, expected_policy in ((True, -.02), (False, -.08)):
            with self.subTest(clip=clip):
                names = ('tracking_ang_vel', 'termination')
                logger = PhaseRewardLogger(names, .02, 'cpu')
                r = SimpleNamespace(commands=torch.tensor([[0., 0., -.25]]),
                    reward_names=['tracking_ang_vel'], reward_scales={n: .02 for n in names},
                    reward_functions=[lambda: torch.tensor([-3.])],
                    _reward_termination=lambda: torch.tensor([-1.]),
                    rew_buf=torch.zeros(1), episode_sums={n: torch.zeros(1) for n in names},
                    cfg=SimpleNamespace(rewards=SimpleNamespace(only_positive_rewards=clip)),
                    phase_reward_logger=logger)
                LeggedRobot.compute_reward(r)
                self.assertAlmostEqual(r.rew_buf.item(), expected_policy, places=6)
                result = logger.pop()
                self.assertAlmostEqual(result['PhaseReward/turn_negative/rew_termination'], -1., places=5)
                self.assertAlmostEqual(result['PhaseReward/turn_negative/total_unclipped'], -4., places=5)
                self.assertAlmostEqual(result['PhaseReward/turn_negative/total_policy'], expected_policy / .02, places=5)
                self.assertEqual(result['PhaseReward/turn_negative/clipped_fraction'], float(clip))


if __name__ == '__main__':
    unittest.main()

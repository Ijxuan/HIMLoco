"""Check continuous overdue cost, touchdown scores and command-gate boundaries."""
import types
import unittest

from legged_gym.envs.base.legged_robot import LeggedRobot
import torch


class TestAirTimeContinuousCost(unittest.TestCase):
    def make_robot(self, target=True):
        r = LeggedRobot.__new__(LeggedRobot)
        r.dt = .02
        r.cfg = types.SimpleNamespace(rewards=types.SimpleNamespace())
        if target:
            r.cfg.rewards.feet_air_time_target = .25
            r.cfg.rewards.feet_air_time_min = .20
            r.cfg.rewards.feet_air_time_max = .30
        r.feet_indices = torch.arange(4)
        r.contact_forces = torch.zeros(1, 4, 3)
        r.contact_forces[:, 1:, 2] = 2.
        r.last_contacts = torch.zeros(1, 4, dtype=torch.bool)
        r.feet_air_time = torch.zeros(1, 4)
        r.feet_air_time_penalty_active = torch.zeros(1, 4, dtype=torch.bool)
        r.commands = torch.tensor([[0., 0., .25, 0.]])
        return r

    def flight(self, steps):
        r = self.make_robot()
        scores = []
        for step in range(steps):
            if step == steps - 1:
                r.contact_forces[:, 0, 2] = 2.
            scores.append(r._reward_feet_air_time().item())
        return r, scores

    def test_normal_and_short_touchdown_scores_preserved(self):
        for steps in (5, 10, 12, 13, 15):
            with self.subTest(steps=steps):
                _, scores = self.flight(steps)
                duration = steps * .02
                width = .05
                expected = 1 - abs(duration - .25) / width
                self.assertAlmostEqual(sum(scores[:-1]), 0., places=5)
                self.assertAlmostEqual(scores[-1], expected, places=5)

    def test_long_flight_pays_incrementally_without_double_charge(self):
        r, scores = self.flight(50)  # 1 s total, including touchdown timer tick.
        self.assertAlmostEqual(sum(scores), -(1. - .3) / .05, places=4)
        self.assertAlmostEqual(sum(scores[:15]), 0., places=5)
        for score in scores[15:]:
            self.assertAlmostEqual(score, -.4, places=5)
        self.assertAlmostEqual(scores[-1], -.4, places=5)
        self.assertEqual(r.feet_air_time[0, 0].item(), 0.)
        self.assertEqual(r._reward_feet_air_time().item(), 0.)

    def test_penalty_is_received_before_touchdown(self):
        r = self.make_robot()
        scores = [r._reward_feet_air_time().item() for _ in range(50)]
        self.assertLess(sum(scores), -13.9)
        self.assertFalse(r.last_contacts[0, 0])
        self.assertGreater(r.feet_air_time[0, 0].item(), .99)

    def test_increment_survives_existing_total_reward_clipping(self):
        _, scores = self.flight(50)
        # Same raw total overdue cost (-14), spread across 35 steps. A single
        # foot's -0.008 weighted increment remains below +0.04 dense reward.
        actual = sum(max(.04 + score * .02, 0.) for score in scores)
        self.assertAlmostEqual(actual, 50 * .04 - 14 * .02, places=4)

    def test_turn_cost_continues_until_touchdown_after_command_change(self):
        for command in ((0., 0., 0.), (.2, 0., .25), (0., 0., .2)):
            with self.subTest(command=command):
                r = self.make_robot()
                for _ in range(20):
                    r._reward_feet_air_time()
                self.assertEqual(r.feet_air_time_penalty_active.tolist(), [[True, False, False, False]])
                r.commands[:, :3] = torch.tensor(command)
                for _ in range(10):
                    self.assertAlmostEqual(r._reward_feet_air_time().item(), -.4, places=5)
                r.contact_forces[:, 0, 2] = 2.
                self.assertAlmostEqual(r._reward_feet_air_time().item(), -.4, places=5)
                self.assertFalse(r.feet_air_time_penalty_active.any())
                self.assertEqual(r.feet_air_time[0, 0].item(), 0.)
                self.assertEqual(r._reward_feet_air_time().item(), 0.)
                r.contact_forces[:, 0, 2] = 0.
                for _ in range(30):
                    self.assertEqual(r._reward_feet_air_time().item(), 0.)
                self.assertFalse(r.feet_air_time_penalty_active.any())

    def test_turn_ending_before_overdue_cost_does_not_grant_eligibility(self):
        r = self.make_robot()
        for _ in range(10):
            r._reward_feet_air_time()
        r.commands.zero_()
        for _ in range(30):
            self.assertEqual(r._reward_feet_air_time().item(), 0.)
        self.assertFalse(r.feet_air_time_penalty_active.any())

    def test_overdue_flight_acquires_eligibility_when_turn_begins(self):
        r = self.make_robot()
        r.commands.zero_()
        for _ in range(30):
            self.assertEqual(r._reward_feet_air_time().item(), 0.)
        r.commands[:, 2] = -.25
        self.assertAlmostEqual(r._reward_feet_air_time().item(), -.4, places=5)
        r.commands.zero_()
        self.assertAlmostEqual(r._reward_feet_air_time().item(), -.4, places=5)

    def test_yaw_threshold_translation_and_stance_remain_gated(self):
        for command in ((0., 0., .2), (.2, 0., .25), (0., 0., 0.)):
            r = self.make_robot()
            r.commands[:, :3] = torch.tensor(command)
            r.feet_air_time[:, 0] = 1.
            self.assertEqual(r._reward_feet_air_time().item(), 0.)
        r = self.make_robot()
        r.contact_forces[:, :, 2] = 2.
        for _ in range(30):
            self.assertEqual(r._reward_feet_air_time().item(), 0.)
        self.assertTrue(torch.all(r.feet_air_time == 0))

    def test_legacy_without_target_keeps_touchdown_formula(self):
        r = self.make_robot(target=False)
        for _ in range(19):
            self.assertEqual(r._reward_feet_air_time().item(), 0.)
        r.contact_forces[:, 0, 2] = 2.
        self.assertAlmostEqual(r._reward_feet_air_time().item(), .4 - .5, places=5)


if __name__ == '__main__':
    unittest.main()

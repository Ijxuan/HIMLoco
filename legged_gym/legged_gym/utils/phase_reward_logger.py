"""Sample-weighted reward diagnostics for one training rollout."""
import torch


class PhaseRewardLogger:
    phases = ("turn", "translate", "stand", "turn_positive", "turn_negative")

    def __init__(self, reward_names, dt, device):
        self.reward_names = tuple(reward_names)
        self.dt = dt
        self.counts = torch.zeros(len(self.phases), device=device)
        self.sums = torch.zeros(len(self.phases), len(self.reward_names) + 3, device=device)

    @torch.no_grad()
    def update(self, commands, rewards, policy_total, clipped):
        # Translation takes precedence for combined linear/yaw commands.
        translating = torch.norm(commands[:, :2], dim=1) >= 0.1
        turning = ~translating & (commands[:, 2].abs() > 1e-6)
        standing = ~translating & ~turning
        masks = torch.stack((turning, translating, standing,
                             turning & (commands[:, 2] > 0),
                             turning & (commands[:, 2] < 0)), dim=0).to(self.sums.dtype)
        terms = torch.stack([rewards[name] for name in self.reward_names], dim=1)
        values = torch.cat((terms, terms.sum(dim=1, keepdim=True),
                            policy_total.unsqueeze(1), clipped.unsqueeze(1)), dim=1)
        self.counts += masks.sum(dim=1)
        self.sums += masks @ values

    @torch.no_grad()
    def pop(self):
        counts = self.counts.cpu().tolist()
        sums = self.sums.cpu().tolist()
        total_count = sum(counts[:3])
        result = {}
        names = ["rew_" + name for name in self.reward_names]
        names += ["total_unclipped", "total_policy"]
        for phase, count, values in zip(self.phases, counts, sums):
            prefix = "PhaseReward/" + phase + "/"
            result[prefix + "samples"] = count
            result[prefix + "sample_fraction"] = count / total_count if total_count else 0.
            if count:
                for name, value in zip(names, values[:-1]):
                    result[prefix + name] = value / (count * self.dt)
                result[prefix + "clipped_fraction"] = values[-1] / count
        self.counts.zero_()
        self.sums.zero_()
        return result


class TurnLiftLogger:
    """Weight-independent lift rates and full-window pass fractions by direction."""
    phases = ('turn', 'turn_positive', 'turn_negative', 'startup', 'steady')
    feet = ('FL', 'FR', 'RL', 'RR')

    def __init__(self, dt, device):
        self.dt = dt
        # samples, mature samples, 4 events, 4 passes, all pass, 4 overdue seconds
        self.sums = torch.zeros(len(self.phases), 15, device=device)

    @torch.no_grad()
    def update(self, commands, active, mature, events, passed, overdue_seconds):
        masks = torch.stack((active, active & (commands[:, 2] > 0),
            active & (commands[:, 2] < 0), active & ~mature, active & mature)).float()
        values = torch.cat((torch.ones_like(mature[:, None], dtype=torch.float),
            mature[:, None].float(), events.float(),
            (passed & mature[:, None]).float(),
            (passed.all(dim=1) & mature)[:, None].float(), overdue_seconds), dim=1)
        self.sums += masks @ values

    @torch.no_grad()
    def pop(self):
        result = {}
        for phase, row in zip(self.phases, self.sums.cpu().tolist()):
            prefix = 'TurnLift/' + phase + '/'
            samples, mature = row[:2]
            result[prefix + 'samples'] = samples
            result[prefix + 'full_window_samples'] = mature
            if samples:
                for i, foot in enumerate(self.feet):
                    result[prefix + foot + '/lift_rate_hz'] = row[2+i] / (samples*self.dt)
                    result[prefix + foot + '/overdue_seconds'] = row[11+i]
                    result[prefix + foot + '/overdue_fraction'] = row[11+i] / (samples*self.dt)
            if mature:
                fractions = [row[6+i] / mature for i in range(4)]
                for foot, fraction in zip(self.feet, fractions):
                    result[prefix + foot + '/window_pass_fraction'] = fraction
                result[prefix + 'worst_foot_window_pass_fraction'] = min(fractions)
                result[prefix + 'all_feet_window_pass_fraction'] = row[10] / mature
        self.sums.zero_()
        return result

"""Per-foot swing peak scoring, independent of foresight or stair detection."""
import math
import torch


def peak_height_score(height, lower=0.025, upper=0.08, target=0.05):
    """Bounded asymmetric curve: zero at bounds, +1 at the target."""
    if not 0 < lower < target < upper:
        raise ValueError('Expected 0 < lower < target < upper')
    width = torch.where(height <= target, target - lower, upper - target)
    inside = 1 - ((height - target) / width).square()
    below = -((lower - height) / lower).clamp(0, 1).square()
    above = -((height - upper) / (upper - target)).clamp(0, 1).square()
    return torch.where(height < lower, below, torch.where(height > upper, above, inside))


class SwingPeakHeight:
    def __init__(self, num_envs, num_feet, device, cfg, dt):
        self.cfg, self.dt = cfg, dt
        for name in ('min', 'target', 'max', 'min_air_time', 'timeout', 'stance_time', 'landing_time'):
            value = getattr(cfg, 'swing_peak_height_' + name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid swing_peak_height_' + name)
        if not (0 < dt <= cfg.swing_peak_height_min_air_time < cfg.swing_peak_height_timeout):
            raise ValueError('Invalid swing height timing')
        if not cfg.swing_peak_height_min < cfg.swing_peak_height_target < cfg.swing_peak_height_max:
            raise ValueError('Invalid swing height band')
        shape = (num_envs, num_feet)
        self.active = torch.zeros(shape, dtype=torch.bool, device=device)
        self.armed = torch.zeros_like(self.active)
        self.stance = torch.zeros(shape, device=device)
        self.landing = torch.zeros_like(self.stance)
        self.elapsed = torch.zeros_like(self.stance)
        self.peak = torch.zeros_like(self.stance)
        self.command_direction = torch.zeros(*shape, 2, device=device)
        self.event = torch.zeros_like(self.active)
        self.event_peak = torch.zeros_like(self.stance)
        self.timed_out = torch.zeros_like(self.active)
        self.score = torch.zeros(num_envs, device=device)
        self.stats = {k: torch.zeros_like(self.stance) for k in
                      ('count', 'peak_sum_cm', 'under', 'in_band', 'over', 'timeout',
                       'started', 'cancelled', 'short', 'valid_landing', 'score_sum',
                       'positive_sum', 'negative_sum', 'enabled_seconds')}

    def reset(self, ids):
        self.stats["cancelled"][ids] += self.active[ids]
        for value in (self.active, self.armed, self.stance, self.landing, self.elapsed,
                      self.peak, self.command_direction, self.event, self.event_peak,
                      self.timed_out, self.score):
            value[ids] = 0

    def update(self, clearance, support, commands, enabled):
        c = self.cfg
        self.score.zero_(); self.event.zero_(); self.event_peak.zero_(); self.timed_out.zero_()
        direction = commands[:, :2] / commands[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        aligned = (self.command_direction * direction[:, None]).sum(-1) > .5
        valid = enabled[:, None].expand_as(support)
        cancel = ~valid | (self.active & ~aligned)
        self.stats["cancelled"] += cancel & self.active
        self.stats["enabled_seconds"] += valid * self.dt
        self.active &= ~cancel; self.armed &= ~cancel
        self.stance.masked_fill_(cancel, 0.); self.landing.masked_fill_(cancel, 0.)
        self.elapsed.masked_fill_(cancel, 0.); self.peak.masked_fill_(cancel, 0.)
        valid = valid & ~cancel
        idle = ~self.active
        self.stance[:] = torch.where(idle & support & valid, self.stance + self.dt, 0.)
        self.armed |= idle & (self.stance + 1e-7 >= c.swing_peak_height_stance_time)
        start = idle & self.armed & ~support & valid
        self.stats["started"] += start
        self.active |= start; self.armed &= ~start
        self.peak[start] = 0.; self.elapsed[start] = 0.; self.landing[start] = 0.
        self.command_direction[:] = torch.where(start[..., None], direction[:, None], self.command_direction)
        self.elapsed += self.active * self.dt
        # No ground/stance height samples enter the swing peak.
        self.peak[:] = torch.where(self.active & ~support, torch.maximum(self.peak, clearance.clamp_min(0.)), self.peak)
        self.landing[:] = torch.where(self.active & support, self.landing + self.dt, 0.)
        landed = self.active & (self.landing + 1e-7 >= c.swing_peak_height_landing_time)
        timeout = self.active & ~landed & (self.elapsed + 1e-7 >= c.swing_peak_height_timeout)
        self.event[:] = landed | timeout
        self.event_peak[:] = torch.where(self.event, self.peak, 0.)
        self.timed_out[:] = timeout
        score = peak_height_score(self.peak, c.swing_peak_height_min, c.swing_peak_height_max, c.swing_peak_height_target)
        # Tiny contact dropouts cannot earn positive credit; hanging cannot defer settlement.
        air_duration = self.elapsed - self.landing
        score = torch.where(air_duration + 1e-7 < c.swing_peak_height_min_air_time, score.clamp_max(0.), score)
        score = torch.where(timeout, -torch.ones_like(score), score)
        # Environment multiplies by dt; coefficient is a per-foot/per-swing budget.
        self.score[:] = (score * self.event).sum(-1) / self.dt
        self.stats['count'] += self.event
        self.stats['peak_sum_cm'] += self.event_peak * 100
        self.stats['under'] += self.event & (self.peak < c.swing_peak_height_min)
        self.stats['in_band'] += self.event & (self.peak >= c.swing_peak_height_min) & (self.peak <= c.swing_peak_height_max)
        self.stats['over'] += self.event & (self.peak > c.swing_peak_height_max)
        self.stats['timeout'] += timeout
        short = air_duration + 1e-7 < c.swing_peak_height_min_air_time
        self.stats['short'] += self.event & short
        self.stats['valid_landing'] += landed & ~short & (self.peak >= c.swing_peak_height_min) & (self.peak <= c.swing_peak_height_max)
        self.stats['score_sum'] += score * self.event
        self.stats['positive_sum'] += score.clamp_min(0.) * self.event
        self.stats['negative_sum'] += score.clamp_max(0.) * self.event
        self.active &= ~self.event
        self.stance.masked_fill_(self.event, 0.)
        # A timeout stays disarmed until stable support has been observed again.
        self.armed &= ~self.event

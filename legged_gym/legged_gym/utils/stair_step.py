"""Event-based diagonal stepping at near-zero commands (no policy clock input)."""
import math
import torch


class StairStep:
    def __init__(self, num_envs, feet_names, device, cfg, dt):
        self.cfg, self.dt = cfg, dt
        names = ("FL_foot", "RR_foot", "FR_foot", "RL_foot")
        if len(feet_names) != 4 or any(feet_names.count(n) != 1 for n in names):
            raise ValueError("stair_step requires named Mini Cheetah feet")
        self.order = torch.tensor([feet_names.index(n) for n in names], device=device)
        values = [getattr(cfg, 'stair_step_' + name) for name in (
            'on_deg', 'off_deg', 'linear_threshold', 'yaw_threshold', 'min_height',
            'target_height', 'max_height', 'sync_time', 'confirm_time',
            'min_air_time', 'max_air_time', 'overdue_time')]
        if (not all(math.isfinite(v) and v > 0 for v in values) or not math.isfinite(dt) or not 0 < dt <= cfg.stair_step_confirm_time
                or not 0 < cfg.stair_step_off_deg < cfg.stair_step_on_deg < 90
                or not cfg.stair_step_min_height < cfg.stair_step_target_height < cfg.stair_step_max_height
                or not cfg.stair_step_confirm_time < cfg.stair_step_min_air_time < cfg.stair_step_max_air_time
                or cfg.stair_step_sync_time >= cfg.stair_step_max_air_time):
            raise ValueError("Invalid stair_step parameters")
        self.slope_mode = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.enabled = torch.zeros_like(self.slope_mode)
        self.expected = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self.elapsed = torch.zeros(num_envs, device=device)
        self.score = torch.zeros_like(self.elapsed)
        self.overdue = torch.zeros_like(self.elapsed)
        self.active = torch.zeros(num_envs, 2, dtype=torch.bool, device=device)
        self.valid = torch.zeros_like(self.active)
        self.age = torch.zeros(num_envs, 2, device=device)
        self.stance = torch.zeros(num_envs, 2, 2, device=device)
        self.air = torch.zeros_like(self.stance)
        self.peak = torch.zeros_like(self.stance)
        self.lift_time = torch.full_like(self.stance, -1.)
        self.launch_z = torch.zeros_like(self.stance)
        self.stats = {name: torch.zeros_like(self.elapsed) for name in
                      ('enabled_seconds', 'valid_events', 'sync_sum_s', 'peak_sum_m', 'overdue_seconds')}

    def reset(self, ids):
        for name in ('slope_mode', 'enabled', 'elapsed', 'score', 'overdue',
                     'active', 'valid', 'age', 'stance', 'air', 'peak', 'launch_z'):
            getattr(self, name)[ids] = 0
        self.expected[ids] = -1
        self.lift_time[ids] = -1

    def update(self, normal, commands, sole_z, ground, support):
        c, dt = self.cfg, self.dt
        angle = torch.atan2(normal[:, :2].norm(dim=1), normal[:, 2])
        self.slope_mode = ((angle > math.radians(c.stair_step_on_deg)) |
                           (self.slope_mode & (angle >= math.radians(c.stair_step_off_deg))))
        enabled = (self.slope_mode & (commands[:, :2].norm(dim=1) < c.stair_step_linear_threshold)
                   & (commands[:, 2].abs() < c.stair_step_yaw_threshold))
        # Clear partial swings across mode transitions, retaining slope hysteresis.
        clear = ~enabled | ~self.enabled
        self.expected[clear] = -1
        self.elapsed[clear] = 0
        self.active[clear] = False
        self.valid[clear] = False
        self.stance[clear] = 0
        self.air[clear] = 0
        self.peak[clear] = 0
        self.lift_time[clear] = -1
        self.enabled[:] = enabled
        self.score.zero_()
        self.elapsed += enabled * dt
        shape = (-1, 2, 2)
        contact = support[:, self.order].reshape(shape)
        z = sole_z[:, self.order].reshape(shape)
        clearance = (sole_z - ground)[:, self.order].reshape(shape)
        stable = self.stance >= c.stair_step_confirm_time - 1e-6
        # Only start from confirmed four-foot support; either diagonal may lead.
        start = (~self.active & stable.all(dim=2) & (~contact).any(dim=2)
                 & stable.flip(1).all(dim=2) & contact.flip(1).all(dim=2) & enabled[:, None])
        self.launch_z = torch.where(start[..., None], z - clearance, self.launch_z)
        self.age[start] = 0
        self.air[start] = 0
        self.peak[start] = 0
        self.lift_time[start] = -1
        self.valid[start] = True
        self.active |= start
        self.age += self.active * dt
        relift = (self.active[..., None] & ~contact & (self.lift_time >= 0) & (self.stance > 0))
        self.valid &= ~relift.any(dim=2)  # Do not sum several tiny hops into one swing.
        first_lift = self.active[..., None] & ~contact & (self.lift_time < 0)
        self.lift_time = torch.where(first_lift, self.age[..., None], self.lift_time)
        self.air += (self.active[..., None] & ~contact) * dt
        height = torch.minimum(clearance, z - self.launch_z).clamp_min(0.)
        self.peak = torch.where(self.active[..., None] & ~contact, torch.maximum(self.peak, height), self.peak)
        self.stance = torch.where(contact & enabled[:, None, None], self.stance + dt, 0.)
        # Loss of opposite-diagonal support invalidates the whole event.
        self.valid &= contact.flip(1).all(dim=2)
        landed = self.active & (self.stance >= c.stair_step_confirm_time - 1e-6).all(dim=2)
        sync = self.lift_time.max(dim=2).values - self.lift_time.min(dim=2).values
        valid = (landed & self.valid & (self.lift_time >= 0).all(dim=2)
                 & (sync <= c.stair_step_sync_time + 1e-6)
                 & (self.air >= c.stair_step_min_air_time - 1e-6).all(dim=2)
                 & (self.air <= c.stair_step_max_air_time + 1e-6).all(dim=2))
        height_score = torch.minimum(
            (self.peak - c.stair_step_min_height) / (c.stair_step_target_height - c.stair_step_min_height),
            (c.stair_step_max_height - self.peak) / (c.stair_step_max_height - c.stair_step_target_height)
        ).clamp(-1., 1.).min(dim=2).values
        for group in (0, 1):
            eligible = valid[:, group] & ((self.expected < 0) | (self.expected == group))
            quality = height_score[:, group] * (1. - .5 * sync[:, group] / c.stair_step_sync_time).clamp(0., 1.)
            self.score += eligible * quality / dt  # Cancel framework dt: one event budget.
            success = eligible & (height_score[:, group] > 0)
            self.stats['valid_events'] += success
            self.stats['sync_sum_s'] += success * sync[:, group]
            self.stats['peak_sum_m'] += success * self.peak[:, group].mean(dim=1)
            self.expected[success] = 1 - group
            self.elapsed[success] = 0
        self.active &= ~(landed | (self.age > c.stair_step_overdue_time))
        self.overdue[:] = enabled * ((self.elapsed - c.stair_step_overdue_time) / c.stair_step_overdue_time).clamp(0., 1.)
        self.stats['enabled_seconds'] += enabled * dt
        self.stats['overdue_seconds'] += (self.overdue > 0) * dt

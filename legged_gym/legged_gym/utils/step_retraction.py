"""One-shot, command-opposed foot retraction after a stair-riser collision."""
import math
import torch


class StepRetraction:
    IDLE, RETRACT, COMPLETE, TIMED_OUT = range(4)

    def __init__(self, num_envs, num_feet, device, cfg, dt):
        self.cfg, self.dt = cfg, dt
        for name in ('distance', 'trigger_distance', 'force', 'confirm_time', 'timeout', 'progress_epsilon'):
            if not math.isfinite(getattr(cfg, 'step_retraction_' + name)) or getattr(cfg, 'step_retraction_' + name) <= 0:
                raise ValueError('Invalid step_retraction_' + name)
        shape = (num_envs, num_feet)
        self.stage = torch.zeros(shape, dtype=torch.long, device=device)
        self.contact_time = torch.zeros(shape, device=device)
        self.elapsed = torch.zeros(shape, device=device)
        self.origin = torch.zeros(*shape, 2, device=device)
        self.direction = torch.zeros_like(self.origin)
        self.best = torch.zeros(shape, device=device)
        self.progress_reward = torch.zeros(num_envs, device=device)
        self.stall_cost = torch.zeros(num_envs, device=device)

    def reset(self, ids):
        for value in (self.stage, self.contact_time, self.elapsed, self.origin,
                      self.direction, self.best, self.progress_reward, self.stall_cost):
            value[ids] = 0

    def update(self, feet, forces, clearance):
        c = self.cfg
        active = clearance.active
        # One attempt per latched edge, not one per contact pulse.
        self.stage.masked_fill_(~active, self.IDLE)
        self.contact_time.masked_fill_(~active, 0.)
        self.elapsed.masked_fill_(~active, 0.)
        self.best.masked_fill_(~active, 0.)
        xy = feet[..., :2]
        sole = feet[..., 2] - c.step_clearance_foot_radius
        low = sole < clearance.target_height
        distance = ((clearance.edge - xy) * clearance.direction).sum(-1)
        opposing_force = -(forces[..., :2] * clearance.direction).sum(-1)
        blocked = active & low & (distance >= -c.step_clearance_foot_radius) & (distance <= c.step_retraction_trigger_distance) & (opposing_force >= c.step_retraction_force)
        self.contact_time[:] = torch.where(blocked, self.contact_time + self.dt, torch.zeros_like(self.contact_time))
        start = (self.stage == self.IDLE) & (self.contact_time >= c.step_retraction_confirm_time)
        self.stage[start] = self.RETRACT
        self.origin[:] = torch.where(start[..., None], xy, self.origin)
        self.direction[:] = torch.where(start[..., None], clearance.direction, self.direction)
        self.elapsed[start] = 0.
        self.best[start] = 0.
        retract = self.stage == self.RETRACT
        # World-frame foot displacement against the latched TARGET velocity.
        # Neither body displacement nor current body yaw contributes to reward.
        displacement = (-(xy - self.origin) * self.direction).sum(-1).clamp(0, c.step_retraction_distance)
        increment = (displacement - self.best).clamp_min(0.) * retract
        self.best[:] = torch.where(retract, torch.maximum(self.best, displacement), self.best)
        # Env multiplies reward scales by dt: integrate to <= 1 per foot/edge.
        self.progress_reward[:] = (increment / (c.step_retraction_distance * self.dt)).sum(-1)
        self.elapsed += retract * self.dt
        complete = retract & ((self.best >= c.step_retraction_distance - 1e-6) | ~low)
        timeout = retract & ~complete & (self.elapsed >= c.step_retraction_timeout)
        self.stage[complete] = self.COMPLETE
        self.stage[timeout] = self.TIMED_OUT
        # No penalty on the trigger sample (there was no opportunity to act).
        stalled = ((self.stage == self.RETRACT) & ~start & (increment <= c.step_retraction_progress_epsilon)) | (self.stage == self.TIMED_OUT)
        self.stall_cost[:] = (stalled * low * (1 - self.best / c.step_retraction_distance)).sum(-1)
        # Allow brief unloading before asking for height. After timeout, restore
        # clearance cost; keep stall cost until this edge is cleared/cancelled.
        clearance.per_foot_cost.masked_fill_(self.stage == self.RETRACT, 0.)
        clearance.cost[:] = clearance.per_foot_cost.sum(-1)

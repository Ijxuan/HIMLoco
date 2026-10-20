"""Bounded straight-swing retraction with reverse body travel removed from credit."""
import math
import torch


class StepRetraction:
    IDLE, RETRACT, COMPLETE, TIMED_OUT = range(4)

    def __init__(self, num_envs, num_feet, device, cfg, dt):
        self.cfg, self.dt = cfg, dt
        names = ('distance', 'max_distance', 'reference_time', 'timeout', 'min_forward_speed',
                 'max_yaw_rate', 'stance_confirm_time', 'swing_confirm_time',
                 'landing_confirm_time', 'contact_force', 'landing_penalty_time')
        for name in names:
            value = getattr(cfg, 'step_retraction_' + name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid step_retraction_' + name)
        if not math.isfinite(dt) or dt <= 0 or cfg.step_retraction_timeout < cfg.step_retraction_reference_time:
            raise ValueError('Invalid retraction timing')
        if cfg.step_retraction_max_distance <= cfg.step_retraction_distance:
            raise ValueError("Retraction maximum must exceed minimum")
        shape = (num_envs, num_feet)
        self.stage = torch.zeros(shape, dtype=torch.long, device=device)
        self.armed = torch.zeros(shape, dtype=torch.bool, device=device)
        self.initialized = torch.zeros(shape, dtype=torch.bool, device=device)
        self.stance_time = torch.zeros(shape, device=device)
        self.air_time = torch.zeros(shape, device=device)
        self.landing_time = torch.zeros(shape, device=device)
        self.elapsed = torch.zeros(shape, device=device)
        self.origin = torch.zeros(*shape, 2, device=device)
        self.direction = torch.zeros_like(self.origin)
        self.command_direction = torch.zeros_like(self.origin)
        self.previous_xy = torch.zeros_like(self.origin)
        self.previous_base_xy = torch.zeros(num_envs, 2, device=device)
        self.body_backward = torch.zeros(shape, device=device)
        self.world_displacement = torch.zeros(shape, device=device)
        self.world_best = torch.zeros(shape, device=device)
        self.over_cost = torch.zeros(shape, device=device)
        self.effective_peak = torch.zeros(shape, device=device)
        self.best = torch.zeros(shape, device=device)
        self.displacement = torch.zeros(shape, device=device)
        self.per_foot_lag_cost = torch.zeros(shape, device=device)
        self.started = torch.zeros(shape, dtype=torch.bool, device=device)
        self.completed = torch.zeros(shape, dtype=torch.bool, device=device)
        self.failed_landing = torch.zeros(shape, dtype=torch.bool, device=device)
        self.progress_reward = torch.zeros(num_envs, device=device)
        # Retain the logging name for lag, overshoot and terminal band penalties.
        self.stall_cost = torch.zeros(num_envs, device=device)
        self.stats = {k: torch.zeros(shape, device=device) for k in (
            'started', 'count', 'under', 'over', 'in_band', 'timeout', 'cancelled',
            'world_peak_sum_cm', 'effective_peak_sum_cm', 'body_backward_sum_cm',
            'lag_integral', 'over_integral', 'landing_integral', 'progress_sum', 'enabled_seconds')}

    def reset(self, ids):
        self.stats["cancelled"][ids] += self.stage[ids] != self.IDLE
        for value in (self.stage, self.armed, self.initialized, self.stance_time,
                      self.air_time, self.landing_time, self.elapsed, self.origin,
                      self.direction, self.command_direction, self.previous_xy, self.best, self.displacement,
                      self.previous_base_xy, self.body_backward, self.world_displacement,
                      self.world_best, self.effective_peak, self.over_cost, self.per_foot_lag_cost, self.started, self.completed,
                      self.failed_landing, self.progress_reward, self.stall_cost):
            value[ids] = 0

    def update(self, feet, commands, command_world_xy, support, base_xy):
        """Update once per control step, with terrain-independent swing state.

        support identifies horizontal support, not merely a large vertical force
        (wall friction can also have a vertical component). The caller may use
        privileged terrain geometry for this TRAINING reward, never actor input.
        base_xy is the trunk root position in world coordinates, shape (N, 2).
        """
        c = self.cfg
        xy = feet[..., :2]
        speed = command_world_xy.norm(dim=-1)
        desired_direction = command_world_xy / speed[:, None].clamp_min(1e-6)
        input_direction = commands[:, :2] / commands[:, :2].norm(dim=-1, keepdim=True).clamp_min(1e-6)
        enabled = ((commands[:, 0].abs() >= c.step_retraction_min_forward_speed)
                   & (commands[:, 2].abs() < c.step_retraction_max_yaw_rate))[:, None]
        self.started.zero_(); self.completed.zero_(); self.failed_landing.zero_()
        self.progress_reward.zero_(); self.stall_cost.zero_(); self.per_foot_lag_cost.zero_()
        # Turning, stopping or reversing cancels the old direction. Re-arming
        # requires stable support; a mid-air command switch is not a new swing.
        cycle = self.stage != self.IDLE
        # Cancel on an external command change, not on the robot yawing.
        aligned = (self.command_direction * input_direction[:, None]).sum(-1) > .5
        cancel = (~enabled.expand_as(cycle) | (cycle & ~aligned)
                  | (self.armed & ~aligned & (self.air_time > 0)))
        self.stats["cancelled"] += cancel & cycle
        self.stats["enabled_seconds"] += enabled * self.dt
        self.effective_peak.masked_fill_(cancel, 0.)
        self.stage.masked_fill_(cancel, self.IDLE)
        self.armed.masked_fill_(cancel, False)
        self.air_time.masked_fill_(cancel, 0.)
        self.stance_time.masked_fill_(cancel, 0.)
        self.landing_time.masked_fill_(cancel, 0.)
        self.elapsed.masked_fill_(cancel, 0.)
        self.best.masked_fill_(cancel, 0.)
        self.displacement.masked_fill_(cancel, 0.)
        self.body_backward.masked_fill_(cancel, 0.)
        self.world_best.masked_fill_(cancel, 0.)
        eligible = enabled & ~cancel

        idle = self.stage == self.IDLE
        self.stance_time[:] = torch.where(idle & support & eligible,
            self.stance_time + self.dt, torch.zeros_like(self.stance_time))
        self.armed |= idle & (self.stance_time + 1e-7 >= c.step_retraction_stance_confirm_time)
        first_air = idle & self.armed & ~support & (self.air_time == 0) & eligible
        # Capture the LAST support sample, not the delayed confirmation sample.
        self.origin[:] = torch.where(first_air[..., None], self.previous_xy, self.origin)
        self.direction[:] = torch.where(first_air[..., None], desired_direction[:, None], self.direction)
        self.command_direction[:] = torch.where(first_air[..., None], input_direction[:, None], self.command_direction)
        self.best[first_air] = 0.
        self.world_best[first_air] = 0.
        self.effective_peak[first_air] = 0.
        self.body_backward[first_air] = 0.
        self.elapsed[first_air] = 0.
        self.air_time[:] = torch.where(idle & self.armed & ~support & eligible,
            self.air_time + self.dt, torch.zeros_like(self.air_time))
        self.started[:] = idle & self.armed & (self.air_time + 1e-7 >= c.step_retraction_swing_confirm_time) & eligible & self.initialized
        self.stats["started"] += self.started
        self.stage[self.started] = self.RETRACT
        self.elapsed[:] = torch.where(self.started, (self.air_time - self.dt).clamp_min(0.), self.elapsed)
        self.armed[self.started] = False

        cycle = self.stage != self.IDLE
        unfinished = ((self.stage == self.RETRACT) | (self.stage == self.TIMED_OUT)) & eligible
        self.elapsed += unfinished * self.dt
        # Accumulate only reverse body travel along each foot's locked world axis.
        # Include the first unconfirmed air samples; forward recovery cannot refund it.
        tracking = eligible & (cycle | (idle & self.armed & ~support))
        body_step = -((base_xy - self.previous_base_xy)[:, None] * self.direction).sum(-1)
        self.body_backward += body_step.clamp_min(0.) * tracking * self.initialized
        world_signed = (-(xy - self.origin) * self.direction).sum(-1)
        self.world_displacement[:] = torch.where(cycle, world_signed, torch.zeros_like(world_signed))
        signed = world_signed - self.body_backward
        self.effective_peak[:] = torch.where(cycle & eligible & ~support,
            torch.maximum(self.effective_peak, signed.clamp_min(0.)), self.effective_peak)
        # Upper bound remains in the WORLD frame, so body retreat cannot hide overshoot.
        self.world_best[:] = torch.where(tracking & ~support,
            torch.maximum(self.world_best, world_signed.clamp_min(0.)), self.world_best)
        self.over_cost[:] = ((self.world_best - c.step_retraction_max_distance)
            / (c.step_retraction_max_distance - c.step_retraction_distance)).clamp(0, 1).square()
        self.over_cost *= cycle & eligible
        self.displacement[:] = torch.where(cycle, signed, torch.zeros_like(signed))
        distance = signed.clamp(0, c.step_retraction_distance)
        increment = (distance - self.best).clamp_min(0.) * unfinished * ~support
        self.best[:] = torch.where(unfinished & ~support, torch.maximum(self.best, distance), self.best)
        self.progress_reward[:] = (increment / (c.step_retraction_distance * self.dt)).sum(-1)
        self.completed[:] = unfinished & (self.best >= c.step_retraction_distance - 1e-6)
        self.stage[self.completed] = self.COMPLETE
        late = unfinished & ~self.completed & (self.elapsed + 1e-7 >= c.step_retraction_timeout)
        self.stats["timeout"] += late & (self.stage == self.RETRACT)
        self.stage[late] = self.TIMED_OUT
        reference = c.step_retraction_distance * (self.elapsed / c.step_retraction_reference_time).clamp(0, 1)
        lag = ((reference - signed) / c.step_retraction_distance).clamp(0, 1).square()
        self.per_foot_lag_cost[:] = lag * unfinished * ~self.completed

        self.landing_time[:] = torch.where(cycle & support, self.landing_time + self.dt,
                                            torch.zeros_like(self.landing_time))
        landed = cycle & (self.landing_time + 1e-7 >= c.step_retraction_landing_confirm_time)
        self.failed_landing[:] = landed & ((self.stage != self.COMPLETE) | (self.over_cost > 0))
        # A brief swing cannot erase an incomplete task by touching down early.
        deficit = (1 - self.best / c.step_retraction_distance).clamp(0, 1).square()
        terminal = landed * (deficit + self.over_cost) * (c.step_retraction_landing_penalty_time / self.dt)
        self.stall_cost[:] = (self.per_foot_lag_cost + self.over_cost + terminal).sum(-1)
        self.stats['count'] += landed
        self.stats['under'] += landed & (self.stage != self.COMPLETE)
        self.stats['over'] += landed & (self.over_cost > 0)
        self.stats['in_band'] += landed & ~self.failed_landing
        self.stats['world_peak_sum_cm'] += landed * self.world_best * 100
        self.stats['effective_peak_sum_cm'] += landed * self.effective_peak * 100
        self.stats['body_backward_sum_cm'] += landed * self.body_backward * 100
        self.stats['lag_integral'] += self.per_foot_lag_cost * self.dt
        self.stats['over_integral'] += self.over_cost * self.dt
        self.stats['landing_integral'] += terminal * self.dt
        self.stats['progress_sum'] += increment / c.step_retraction_distance
        self.stage[landed] = self.IDLE
        self.armed[landed] = False
        self.stance_time[landed] = 0.
        self.elapsed[landed] = 0.
        self.landing_time[landed] = 0.
        self.previous_xy[:] = xy
        self.previous_base_xy[:] = base_xy
        self.initialized[:] = True

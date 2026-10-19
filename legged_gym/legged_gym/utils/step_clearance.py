"""Privileged, stateful stair-edge clearance cost; no policy observation changes."""
import math
import torch


class StepClearance:
    def __init__(self, num_envs, num_feet, device, cfg):
        self.cfg = cfg
        for name in ('lookahead', 'sample_spacing', 'margin', 'foot_radius',
                     'min_rise', 'support_tolerance', 'release_distance', 'min_speed'):
            value = getattr(cfg, 'step_clearance_' + name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid step_clearance_' + name)
        shape = (num_envs, num_feet)
        self.active = torch.zeros(shape, dtype=torch.bool, device=device)
        self.target_height = torch.zeros(shape, device=device)
        self.edge = torch.zeros(*shape, 2, device=device)
        self.direction = torch.zeros_like(self.edge)
        self.cost = torch.zeros(num_envs, device=device)
        self.per_foot_cost = torch.zeros(shape, device=device)
        count = math.ceil(cfg.step_clearance_lookahead / cfg.step_clearance_sample_spacing)
        self.offsets = torch.linspace(0, cfg.step_clearance_lookahead, count + 1, device=device)

    def reset(self, ids):
        for value in (self.active, self.target_height, self.edge, self.direction,
                      self.cost, self.per_foot_cost):
            value[ids] = 0

    def update(self, feet, forces, command_world_xy, sample_height):
        """Call once per physics/control step. sample_height accepts (...,2)."""
        c = self.cfg
        speed = command_world_xy.norm(dim=-1)
        direction = command_world_xy / speed[:, None].clamp_min(1e-6)
        moving = speed[:, None] >= c.step_clearance_min_speed
        sole = feet[..., 2] - c.step_clearance_foot_radius
        xy = feet[..., :2]
        points = xy[:, :, None] + direction[:, None, None] * self.offsets[None, None, :, None]
        heights = sample_height(points)
        ground = heights[..., 0]
        # Vertical wall friction must not masquerade as a horizontal support.
        support = ((sole - ground).abs() <= c.step_clearance_support_tolerance) & (forces[..., 2] > 1.)
        progress = ((xy - self.edge) * self.direction).sum(-1)
        aligned = (self.direction * direction[:, None]).sum(-1) > .5
        released = progress >= c.step_clearance_foot_radius + c.step_clearance_release_distance
        landed = support & (ground >= self.target_height - c.step_clearance_margin - .01)
        cancelled = (~moving) | (~aligned) | (progress < -2 * c.step_clearance_lookahead)
        finished = self.active & (released | landed | cancelled)
        self.active &= ~finished

        # First discrete upward edge, not the maximum over multiple treads.
        rise = heights[..., 1:] - heights[..., :-1]
        edges = (rise >= c.step_clearance_min_rise) & (heights[..., 1:] >= ground[..., None] + c.step_clearance_min_rise)
        found = edges.any(-1)
        index = edges.to(torch.int64).argmax(-1) + 1
        distance = self.offsets[index]
        next_height = heights.gather(-1, index[..., None]).squeeze(-1)
        start = (~self.active) & (~finished) & moving & (~support) & found
        self.edge[:] = torch.where(start[..., None], xy + direction[:, None] * distance[..., None], self.edge)
        self.direction[:] = torch.where(start[..., None], direction[:, None], self.direction)
        self.target_height[:] = torch.where(start, next_height + c.step_clearance_margin, self.target_height)
        self.active |= start
        progress = ((xy - self.edge) * self.direction).sum(-1)
        # Smooth approach weighting and a fade once the whole foot crosses.
        approach = (1 + progress / c.step_clearance_lookahead).clamp(0, 1)
        release = (1 - (progress - c.step_clearance_foot_radius).clamp_min(0) / c.step_clearance_release_distance).clamp(0, 1)
        deficit = ((self.target_height - sole) / c.step_clearance_margin).clamp(0, 1)
        self.per_foot_cost[:] = self.active * approach * release * deficit.square()
        self.cost[:] = self.per_foot_cost.sum(-1)

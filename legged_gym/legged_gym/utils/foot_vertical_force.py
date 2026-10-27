"""Substep world-Z foot-force cost; returned rate is scaled by control dt once."""
import math
import torch


class FootVerticalForce:
    def __init__(self, num_envs, device, cfg, decimation):
        self.flat = float(cfg.foot_vertical_force_flat_threshold)
        self.stairs = float(cfg.foot_vertical_force_stair_threshold)
        self.flat_angle = float(cfg.foot_vertical_force_flat_angle_deg)
        self.stair_angle = float(cfg.foot_vertical_force_stair_angle_deg)
        self.min_speed = float(cfg.foot_vertical_force_min_forward_speed)
        values = (self.flat, self.stairs, self.flat_angle,
                  self.stair_angle, self.min_speed)
        if (not all(math.isfinite(v) for v in values)
                or not 0 < self.flat <= self.stairs
                or not 0 <= self.flat_angle < self.stair_angle < 90
                or self.min_speed <= 0 or decimation < 1):
            raise ValueError("Invalid foot_vertical_force configuration")
        self.decimation = decimation
        self.cost = torch.zeros(num_envs, device=device)
        self.threshold = torch.full_like(self.cost, self.flat)
        self.target_uphill_deg = torch.zeros_like(self.cost)
        self.active = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def begin(self, target_normal_world, base_quat_xyzw, commands):
        """Freeze threshold and command gate at action start for all substeps.

        Uses the last available filtered terrain normal. Actual body pitch does
        not select the threshold; yaw only supplies the forward world direction.
        """
        x, y, z, w = base_quat_xyzw.unbind(-1)
        yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y*y + z*z))
        normal = target_normal_world
        forward_normal = normal[:, 0] * yaw.cos() + normal[:, 1] * yaw.sin()
        self.target_uphill_deg.copy_(torch.atan2(-forward_normal, normal[:, 2])
                                     * (180. / math.pi))
        alpha = ((self.target_uphill_deg - self.flat_angle)
                 / (self.stair_angle - self.flat_angle)).clamp(0., 1.)
        self.threshold.copy_(self.flat + (self.stairs - self.flat) * alpha)
        self.active.copy_(commands[:, 0] >= self.min_speed)
        self.cost.zero_()

    def accumulate(self, foot_force_world):
        """Call after each simulate/force refresh; only positive Fz excess counts.

        mean_substeps(sum_feet(relu(Fz - threshold)^2)), units N^2.
        No dt here: the reward framework multiplies by control dt once.
        """
        excess = (foot_force_world[..., 2] - self.threshold[:, None]).clamp_min(0.)
        self.cost.add_(excess.square().sum(-1) * self.active / self.decimation)

    def reset(self, env_ids):
        self.cost[env_ids] = 0.
        self.threshold[env_ids] = self.flat
        self.target_uphill_deg[env_ids] = 0.
        self.active[env_ids] = False

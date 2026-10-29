"""Phase-conditioned stair marching, with per-control-step costs.

The two clock values are appended to the actor's proprioception. Zero/zero means
inactive; active values are sin/cos of phase. Clock state advances exactly once
AFTER scoring the action taken from that clock, never in observation accessors.
"""
import math
import torch


class StairGait:
    def __init__(self, num_envs, feet_names, device, cfg, dt):
        self.cfg, self.dt = cfg, dt
        names = ('FL_foot', 'FR_foot', 'RL_foot', 'RR_foot')
        if len(feet_names) != 4 or any(feet_names.count(n) != 1 for n in names):
            raise ValueError('stair_gait requires four named Mini Cheetah feet')
        positive = ('frequency', 'height', 'force_sigma', 'velocity_sigma')
        if (not math.isfinite(dt) or dt <= 0 or
                any(not math.isfinite(getattr(cfg, 'stair_gait_' + k)) or
                    getattr(cfg, 'stair_gait_' + k) <= 0 for k in positive) or
                not .5 <= cfg.stair_gait_duty_factor < 1. or
                not 0 < cfg.stair_step_off_deg < cfg.stair_step_on_deg < 90 or
                dt * cfg.stair_gait_frequency >= (1 - cfg.stair_gait_duty_factor) / 2):
            raise ValueError('Invalid stair gait timing, thresholds, or scales')
        self.offsets = torch.tensor([0. if n in ('FL_foot', 'RR_foot') else .5
                                     for n in feet_names], device=device)
        self.diagonals = torch.tensor([[feet_names.index(n) for n in pair] for pair in
                                       (('FL_foot', 'RR_foot'), ('FR_foot', 'RL_foot'))],
                                      device=device)
        self.phase = torch.zeros(num_envs, device=device)
        self.enabled = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.slope_mode = torch.zeros_like(self.enabled)
        self.clock = torch.zeros(num_envs, 2, device=device)
        self.swing_weight = torch.zeros(num_envs, 4, device=device)
        self.target_height = torch.zeros_like(self.swing_weight)
        self.force_cost = torch.zeros_like(self.phase)
        self.height_cost = torch.zeros_like(self.phase)
        self.velocity_cost = torch.zeros_like(self.phase)
        self.pair_deficit_cost = torch.zeros_like(self.phase)
        self.stats = {name: torch.zeros_like(self.phase) for name in
                      ('enabled_seconds', 'force_cost_integral', 'height_cost_integral',
                       'velocity_cost_integral', 'pair_deficit_cost_integral')}

    def reset(self, ids):
        # Episode stats are consumed by the environment's reset logger.
        for name in ('phase', 'enabled', 'slope_mode', 'clock', 'swing_weight',
                     'target_height', 'force_cost', 'height_cost', 'velocity_cost',
                     'pair_deficit_cost'):
            getattr(self, name)[ids] = 0

    def targets(self):
        """Pure targets for the CURRENT observation clock, in asset foot order."""
        c = self.cfg
        phase = torch.remainder(self.phase[:, None] + self.offsets, 1.)
        progress = ((phase - c.stair_gait_duty_factor) / (1 - c.stair_gait_duty_factor)).clamp(0., 1.)
        wave = torch.sin(math.pi * progress).clamp_min(0.)
        return wave.square(), c.stair_gait_height * wave

    def update(self, normal, commands, clearance, forces, velocities):
        c = self.cfg
        angle = torch.atan2(normal[:, :2].norm(dim=1), normal[:, 2])
        self.slope_mode = ((angle > math.radians(c.stair_step_on_deg)) |
                           (self.slope_mode & (angle >= math.radians(c.stair_step_off_deg))))
        # Terrain enables the clock at every command, including standstill.
        # Keep commands in the shared environment update interface.
        requested = self.slope_mode
        # An activation step cannot score a clock the actor has not observed yet.
        scoring = self.enabled & requested
        swing, height = self.targets()
        self.swing_weight[:] = swing
        self.target_height[:] = height
        self.force_cost[:] = scoring * (swing * (
            1 - torch.exp(-forces.square().sum(-1) / c.stair_gait_force_sigma**2))).mean(-1)
        self.height_cost[:] = scoring * (swing * (
            (clearance - height) / c.stair_gait_height).square().clamp(max=1.)).mean(-1)
        self.velocity_cost[:] = scoring * ((1 - swing) * (
            1 - torch.exp(-velocities.square().sum(-1) / c.stair_gait_velocity_sigma**2))).mean(-1)
        # Excess lift cannot compensate for a partner left on the ground.
        # Existing height tracking still penalizes excess clearance.
        height_deficit = ((height - clearance) / c.stair_gait_height).clamp(0., 1.).square()
        force_deficit = 1 - torch.exp(-forces.square().sum(-1) / c.stair_gait_force_sigma**2)
        deficit = .5 * (height_deficit + force_deficit)
        # Named partners have identical swing weights. Sum the two groups;
        # duty >= .5 prevents swing overlap, bounding this cost by 1.
        # Like existing costs, score the phase observed before the action.
        self.pair_deficit_cost[:] = scoring * (
            swing[:, self.diagonals[:, 0]] * deficit[:, self.diagonals].amax(dim=-1)).sum(-1)
        self.stats['enabled_seconds'] += scoring * self.dt
        for name in ('force', 'height', 'velocity', 'pair_deficit'):
            self.stats[name + '_cost_integral'] += getattr(self, name + '_cost') * self.dt
        # First active observation is phase zero. Following observations advance
        # by f*dt independently of translation speed, including cmd == 0.
        continuing = self.enabled & requested
        self.phase[:] = torch.where(continuing, torch.remainder(
            self.phase + c.stair_gait_frequency * self.dt, 1.), 0.)
        self.enabled[:] = requested
        self.clock[:, 0] = requested * torch.sin(2 * math.pi * self.phase)
        self.clock[:, 1] = requested * torch.cos(2 * math.pi * self.phase)

    def contract(self):
        return {
            'version': 'minich_stair_clock_v2', 'frame_dim': 47, 'history_frames': 6,
            'clock_indices': [45, 46], 'clock': ['sin(2*pi*phase)', 'cos(2*pi*phase)'],
            'inactive_clock': [0., 0.], 'activation_phase': 0.,
            'frequency_hz': self.cfg.stair_gait_frequency,
            'duty_factor': self.cfg.stair_gait_duty_factor,
            'height_m': self.cfg.stair_gait_height,
            'tilt_on_deg': self.cfg.stair_step_on_deg,
            'tilt_off_deg': self.cfg.stair_step_off_deg,
            'activation': 'terrain_tilt_hysteresis_all_commands',
            'policy_dt': self.dt, 'phase_update': 'once after action; reset on deactivation',
            'tilt_source': 'filtered terrain target normal in world frame, NOT actual body tilt',
        }

# SPDX-License-Identifier: BSD-3-Clause
"""Three-policy Mini Cheetah playback on twenty exact 8 cm / 20 cm stairs.

From this directory:
    conda activate gym4
    python -u play_minich_stairs.py --task=minich

CPU smoke playback:
    python -u play_minich_stairs.py --headless --sim_device=cpu --pipeline=cpu --rl_device=cpu
"""
from types import SimpleNamespace

import play_minich_flat_height as flat  # Import Isaac Gym before torch.
import numpy as np
import torch

from legged_gym.envs.base.legged_robot import LeggedRobot
from legged_gym.utils import get_args, task_registry

# Same selection as the flat viewer. Each network controls one robot only.
POLICY_SPECS = (
    ("manual_1", "Sep11_10-19-10_smoke_minich", 1000),
    ("manual_2", -2, -1),
    ("latest", -1, -1),
)
STAIR_START_X_M = 1.0
STAIR_HEIGHT_M = 0.08
STAIR_DEPTH_M = 0.20
STAIR_COUNT = 20
ENV_SPACING_M = 1.0
INITIAL_BASE_HEIGHT_M = flat.INITIAL_BASE_HEIGHT_M
ACTION_SCALE = flat.ACTION_SCALE
FORWARD_SPEED_M_S = 1.0
STAND_BEFORE_S = 3.0
FORWARD_S = 10.0
STAND_AFTER_S = 6.0
HEIGHT_PRINT_INTERVAL_S = flat.HEIGHT_PRINT_INTERVAL_S
IMU_PRINT_INTERVAL_S = flat.IMU_PRINT_INTERVAL_S
FOOT_FORCE_PRINT_INTERVAL_S = flat.FOOT_FORCE_PRINT_INTERVAL_S

# A wide shared staircase keeps all three lanes at identical geometry.
# Beyond the last 20 cm tread, extend the landing so walking can continue.
GROUND_BACK_M = 4.0
TOP_LANDING_END_X_M = 15.0
HEIGHT_SAMPLE_RESOLUTION_M = 0.02
VERTICAL_RESOLUTION_M = 0.005
CAMERA_OFFSET = np.array([-3.0, -3.0, 2.0])
CAMERA_LOOKAHEAD = np.array([2.0, 0.0, 0.65])


def stair_height_at_x(x):
    """World ground height: flat up to x=1, then exactly twenty rises."""
    x = np.asarray(x, dtype=np.float64)
    # Tolerance only stabilizes exact tread boundaries in floating point.
    level = np.floor((x - STAIR_START_X_M) / STAIR_DEPTH_M + 1e-9) + 1
    level = np.where(x < STAIR_START_X_M, 0, level)
    return np.clip(level, 0, STAIR_COUNT) * STAIR_HEIGHT_M


def build_stair_terrain(cfg, num_envs):
    """Exact horizontal treads and vertical risers, plus a sensing height grid.

    The collision mesh is constructed directly rather than converting a sampled
    heightfield, so 20 cm treads and vertical 8 cm faces stay exact.
    Mesh coordinates include border_size for the inherited mesh transform.
    """
    y_min = -GROUND_BACK_M
    y_max = (num_envs - 1) * ENV_SPACING_M + GROUND_BACK_M
    vertices, triangles = [], []

    def quad(corners):
        first = len(vertices)
        vertices.extend(corners)
        triangles.extend(((first, first + 1, first + 2),
                          (first, first + 2, first + 3)))

    def tread(x0, x1, height):
        # Upward-facing horizontal surface.
        quad(((x0, y_min, height), (x1, y_min, height),
              (x1, y_max, height), (x0, y_max, height)))

    tread(-GROUND_BACK_M, STAIR_START_X_M, 0.0)
    for step in range(STAIR_COUNT):
        x0 = STAIR_START_X_M + step * STAIR_DEPTH_M
        x1 = STAIR_START_X_M + (step + 1) * STAIR_DEPTH_M
        z0, z1 = step * STAIR_HEIGHT_M, (step + 1) * STAIR_HEIGHT_M
        tread(x0, x1, z1)
        # Riser normal faces the approaching robot (-X).
        quad(((x0, y_min, z0), (x0, y_min, z1),
              (x0, y_max, z1), (x0, y_max, z0)))
    stair_end = STAIR_START_X_M + STAIR_COUNT * STAIR_DEPTH_M
    tread(stair_end, TOP_LANDING_END_X_M, STAIR_COUNT * STAIR_HEIGHT_M)
    vertices = np.asarray(vertices, dtype=np.float32)
    vertices[:, :2] += cfg.border_size

    # The inherited height sensing indexes world positions with border_size.
    rows = int(round((TOP_LANDING_END_X_M + GROUND_BACK_M) / cfg.horizontal_scale)) + 1
    cols = int(round((y_max + GROUND_BACK_M) / cfg.horizontal_scale)) + 1
    sample_x = np.arange(rows) * cfg.horizontal_scale - cfg.border_size
    profile = np.rint(stair_height_at_x(sample_x) / cfg.vertical_scale).astype(np.int16)
    heights = np.repeat(profile[:, None], cols, axis=1)
    return SimpleNamespace(
        cfg=cfg, vertices=vertices,
        triangles=np.asarray(triangles, dtype=np.uint32),
        heightsamples=heights, tot_rows=rows, tot_cols=cols,
    )


class StairPlaybackRobot(LeggedRobot):
    """Playback-only terrain; training environment and reward methods inherited."""

    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(
            self.sim_device_id, self.graphics_device_id,
            self.physics_engine, self.sim_params,
        )
        self.terrain = build_stair_terrain(self.cfg.terrain, self.num_envs)
        self._create_trimesh()
        self._create_envs()

    def _get_env_origins(self):
        # All dogs face +X and start at x=0. Parallel Y lanes have the same
        # distance to the first riser, unlike the flat viewer's X-axis lineup.
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self.env_origins[:, 1] = torch.arange(self.num_envs, device=self.device) * ENV_SPACING_M


def configure_stair_env(cfg):
    flat.configure_flat_sequence_env(cfg)
    cfg.env.num_envs = len(POLICY_SPECS)
    cfg.env.episode_length_s = max(20., STAND_BEFORE_S + FORWARD_S + STAND_AFTER_S + 1.)
    cfg.init_state.pos[2] = INITIAL_BASE_HEIGHT_M
    cfg.control.action_scale = ACTION_SCALE
    cfg.terrain.mesh_type = "trimesh"
    cfg.terrain.curriculum = False
    cfg.terrain.horizontal_scale = HEIGHT_SAMPLE_RESOLUTION_M
    cfg.terrain.vertical_scale = VERTICAL_RESOLUTION_M
    cfg.terrain.border_size = GROUND_BACK_M
    cfg.commands.resampling_time = cfg.env.episode_length_s + 1.


def scheduled_phase(step, dt):
    before = int(round(STAND_BEFORE_S / dt))
    forward_end = before + int(round(FORWARD_S / dt))
    if step < before:
        return "stand_before"
    if step < forward_end:
        return "forward"
    return "stand_after"


def aim_camera(env):
    if env.viewer is None:
        return
    position = env.root_states[-1, :3].detach().cpu().numpy()
    env.set_camera(position + CAMERA_OFFSET, position + CAMERA_LOOKAHEAD)


def load_policies(env, args, train_cfg):
    policies = []
    original_load = torch.load

    def load_on_policy_device(*a, **kw):
        kw.setdefault("map_location", env.device)
        return original_load(*a, **kw)

    # The reference runner loads GPU-saved checkpoints without map_location.
    # Scope this adaptation to playback loading, including CPU smoke runs.
    torch.load = load_on_policy_device
    try:
        for label, run, checkpoint in POLICY_SPECS:
            train_cfg.runner.resume = True
            train_cfg.runner.load_run = run
            train_cfg.runner.checkpoint = checkpoint
            runner, _ = task_registry.make_alg_runner(
                env=env, name="minich", args=args, train_cfg=train_cfg,
            )
            policies.append(runner.get_inference_policy(device=env.device))
            print(f"已加载策略 {label}: run={run}, checkpoint={checkpoint}", flush=True)
    finally:
        torch.load = original_load
    return policies


@torch.no_grad()
def play(args):
    if len(POLICY_SPECS) != 3 or POLICY_SPECS[-1][0] != "latest":
        raise ValueError("Specify three policies, with latest in the last slot")
    if args.rl_device == "cpu":
        torch.set_num_threads(1)
    args.task = "minich"
    args.num_envs = 3
    args.load_run = args.checkpoint = None
    cfg, train_cfg = task_registry.get_cfgs("minich")
    configure_stair_env(cfg)
    task_name = "minich_stairs_playback"
    task_registry.register(task_name, StairPlaybackRobot, cfg, train_cfg)
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=cfg)
    original_specs = flat.POLICY_SPECS
    flat.POLICY_SPECS = POLICY_SPECS  # Shared printers/meters use these labels.
    try:
        flat.install_fixed_reset(env)
        flat.reset_to_fixed_state(env)
        flat.install_termination_override(env)
        policies = load_policies(env, args, train_cfg)
        flat.reset_to_fixed_state(env)
        meter = flat.install_latest_policy_reward_meter(env)
        aim_camera(env)
        steps = int(round((STAND_BEFORE_S + FORWARD_S + STAND_AFTER_S) / env.dt))
        height_interval = max(1, int(round(HEIGHT_PRINT_INTERVAL_S / env.dt))) if HEIGHT_PRINT_INTERVAL_S > 0 else None
        imu_interval = max(1, int(round(IMU_PRINT_INTERVAL_S / env.dt))) if IMU_PRINT_INTERVAL_S > 0 else None
        forces = flat.FootContactForcePrinter(env, FOOT_FORCE_PRINT_INTERVAL_S) if FOOT_FORCE_PRINT_INTERVAL_S > 0 else None
        print(
            f"Stairs viewer: robots=3; first riser x={STAIR_START_X_M:.2f}m; "
            f"{STAIR_COUNT} steps, rise={STAIR_HEIGHT_M:.2f}m, tread={STAIR_DEPTH_M:.2f}m; "
            f"top={STAIR_COUNT * STAIR_HEIGHT_M:.2f}m; "
            f"sequence={STAND_BEFORE_S}s stand -> {FORWARD_S}s forward "
            f"at {FORWARD_SPEED_M_S}m/s -> {STAND_AFTER_S}s stand",
            flush=True,
        )
        for step in range(steps):
            phase = scheduled_phase(step, env.dt)
            meter.set_phase(phase)
            command = (FORWARD_SPEED_M_S, 0., 0.) if phase == "forward" else (0., 0., 0.)
            flat.set_command(env, command)
            env.compute_observations()
            obs = env.get_observations()
            actions = torch.cat([policy(obs[i:i+1].detach()) for i, policy in enumerate(policies)])
            env.step(actions.detach())
            if height_interval is not None and step % height_interval == 0:
                heights = flat.get_base_heights_above_ground(env)
                positions = env.root_states[:, 0].cpu().tolist()
                values = [f"{spec[0]}: x={x:.3f}m, world_z={world:.3f}m, above_ground={ground:.3f}m, error={error:+.3f}m"
                          for spec, x, (world, ground, error) in zip(POLICY_SPECS, positions, heights)]
                print(f"[台阶回放] t={(step+1)*env.dt:.2f}s, phase={phase}; " + " | ".join(values), flush=True)
                aim_camera(env)
            if imu_interval is not None and step % imu_interval == 0:
                angles = flat.get_base_imu_angles_deg(env)
                print(f"[IMU] t={(step+1)*env.dt:.2f}s; " + " | ".join(
                    f"{spec[0]}: roll={r:+.2f}, pitch={p:+.2f}, yaw={y:+.2f} deg"
                    for spec, (r, p, y) in zip(POLICY_SPECS, angles)), flush=True)
            if forces is not None:
                forces.sample(env, step, phase)
        meter.print_summary()
    finally:
        flat.POLICY_SPECS = original_specs
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    play(get_args())

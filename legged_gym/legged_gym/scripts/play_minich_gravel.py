# SPDX-License-Identifier: BSD-3-Clause
"""Mini Cheetah 固定粗糙地面评估（碎石近似，不模拟可移动石子）。

默认选 rough_minich 最新 run 的最高编号模型，保存解析后的路径及 SHA256。
    python -u play_minich_gravel.py --headless --sim_device cuda:0 --pipeline gpu
    python -u play_minich_gravel.py --speed .3 --load_run Nov07_10-58-27_smoke_minich

默认包含平地和训练难度 0/5/9 的水平随机起伏，每档三个独立地形种子。
去掉 --headless 可查看；评估不会训练、导出或修改模型。
"""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import play_minich_stairs as stairs  # Isaac Gym 必须先于 torch 导入。
from isaacgym import gymutil, terrain_utils
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import class_to_dict, get_load_path, task_registry
from rsl_rl.modules import HIMActorCritic

flat = stairs.flat
LANE_WIDTH = 3.0
BORDER = 2.0
ROUGH_START = 1.0
FINISH_X = 5.0
END_X = 16.0
STAND_S = 3.0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build_terrain(cfg, cases):
    """训练同款 uniform + 0.2 m 下采样；取消坡度以单独测起伏影响。"""
    nx = round((END_X + BORDER) / cfg.horizontal_scale) + 1
    ny = round((len(cases) * LANE_WIDTH + 2 * BORDER) / cfg.horizontal_scale) + 1
    heights = np.zeros((nx, ny), dtype=np.int16)
    random_state = np.random.get_state()
    try:
        for i, case in enumerate(cases):
            if case['amplitude_m'] == 0:
                continue
            np.random.seed(case['terrain_seed'])
            width = round(LANE_WIDTH / cfg.horizontal_scale)
            tile = terrain_utils.SubTerrain(
                'rough', width=nx, length=width, vertical_scale=cfg.vertical_scale,
                horizontal_scale=cfg.horizontal_scale)
            terrain_utils.random_uniform_terrain(
                tile, min_height=-case['amplitude_m'], max_height=case['amplitude_m'],
                step=.005, downsampled_scale=.2)
            first = round((ROUGH_START + BORDER) / cfg.horizontal_scale)
            tile.height_field_raw[:first] = 0
            start_y = round((BORDER + i * LANE_WIDTH) / cfg.horizontal_scale)
            heights[:, start_y:start_y + width] = tile.height_field_raw
    finally:
        np.random.set_state(random_state)
    vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(
        heights, cfg.horizontal_scale, cfg.vertical_scale, cfg.slope_treshold)
    return SimpleNamespace(cfg=cfg, vertices=vertices, triangles=triangles,
                           heightsamples=heights, tot_rows=nx, tot_cols=ny)


class GravelPlaybackRobot(stairs.StairPlaybackRobot):
    def create_sim(self):
        self.up_axis_idx = 2
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id,
                                       self.physics_engine, self.sim_params)
        self.terrain = build_terrain(self.cfg.terrain, self.cases)
        self._create_trimesh()
        self._create_envs()

    def _get_env_origins(self):
        self.custom_origins = False
        self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
        self.env_origins[:, 1] = (torch.arange(self.num_envs, device=self.device) + .5) * LANE_WIDTH


def score(samples, cases, dt, forward_s, speed):
    """仅在首次通过/跌倒/出界之前统计行走质量；另记全程跌倒。"""
    results = []
    moving = samples['command'][:, :, 0] > 0
    for i, case in enumerate(cases):
        valid = moving[:, i] & (samples['time'] > STAND_S)
        # 优先给出粗糙段的质量，入场失败时退回整个行走段并标明。
        rough = valid & (samples['position'][:, i, 0] >= ROUGH_START)
        selected = rough if rough.any() else valid
        vel = samples['velocity'][selected, i, 0]
        pos = samples['position'][:, i] - np.array([0., (i + .5) * LANE_WIDTH, 0.])
        angles = samples['rpy_deg'][selected, i]
        complete = np.flatnonzero(samples['completed'][:, i])
        fallen = np.flatnonzero(samples['fallen'][:, i])
        outside = bool(samples['outside'][:, i].any())
        # 所有预期测试均应有有效行走样本；站立期间失败直接失败，不填假数值。
        def mean_or_none(x):
            return float(np.mean(x)) if x.size else None
        results.append(dict(case, speed_m_s=speed,
            completed=bool(len(complete)), fallen=bool(len(fallen)), outside=outside,
            completion_time_s=float(samples['time'][complete[0]] - STAND_S) if len(complete) else None,
            fall_time_s=float(samples['time'][fallen[0]]) if len(fallen) else None,
            max_x_m=float(pos[:, 0].max()), final_x_m=float(pos[-1, 0]),
            max_lateral_m=float(np.abs(pos[:, 1]).max()),
            metric_scope='rough_before_outcome' if rough.any() else 'approach_before_outcome',
            metric_samples=int(selected.sum()), mean_vx_m_s=mean_or_none(vel),
            vx_rmse_m_s=mean_or_none((vel - speed) ** 2) ** .5 if vel.size else None,
            max_abs_roll_pitch_deg=float(np.abs(angles[:, :2]).max()) if angles.size else None,
            foot_names=case['foot_names'],
            foot_fz_p95_N=np.percentile(samples['foot_force'][selected, i, :, 2], 95, axis=0).tolist() if vel.size else None,
            foot_fz_max_N=samples['foot_force'][selected, i, :, 2].max(axis=0).tolist() if vel.size else None))
    return results


@torch.no_grad()
def play():
    args = gymutil.parse_arguments(description=__doc__, custom_parameters=[
        {'name': '--headless', 'action': 'store_true'},
        {'name': '--load_run', 'type': str, 'default': '-1'},
        {'name': '--checkpoint', 'type': int, 'default': -1},
        {'name': '--speed', 'type': float, 'default': .5},
        {'name': '--levels', 'type': str, 'default': 'flat,0,5,9'},
        {'name': '--seeds', 'type': int, 'default': 3},
        {'name': '--seed', 'type': int, 'default': 20261109},
        {'name': '--output', 'type': str},
    ])
    levels = args.levels.split(',')
    if not np.isfinite(args.speed) or not .1 <= args.speed <= 1.5 or args.seeds < 1:
        raise ValueError('speed must be 0.1..1.5 m/s; seeds must be positive')
    if any(x != 'flat' and (not x.isdigit() or not 0 <= int(x) <= 9) for x in levels):
        raise ValueError('levels must contain flat or integers 0..9')
    root = Path(LEGGED_GYM_ROOT_DIR)
    run = int(args.load_run) if args.load_run in ('-1', '-2') else args.load_run
    checkpoint = Path(get_load_path(str(root / 'logs/rough_minich'), run, args.checkpoint)).resolve()
    model_hash = sha256(checkpoint)
    saved = torch.load(str(checkpoint), map_location='cpu')
    frame = saved['model_state_dict']['actor.0.weight'].shape[1] - 19
    history = saved['model_state_dict']['estimator.encoder.0.weight'].shape[1]
    cfg, train_cfg = task_registry.get_cfgs('minich')
    # 不猜测相位输入语义；不同训练接口必须明确匹配配置。
    if (frame, history) != (cfg.env.num_one_step_observations, cfg.env.num_observations):
        raise ValueError(f'Checkpoint {frame}/{history} does not match configured observation contract')
    out = Path(args.output) if args.output else root / 'logs' / (
        'policy_eval_' + datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f') + '_gravel')
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = [dict(level=level, amplitude_m=0. if level == 'flat' else .01 + .07 * int(level) / 10,
                  terrain_seed=args.seed + s) for level in levels for s in range(args.seeds)]
    args.sim_device = f'cuda:{args.compute_device_id}' if args.sim_device_type == 'cuda' else 'cpu'
    args.num_envs = len(cases)
    forward_s = FINISH_X / args.speed + 5.
    duration = STAND_S + forward_s + 2.
    stairs.configure_stair_env(cfg)
    cfg.seed = args.seed
    cfg.env.num_envs = len(cases)
    cfg.env.episode_length_s = duration + 10
    cfg.commands.resampling_time = duration + 10
    cfg.rewards.log_phase_rewards = False
    cfg.terrain.horizontal_scale = .1
    cfg.terrain.vertical_scale = .005
    cfg.terrain.border_size = BORDER
    for key in dir(cfg.domain_rand):
        if isinstance(getattr(cfg.domain_rand, key), bool):
            setattr(cfg.domain_rand, key, False)
    torch.set_num_threads(1)
    GravelPlaybackRobot.cases = cases
    task_registry.register('minich_gravel_playback', GravelPlaybackRobot, cfg, train_cfg)
    env, _ = task_registry.make_env('minich_gravel_playback', args=args, env_cfg=cfg)
    try:
        runtime = dict(device=str(env.device), gpu_pipeline=bool(env.sim_params.use_gpu_pipeline),
                       gpu_physx=bool(env.sim_params.physx.use_gpu), headless=env.viewer is None)
        if not (runtime['device'].startswith('cuda') and runtime['gpu_pipeline'] and runtime['gpu_physx']):
            raise RuntimeError('Evaluation requires CUDA + GPU pipeline + GPU PhysX')
        if args.headless and env.viewer is not None:
            raise RuntimeError('Headless assertion failed')
        if frame == 47 and saved.get('observation_contract') != env.stair_gait.contract():
            raise ValueError('Checkpoint gait clock contract differs from environment')
        policy = HIMActorCritic(env.num_obs, env.num_privileged_obs, env.num_one_step_obs,
                               env.num_actions, **class_to_dict(train_cfg.policy)).to(env.device)
        policy.load_state_dict(saved['model_state_dict'], strict=True)
        policy.eval()
        flat.install_fixed_reset(env)
        flat.install_termination_override(env)
        flat.reset_to_fixed_state(env)
        if env.viewer is not None:
            p = env.root_states[0, :3].cpu().numpy()
            env.set_camera(p + np.array([-2., -2., 1.5]), p)
        foot_names = [env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])[j]
                      for j in env.feet_indices.cpu().tolist()]
        for case in cases:
            case['foot_names'] = foot_names
        asset = Path(cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=str(root)))
        sources = [Path(__file__), Path(stairs.__file__), Path(flat.__file__),
                   root / 'legged_gym/envs/base/legged_robot.py',
                   root / 'legged_gym/envs/minich/minich_config.py',
                   root / 'legged_gym/utils/terrain.py', Path(terrain_utils.__file__)]
        hashes = {str(p): sha256(p) for p in sources}
        steps = round(duration / env.dt)
        metadata = dict(checkpoint=str(checkpoint), checkpoint_sha256=model_hash,
            runtime=runtime, observation_frame=frame, observation_history=history,
            asset=str(asset), asset_sha256=sha256(asset), source_sha256=hashes,
            dt=env.dt, expected_samples=steps, speed_m_s=args.speed, cases=cases,
            terrain_model='fixed heightfield; no movable gravel; zero slope',
            rough_start_x_m=ROUGH_START, finish_x_m=FINISH_X, stand_s=STAND_S,
            forward_s=forward_s, config=class_to_dict(cfg),
            success_rule='x>=5m, abs(lateral)<=0.75m, upright and height>=0.13m for 0.2s before fall/outside',
            fall_rule='base height above sampled terrain<0.13m OR gravity_z>-0.5 for 0.2s',
            outside_rule='abs(lateral)>1.2m or x<-1m or x>15m',
            controls='fixed initial pose, no noise/domain randomization; training PD/action scale; no gait frequency override')
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        np.savez_compressed(out / 'terrain.npz', heights=env.terrain.heightsamples)
        for p in sources:
            try:
                relative = p.relative_to(root.parent)
            except ValueError:
                relative = Path(p.name)
            target = out / 'sources' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(p.read_bytes())
        records = {k: [] for k in ('time', 'command', 'position', 'velocity', 'rpy_deg',
                   'base_clearance', 'foot_force', 'foot_position', 'torque', 'fallen', 'completed', 'outside')}
        fallen = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        complete = fallen.clone()
        outside = fallen.clone()
        bad_time = torch.zeros(env.num_envs, device=env.device)
        hold = bad_time.clone()
        for step in range(steps):
            t = step * env.dt
            active = ~fallen & ~complete & ~outside
            env.commands.zero_()
            if STAND_S <= t < STAND_S + forward_s:
                env.commands[active, 0] = args.speed
            command = env.commands[:, :3].clone()
            scaled = env.commands[:, :3] * env.commands_scale
            env.obs_buf[:, :3] = scaled
            env.privileged_obs_buf[:, :3] = scaled
            # 与台阶回放相同：只让 step 推进观测历史。
            env.step(policy.act_inference(env.get_observations()).detach())
            clearance = env._get_base_heights()
            bad = (clearance < .13) | (env.projected_gravity[:, 2] > -.5)
            bad_time = torch.where(bad, bad_time + env.dt, torch.zeros_like(bad_time))
            fallen |= bad_time >= .2
            lateral = env.root_states[:, 1] - env.env_origins[:, 1]
            outside |= (lateral.abs() > 1.2) | (env.root_states[:, 0] < -1.) | (env.root_states[:, 0] > END_X - 1.)
            reached = (env.root_states[:, 0] >= FINISH_X) & (lateral.abs() <= .75) & ~bad & ~fallen & ~outside
            hold = torch.where(reached, hold + env.dt, torch.zeros_like(hold))
            complete |= hold >= .2
            values = dict(time=(step + 1) * env.dt, command=command,
                position=env.root_states[:, :3], velocity=env.base_lin_vel,
                rpy_deg=np.array(flat.get_base_imu_angles_deg(env)), base_clearance=clearance,
                foot_force=env.contact_forces[:, env.feet_indices], foot_position=env.feet_pos,
                torque=env.torques, fallen=fallen, completed=complete, outside=outside)
            for key, value in values.items():
                records[key].append(value.cpu().numpy().copy() if isinstance(value, torch.Tensor) else value)
            if step % round(2 / env.dt) == 0:
                print(f't={t:.1f}s completed={complete.sum().item()}/{env.num_envs} fallen={fallen.sum().item()} max_x={env.root_states[:,0].max().item():.2f}', flush=True)
        samples = {key: np.asarray(value) for key, value in records.items()}
        assert all(v.shape[0] == steps and np.isfinite(v).all() for v in samples.values())
        assert sha256(checkpoint) == model_hash and sha256(asset) == metadata['asset_sha256']
        assert all(sha256(p) == h for p, h in hashes.items()), 'Sources changed during evaluation'
        results = score(samples, cases, env.dt, forward_s, args.speed)
        np.savez_compressed(out / 'samples.npz', **samples)
        (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        (out / 'validation.json').write_text(json.dumps(dict(finite=True, sample_count=steps,
            shapes={k: list(v.shape) for k, v in samples.items()}, hashes_unchanged=True, runtime=runtime), indent=2) + '\n')
        print(json.dumps(results, indent=2), flush=True)
        print(f'结果目录: {out}', flush=True)
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    play()

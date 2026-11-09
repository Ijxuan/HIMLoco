"""Collect controlled foot-height traces without changing training rewards.

Run under gym4 with --scene flat|stairs18|stairs8, --output DIRECTORY,
--policy-checkpoint FILE, plus normal legged_gym GPU/headless arguments.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import play_minich_stairs as stairs  # Isaac Gym must precede torch.
import numpy as np
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import class_to_dict, get_args, task_registry
from legged_gym.utils.swing_peak_height import SwingPeakHeight, peak_height_score


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=['flat', 'stairs18', 'stairs8'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--policy-checkpoint', type=Path, required=True)
    opts, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    args = get_args()
    args.task, args.num_envs = 'minich', 3
    args.load_run = args.checkpoint = None
    checkpoint = opts.policy_checkpoint.resolve()
    output = opts.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    speeds = [.2, .3, .4]
    stairs.POLICY_SPECS = tuple((f'vx_{v:g}', checkpoint.parent.name, 1000) for v in speeds)
    stairs.STAIR_COUNT = 2
    stairs.STAIR_HEIGHT_M = .18 if opts.scene == 'stairs18' else .08
    stairs.STAIR_DEPTH_M = .20 if opts.scene == 'stairs18' else 1.20
    stairs.STAIR_START_X_M = .50 if opts.scene == 'stairs18' else 1.20
    # Exact collision mesh; reward sampling retains the training 10 cm grid.
    stairs.HEIGHT_SAMPLE_RESOLUTION_M = .10
    cfg, train = task_registry.get_cfgs('minich')
    stairs.configure_stair_env(cfg)
    cfg.env.episode_length_s = 30.
    cfg.commands.resampling_time = 1000.
    cfg.rewards.log_phase_rewards = False
    cfg.commands.gait_training_fraction = 0.
    for name in dir(cfg.domain_rand):
        if name.startswith('randomize_') or name in ('push_robots', 'disturbance', 'delay'):
            setattr(cfg.domain_rand, name, False)
    if opts.scene == 'flat':
        cfg.terrain.mesh_type = 'plane'
        robot_class = stairs.LeggedRobot
    else:
        robot_class = stairs.StairPlaybackRobot
    sources = [Path(__file__), Path(stairs.__file__), Path(stairs.flat.__file__),
               Path(LEGGED_GYM_ROOT_DIR) / 'legged_gym/envs/base/legged_robot.py',
               Path(LEGGED_GYM_ROOT_DIR) / 'legged_gym/envs/minich/minich_config.py',
               Path(LEGGED_GYM_ROOT_DIR) / 'legged_gym/utils/swing_peak_height.py']
    hashes = {str(p.resolve()): sha256(p) for p in sources}
    before_hash = sha256(checkpoint)
    task_registry.register('swing_height_assessment', robot_class, cfg, train)
    env, _ = task_registry.make_env('swing_height_assessment', args=args, env_cfg=cfg)
    try:
        assert str(env.device) != 'cpu'
        assert env.sim_params.use_gpu_pipeline and env.sim_params.physx.use_gpu
        assert env.viewer is None
        print('GPU_RUNTIME_ASSERTIONS_PASSED', flush=True)
        stairs.flat.install_fixed_reset(env)
        stairs.flat.install_termination_override(env)
        train.runner.resume = False
        runner, _ = task_registry.make_alg_runner(env=env, args=args, train_cfg=train, log_root=None)
        runner.load(str(checkpoint), load_optimizer=False)
        policy = runner.get_inference_policy(env.device)
        stairs.flat.reset_to_fixed_state(env)
        # Keep original scope: custom playback terrain is not a training long-stair column.
        # Two independent meters show counterfactual scores without changing policy/reward.
        sampled = SwingPeakHeight(3, 4, env.device, cfg.rewards, env.dt)
        exact = SwingPeakHeight(3, 4, env.device, cfg.rewards, env.dt)
        records = {}
        elapsed_start = torch.zeros(3, 4, device=env.device)
        elapsed_start_exact = torch.zeros_like(elapsed_start)
        takeoff_z = torch.zeros_like(elapsed_start)
        takeoff_z_exact = torch.zeros_like(elapsed_start)
        world_peak = torch.zeros_like(elapsed_start)
        world_peak_exact = torch.zeros_like(elapsed_start)
        radius = cfg.rewards.step_clearance_foot_radius
        before, moving, after = 3., 20., 2.
        expected_steps = int(round((before + moving + after) / env.dt))
        step_index = 0
        compute_reward = env.compute_reward

        def capture():
            compute_reward()
            feet = env.feet_pos
            sole = feet[..., 2] - radius
            ground_sampled = env._sample_step_heights(feet[..., :2])
            if opts.scene == 'flat':
                ground_exact = torch.zeros_like(sole)
            else:
                levels = torch.floor((feet[..., 0] - stairs.STAIR_START_X_M) / stairs.STAIR_DEPTH_M + 1e-6) + 1
                ground_exact = levels.clamp(0, 2) * stairs.STAIR_HEIGHT_M
            clearance = sole - ground_sampled
            clearance_exact = sole - ground_exact
            force = env.contact_forces[:, env.feet_indices, 2]
            support = (clearance.abs() <= cfg.rewards.step_clearance_support_tolerance) & (force > 1.)
            support_exact = (clearance_exact.abs() <= cfg.rewards.step_clearance_support_tolerance) & (force > 1.)
            enabled = (env.commands[:, 0].abs() >= .1) & (env.commands[:, 2].abs() < .05)
            starts = []
            for meter, heights, supported, origin, peak, times in (
                (sampled, clearance, support, takeoff_z, world_peak, elapsed_start),
                (exact, clearance_exact, support_exact, takeoff_z_exact, world_peak_exact, elapsed_start_exact),
            ):
                previous_started = meter.stats['started'].clone()
                meter.update(heights, supported, env.commands, enabled)
                started = meter.stats['started'] > previous_started
                # Use preceding support sample as takeoff reference.
                prior = records['feet'][-1][..., 2] - radius if records else sole.cpu().numpy()
                origin[started] = torch.as_tensor(prior, device=env.device)[started]
                peak[started] = 0.
                times[started] = step_index * env.dt
                peak[:] = torch.where(meter.active | meter.event,
                                      torch.maximum(peak, sole - origin), peak)
                starts.append(started)
            row = dict(feet=feet, base=env.root_states[:, :3], quat=env.base_quat,
                       velocity=env.base_lin_vel, projected_gravity=env.projected_gravity,
                       force_z=force, commands=env.commands[:, :3],
                       ground_sampled=ground_sampled, ground_exact=ground_exact,
                       support=support, support_exact=support_exact,
                       clearance=clearance, clearance_exact=clearance_exact,
                       started=starts[0], started_exact=starts[1],
                       event=sampled.event, event_exact=exact.event,
                       peak=sampled.event_peak, peak_exact=exact.event_peak,
                       timeout=sampled.timed_out, timeout_exact=exact.timed_out,
                       world_lift=world_peak, world_lift_exact=world_peak_exact,
                       takeoff_time=elapsed_start, takeoff_time_exact=elapsed_start_exact,
                       score_delta=sampled.score * env.dt,
                       actual_height_reward=env.swing_peak_height.score * env.dt * cfg.rewards.scales.swing_peak_height,
                       reward_total=env.rew_buf)
            for key, value in row.items():
                records.setdefault(key, []).append(value.detach().cpu().numpy().copy())

        env.compute_reward = capture
        for step_index in range(expected_steps):
            t = step_index * env.dt
            env.commands.zero_()
            if before <= t < before + moving:
                env.commands[:, 0] = torch.tensor(speeds, device=env.device)
            # Preserve one history shift per physics-policy step.
            env.obs_buf[:, :3] = env.commands[:, :3] * env.commands_scale
            env.privileged_obs_buf[:, :3] = env.commands[:, :3] * env.commands_scale
            env.step(policy(env.get_observations().detach()))
            if step_index % 250 == 0:
                print('COLLECT', opts.scene, step_index, flush=True)
        arrays = {key: np.stack(value) for key, value in records.items()}
        assert arrays['feet'].shape == (expected_steps, 3, 4, 3)
        assert all(np.isfinite(value).all() for value in arrays.values())
        assert not arrays['actual_height_reward'].any()
        assert all(sha256(path) == value for path, value in hashes.items())
        assert sha256(checkpoint) == before_hash
        np.savez_compressed(output / 'samples.npz', **arrays)
        asset = Path(cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR))
        meta = dict(scene=opts.scene, checkpoint=str(checkpoint), checkpoint_sha256=before_hash,
                    dt=env.dt, steps=expected_steps, num_envs=3, feet=env.feet_names,
                    speeds=speeds, seed=args.seed, timings=dict(before=before, moving=moving, after=after),
                    terrain=dict(height=0. if opts.scene == 'flat' else stairs.STAIR_HEIGHT_M,
                                 depth=stairs.STAIR_DEPTH_M, count=2,
                                 start=stairs.STAIR_START_X_M, sample_resolution=cfg.terrain.horizontal_scale),
                    config=class_to_dict(cfg), device=str(env.device),
                    gpu_pipeline=env.sim_params.use_gpu_pipeline,
                    gpu_physx=env.sim_params.physx.use_gpu, headless=env.viewer is None,
                    source_hashes=hashes, urdf_sha256=sha256(asset),
                    diagnostic='Independent command-gated scorer; playback actual height reward is zero. '
                               'Local sampled clearance matches production sampling code; exact clearance uses analytic collision surfaces.',
                    reset='fixed initial state; automatic termination/reset disabled; analysis excludes post-fall samples',
                    validation=dict(finite=True, expected_sample_count=True, sources_unchanged=True))
        (output / 'metadata.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2) + '\n')
        print('DONE', output, flush=True)
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()

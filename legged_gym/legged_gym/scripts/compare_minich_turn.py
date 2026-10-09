"""Instrument the existing flat playback, without changing training or playback files.

COMPARE_MODE=original|controlled; COMPARE_OUTPUT=<directory>.
COMPARE_YAW_RAD_S overrides the playback command at runtime.
COMPARE_HISTORY=playback|single selects the observation history timing.
Use the usual playback CLI, including --seed and CPU device arguments.
"""
import hashlib
import json
import os
from pathlib import Path

import play_minich_flat_height as playback  # Isaac Gym must precede torch.
import numpy as np
import torch
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import class_to_dict, get_args, get_load_path, task_registry


def main():
    args = get_args()
    # Three tiny inference batches do not benefit from a large CPU thread pool.
    torch.set_num_threads(1)
    mode = os.environ.get("COMPARE_MODE", "original")
    if mode not in ("original", "controlled"):
        raise ValueError(mode)
    history = os.environ.get("COMPARE_HISTORY", "playback")
    if history not in ("playback", "single"):
        raise ValueError(history)
    if "COMPARE_YAW_RAD_S" in os.environ:
        playback.TURN_YAW_RATE_RAD_S = float(os.environ["COMPARE_YAW_RAD_S"])
        if not np.isfinite(playback.TURN_YAW_RATE_RAD_S):
            raise ValueError("COMPARE_YAW_RAD_S must be finite")
    output = Path(os.environ["COMPARE_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    # Deserialization adaptation only; the original runner still loads its full state.
    original_load = torch.load
    def cpu_load(*a, **kw):
        kw.setdefault("map_location", "cpu")
        return original_load(*a, **kw)
    torch.load = cpu_load
    paths = [get_load_path(str(Path(LEGGED_GYM_ROOT_DIR) / "logs/rough_minich"), run, ckpt)
             for _, run, ckpt in playback.POLICY_SPECS]
    # Pin selection before creating any runner, and retain the third historical reference.
    playback.POLICY_SPECS = tuple((spec[0], Path(path).parent.name,
                                  int(Path(path).stem.split("_")[-1]))
                                 for spec, path in zip(playback.POLICY_SPECS, paths))
    original_configure = playback.configure_flat_sequence_env
    def configure(cfg):
        original_configure(cfg)
        if mode == "controlled":
            for name in dir(cfg.domain_rand):
                if isinstance(getattr(cfg.domain_rand, name), bool):
                    setattr(cfg.domain_rand, name, False)
    playback.configure_flat_sequence_env = configure
    rows, physics_torques, physics_velocities, reward_terms = [], [], [], []
    recording = [False]
    original_meter = playback.install_latest_policy_reward_meter
    def install_meter(env):
        meter = original_meter(env)
        if history == "single":
            original_compute = env.compute_observations
            original_step = env.step
            inside_step = [False]
            def compute():
                if inside_step[0]:
                    return original_compute()
                env.obs_buf[:, :3] = env.commands[:, :3] * env.commands_scale
                env.privileged_obs_buf[:, :3] = env.commands[:, :3] * env.commands_scale
            def step(actions):
                inside_step[0] = True
                try:
                    return original_step(actions)
                finally:
                    inside_step[0] = False
            env.compute_observations = compute
            env.step = step
        recording[0] = True
        return meter
    playback.install_latest_policy_reward_meter = install_meter
    metadata = {"mode": mode, "history_timing": history, "seed": args.seed, "policies": [
        {"label": spec[0], "path": path,
         "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        for spec, path in zip(playback.POLICY_SPECS, paths)],
        "schedule_s": [playback.STAND_BEFORE_S, playback.FORWARD_S,
                       playback.TURN_S, playback.STAND_AFTER_S],
        "yaw_command_rad_s": playback.TURN_YAW_RATE_RAD_S}
    env_holder = []
    original_make_env = task_registry.make_env
    def make_env(*a, **kw):
        env, cfg = original_make_env(*a, **kw)
        env_holder.append(env)
        metadata["env_config"] = class_to_dict(cfg)
        metadata["dt"] = env.dt
        metadata["physics_dt"] = cfg.sim.dt
        metadata["reward_names"] = list(env.episode_sums)
        metadata["dof_names"] = env.dof_names
        body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
        metadata["foot_body_names"] = [body_names[i] for i in env.feet_indices.cpu().tolist()]
        metadata["torque_limits"] = env.torque_limits.cpu().tolist()
        metadata["realized_random_factors"] = {name: getattr(env, name).cpu().tolist()
            for name in ("Kp_factors", "Kd_factors", "motor_strength_factors", "com_displacements", "com_displacement")
            if hasattr(env, name)}
        original_torques = env._compute_torques
        def compute_torques(actions):
            torque = original_torques(actions)
            if recording[0]:
                physics_torques.append(torque.detach().cpu().numpy().copy())
                physics_velocities.append(env.dof_vel.detach().cpu().numpy().copy())
            return torque
        env._compute_torques = compute_torques
        original_step = env.step
        def step(actions):
            if not recording[0]:
                return original_step(actions)
            before = torch.stack(list(env.episode_sums.values())).clone()
            result = original_step(actions)
            after = torch.stack(list(env.episode_sums.values()))
            reward_terms.append((after - before).T.cpu().numpy())
            contact = env.contact_forces[:, env.feet_indices, 2]
            trunk = env.gym.find_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], "trunk")
            rows.append(np.concatenate([
                env.root_states.detach().cpu().numpy().copy(),
                env.base_lin_vel.cpu().numpy(), env.base_ang_vel.cpu().numpy(),
                np.asarray(playback.get_base_imu_angles_deg(env)),
                env.projected_gravity.cpu().numpy(),
                result[2].cpu().numpy()[:, None],
                contact.cpu().numpy(), env.feet_vel[:, :, :2].norm(dim=-1).cpu().numpy(),
                env.contact_forces[:, trunk].norm(dim=-1).cpu().numpy()[:, None],
                env.dof_pos.cpu().numpy(), env.actions.cpu().numpy(),
                env.feet_pos[:, :, 2].cpu().numpy()], axis=1))
            return result
        env.step = step
        return env, cfg
    task_registry.make_env = make_env
    try:
        with torch.no_grad():
            playback.play(args)
        columns = (["x", "y", "z", "qx", "qy", "qz", "qw", "world_vx", "world_vy", "world_vz",
                    "world_wx", "world_wy", "world_wz", "body_vx", "body_vy", "body_vz",
                    "body_wx", "body_wy", "body_wz", "roll_deg", "pitch_deg", "yaw_deg",
                    "gravity_x", "gravity_y", "gravity_z", "reward"]
                   + ["foot_fz_" + str(i) for i in range(4)]
                   + ["foot_xy_speed_" + str(i) for i in range(4)] + ["trunk_contact_N"]
                   + ["joint_pos_" + name for name in metadata["dof_names"]]
                   + ["action_" + name for name in metadata["dof_names"]]
                   + ["foot_world_z_" + str(i) for i in range(4)])
        metadata["columns"] = columns
        np.savez_compressed(output / "samples.npz", states=np.asarray(rows),
                            torques=np.asarray(physics_torques),
                            joint_velocities=np.asarray(physics_velocities),
                            reward_terms=np.asarray(reward_terms))
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
        print("Saved diagnostic data to", output, flush=True)
    finally:
        for env in env_holder:
            if env.viewer is not None:
                env.gym.destroy_viewer(env.viewer)
            env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()

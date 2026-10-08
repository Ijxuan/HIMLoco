"""Measure joint-side torque and mechanical power with a saved Mini Cheetah policy.

Independent flat-ground command cases run side by side. Capture the clipped PD
torque and pre-integration joint velocity at EVERY physics step (200 Hz), rather
than only the last substep of the 50 Hz policy cycle. No training or export.
"""

import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
from types import MethodType

import isaacgym
from isaacgym import gymutil
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import class_to_dict, task_registry
from play_minich_flat_height import install_fixed_reset, reset_to_fixed_state


CASES = (
    ("stand", (0.0, 0.0, 0.0)),
    ("forward_0.5", (0.5, 0.0, 0.0)),
    ("forward_1.0", (1.0, 0.0, 0.0)),
    ("backward_0.5", (-0.5, 0.0, 0.0)),
    ("lateral_0.5", (0.0, 0.5, 0.0)),
    ("turn_left_1.0", (0.0, 0.0, 1.0)),
    ("turn_right_1.0", (0.0, 0.0, -1.0)),
)
GROUPS = {"hip_abduction": "_hip_joint", "thigh": "_thigh_joint", "calf": "_calf_joint"}


def joint_statistics(torque, velocity, limit):
    """Both inputs are a time series for ONE motor; signed power can cancel."""
    absolute = np.abs(torque)
    power = torque * velocity
    return {
        "torque_mean_abs_Nm": float(absolute.mean()),
        "torque_rms_Nm": float(np.sqrt(np.mean(torque ** 2))),
        "torque_p95_abs_Nm": float(np.percentile(absolute, 95)),
        "torque_p99_abs_Nm": float(np.percentile(absolute, 99)),
        "torque_peak_abs_Nm": float(absolute.max()),
        "torque_limit_Nm": float(limit),
        "torque_saturation_fraction": float(np.mean(absolute >= limit * 0.99)),
        "speed_p99_abs_rad_s": float(np.percentile(np.abs(velocity), 99)),
        "speed_peak_abs_rad_s": float(np.abs(velocity).max()),
        "power_mean_abs_W": float(np.abs(power).mean()),
        "power_mean_positive_W": float(np.maximum(power, 0).mean()),
        "power_mean_negative_abs_W": float(np.maximum(-power, 0).mean()),
        "power_p99_abs_W": float(np.percentile(np.abs(power), 99)),
        "power_peak_abs_W": float(np.abs(power).max()),
    }


def parse_args():
    return gymutil.parse_arguments(
        description=__doc__,
        custom_parameters=[
            {"name": "--policy", "type": str, "default": str(Path(LEGGED_GYM_ROOT_DIR) / "logs/rough_minich/Sep18_18-36-28_smoke_minich/exported/policies/model_1000.jit")},
            {"name": "--output", "type": str, "default": None},
            {"name": "--headless", "action": "store_true"},
            {"name": "--rl_device", "type": str, "default": "cpu"},
            {"name": "--warmup_s", "type": float, "default": 3.0},
            {"name": "--ramp_s", "type": float, "default": 1.0},
            {"name": "--settle_s", "type": float, "default": 2.0},
            {"name": "--sample_s", "type": float, "default": 10.0},
        ],
    )


def configure(args):
    env_cfg, _ = task_registry.get_cfgs("minich")
    env_cfg.env.num_envs = len(CASES)
    env_cfg.env.env_spacing = 5.0
    env_cfg.env.episode_length_s = args.warmup_s + args.ramp_s + args.settle_s + args.sample_s + 5.0
    env_cfg.init_state.pos[2] = 0.36
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    # Disable all randomization, including gains and action delay left enabled
    # by the usual viewer. This isolates load differences between joint types.
    for name in dir(env_cfg.domain_rand):
        value = getattr(env_cfg.domain_rand, name)
        if isinstance(value, bool):
            setattr(env_cfg.domain_rand, name, False)
    env_cfg.commands.curriculum = False
    env_cfg.commands.enable_turn_to_stand = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.in_place_turn_probability = 0.0
    env_cfg.commands.resampling_time = env_cfg.env.episode_length_s + 5.0
    args.task = "minich"
    args.num_envs = len(CASES)
    args.seed = 1
    # Registry only reads these CLI overrides during make_env.
    args.max_iterations = args.resume = args.experiment_name = args.run_name = None
    args.load_run = args.checkpoint = None
    return task_registry.make_env(name="minich", args=args, env_cfg=env_cfg)[0]


def write_results(output, meta, torques, velocities, states, stages, times, env):
    limits = env.torque_limits.detach().cpu().numpy()
    names = env.dof_names
    stable = np.array(stages) == "steady"
    report = {"metadata": meta, "cases": {}}
    rows = []
    for case_index, (label, command) in enumerate(CASES):
        state = states[stable, case_index]
        case = {
            "command_vx_vy_yaw": command,
            "actual_mean_vx_vy_yaw": state[:, :3].mean(axis=0).tolist(),
            "actual_rms_error_vx_vy_yaw": np.sqrt(np.mean((state[:, :3] - command) ** 2, axis=0)).tolist(),
            "min_base_height_m": float(state[:, 3].min()),
            "max_trunk_contact_force_N": float(state[:, 5].max()),
            "fell": bool(np.any(states[:, case_index, 3] < 0.13) or np.any(states[:, case_index, 4] > -0.7)),
            "joints": {}, "groups_worst_motor": {},
        }
        for stage in ("warmup", "ramp", "settle", "steady"):
            mask = np.array(stages) == stage
            if not mask.any():
                continue
            stage_joints = {}
            for j, name in enumerate(names):
                stats = joint_statistics(torques[mask, case_index, j], velocities[mask, case_index, j], limits[j])
                stage_joints[name] = stats
                rows.append({"case": label, "stage": stage, "joint": name, **stats})
            if stage == "steady":
                case["joints"] = stage_joints
        for group, suffix in GROUPS.items():
            motors = {name: stats for name, stats in case["joints"].items() if name.endswith(suffix)}
            case["groups_worst_motor"][group] = {
                metric: max(stats[metric] for stats in motors.values()) for metric in next(iter(motors.values()))
            }
        report["cases"][label] = case
    (output / "summary.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    with (output / "joint_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Full physical-step series, indexed by case and DOF names in metadata.
    np.savez_compressed(output / "physics_samples.npz", torque_Nm=torques, velocity_rad_s=velocities,
                        power_W=torques * velocities, state=states, time_s=times, stage=stages,
                        case_names=[case[0] for case in CASES], dof_names=names)
    lines = ["# Mini Cheetah joint load measurement", "", f"Policy: `{meta['policy']}`", "",
             "Clipped joint-side PD torque, pre-integration joint velocity; mechanical power = torque × velocity.",
             "All randomization disabled. Flat ground. No automatic resets. Steady data exclude warmup, ramp and settling.",
             "Group values below are the worst individual motor of the four legs, not their sum.", "",
             "| Case | Group | RMS torque Nm | P99 torque Nm | Peak torque Nm | Mean abs power W | Peak abs power W |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for label, case in report["cases"].items():
        for group, stats in case["groups_worst_motor"].items():
            lines.append(f"| {label} | {group} | {stats['torque_rms_Nm']:.3f} | {stats['torque_p99_abs_Nm']:.3f} | {stats['torque_peak_abs_Nm']:.3f} | {stats['power_mean_abs_W']:.3f} | {stats['power_peak_abs_W']:.3f} |")
    lines += ["", "## Tracking and validity", ""]
    for label, case in report["cases"].items():
        lines.append(f"- {label}: actual mean vx/vy/yaw = {case['actual_mean_vx_vy_yaw']}; minimum height = {case['min_base_height_m']:.3f} m; fell = {case['fell']}; trunk contact peak = {case['max_trunk_contact_force_N']:.3f} N.")
    lines += ["", "Mechanical power is not electrical consumption. Zero-speed holding torque still causes motor heating.",
              "These loads apply to this model and policy on flat ground, not jumps, impacts, payload or a redesigned robot.",
              "Torque-speed pairs and saturation must be checked before actuator sizing; joint-side values require transmission mapping."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    return report


def main():
    args = parse_args()
    if args.warmup_s < 0 or args.ramp_s <= 0 or args.settle_s < 0 or args.sample_s <= 0:
        raise ValueError("Require warmup/settle >= 0 and ramp/sample > 0")
    torch.set_num_threads(1)
    policy_path = Path(args.policy).resolve()
    policy = torch.jit.load(str(policy_path), map_location=args.rl_device).eval()
    output = Path(args.output) if args.output else Path(LEGGED_GYM_ROOT_DIR) / "logs/joint_loads" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=True)
    env = configure(args)
    try:
        install_fixed_reset(env)
        obs = reset_to_fixed_state(env)

        def no_automatic_reset(self):
            self.reset_buf.zero_()
            self.time_out_buf.zero_()

        env.check_termination = MethodType(no_automatic_reset, env)
        commands = torch.tensor([case[1] for case in CASES], device=env.device, dtype=torch.float)
        captured_torques, captured_velocities = [], []
        original_compute = env._compute_torques

        def capture_torques(self, actions):
            torque = original_compute(actions)
            captured_torques.append(torque.detach().cpu().numpy().copy())
            captured_velocities.append(self.dof_vel.detach().cpu().numpy().copy())
            return torque

        env._compute_torques = MethodType(capture_torques, env)
        stages, times, states = [], [], []
        warmup_steps = int(round(args.warmup_s / env.dt))
        ramp_steps = int(round(args.ramp_s / env.dt))
        settle_end = warmup_steps + ramp_steps + int(round(args.settle_s / env.dt))
        total_steps = settle_end + int(round(args.sample_s / env.dt))
        with torch.inference_mode():
            for step in range(total_steps):
                if step < warmup_steps:
                    stage, factor = "warmup", 0.0
                elif step < warmup_steps + ramp_steps:
                    stage, factor = "ramp", (step - warmup_steps + 1) / ramp_steps
                elif step < settle_end:
                    stage, factor = "settle", 1.0
                else:
                    stage, factor = "steady", 1.0
                env.commands.zero_()
                env.commands[:, :3] = commands * factor
                # Update command channels in the current frame WITHOUT rolling
                # the six-frame history a second time; env.step rolls it once.
                env.obs_buf[:, :3] = env.commands[:, :3] * env.commands_scale
                actions = policy(env.get_observations().to(args.rl_device)).to(env.device)
                obs, _, _, _, _, _, _ = env.step(actions)
                if not torch.isfinite(obs).all() or not torch.isfinite(env.torques).all():
                    raise RuntimeError(f"Non-finite simulation state at step {step}")
                trunk = torch.linalg.vector_norm(env.contact_forces[:, env.termination_contact_indices], dim=-1).max(dim=1).values
                state = torch.stack((env.base_lin_vel[:, 0], env.base_lin_vel[:, 1], env.base_ang_vel[:, 2],
                                     env.root_states[:, 2], env.projected_gravity[:, 2], trunk), dim=1).cpu().numpy()
                for substep in range(env.cfg.control.decimation):
                    stages.append(stage)
                    times.append(step * env.dt + substep * env.sim_params.dt)
                    states.append(state.copy())
                if step % max(1, int(round(2.0 / env.dt))) == 0:
                    print(f"t={step * env.dt:.2f}s stage={stage} heights={state[:, 3].round(3).tolist()}", flush=True)
        meta = {
            "policy": str(policy_path), "policy_sha256": hashlib.sha256(policy_path.read_bytes()).hexdigest(),
            "physics_dt_s": env.sim_params.dt, "policy_dt_s": env.dt,
            "duration_args": vars(args), "dof_names": env.dof_names,
            "state_columns": ["vx_m_s", "vy_m_s", "yaw_rad_s", "height_m", "gravity_z", "trunk_force_N"],
            "sampling": "clipped torque and velocity before each physics integration; state repeated from end of policy step",
            "terrain": "plane", "domain_randomization": False,
            "initial_height_m": 0.36, "source_env": str(Path(__file__).resolve().parents[1] / "envs/base/legged_robot.py"),
            "environment_config": class_to_dict(env.cfg),
        }
        write_results(output, meta, np.stack(captured_torques), np.stack(captured_velocities),
                      np.stack(states), stages, np.array(times), env)
        print(f"Saved results: {output.resolve()}", flush=True)
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()

"""Measure a saved HIM Actor's dependence on its 16-dimensional latent.

Run separately for --terrain flat and stairs, using gym4 and CPU PhysX.
Normal reference trajectories supply phase/scenario-matched, unit-length donor
latents. Counterfactual inference keeps the 45 observations and estimated
velocity fixed. Closed-loop tests intervene only after a 3-second warmup.
"""
import csv
import hashlib
import json
from pathlib import Path
from types import MethodType

import isaacgym
from isaacgym import gymtorch, gymutil
import numpy as np
import torch
import torch.nn.functional as F

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import class_to_dict, task_registry
import play_minich_flat_height as flat
import play_minich_stairs as stairs


MODES = ("normal", "shuffle", "constant", "stand_constant", "zero")
FLAT_CASES = (("stand", (0., 0., 0.)), ("forward", (1., 0., 0.)),
              ("turn_left", (0., 0., 1.)), ("turn_right", (0., 0., -1.)))
STAIR_CASES = (("stairs_forward", (1., 0., 0.)),)


def parse_args():
    return gymutil.parse_arguments(description=__doc__, custom_parameters=[
        {"name": "--policy", "type": str, "default": str(Path(LEGGED_GYM_ROOT_DIR) /
         "logs/rough_minich/Oct09_17-33-13_smoke_minich/exported/policies/model_1000.jit")},
        {"name": "--terrain", "type": str, "default": "flat"},
        {"name": "--output", "type": str, "required": True},
        {"name": "--headless", "action": "store_true"},
        {"name": "--rl_device", "type": str, "default": "cpu"},
        {"name": "--replicas", "type": int, "default": 3},
        {"name": "--seed", "type": int, "default": 20261010},
    ])


def make_env(args, cases, modes):
    cfg, train_cfg = task_registry.get_cfgs("minich")
    if args.terrain == "stairs":
        stairs.configure_stair_env(cfg)
        stairs.ENV_SPACING_M = 3.
    else:
        flat.configure_flat_sequence_env(cfg)
    n = len(cases) * args.replicas * len(modes)
    cfg.env.num_envs = n
    cfg.env.env_spacing = 5.
    cfg.env.episode_length_s = 30.
    cfg.commands.in_place_turn_probability = 0.
    cfg.commands.resampling_time = 40.
    cfg.asset.terminate_after_contacts_on = []
    for name in dir(cfg.domain_rand):
        if isinstance(getattr(cfg.domain_rand, name), bool):
            setattr(cfg.domain_rand, name, False)
    cfg.noise.add_noise = False
    args.task = "latent_diagnostic_" + args.terrain
    args.num_envs = n
    for name in ("max_iterations", "resume", "experiment_name", "run_name", "load_run", "checkpoint"):
        setattr(args, name, None)
    cls = stairs.StairPlaybackRobot if args.terrain == "stairs" else LeggedRobot
    task_registry.register(args.task, cls, cfg, train_cfg)
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=cfg)
    flat.install_fixed_reset(env)
    # Identical initial perturbations across interventions; replicas use
    # distinct small joint-position perturbations rather than identical seeds.
    rng = np.random.RandomState(args.seed)
    jitter = rng.uniform(-.015, .015, (len(cases), args.replicas, 12))
    jitter = np.tile(jitter.reshape(-1, 12), (len(modes), 1))
    original_reset = env._reset_dofs
    def reset_dofs(self, ids):
        original_reset(ids)
        self.dof_pos[ids] += torch.tensor(jitter, device=self.device, dtype=torch.float)[ids]
        ids32 = ids.to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                             gymtorch.unwrap_tensor(ids32), len(ids))
    env._reset_dofs = MethodType(reset_dofs, env)
    def no_reset(self):
        self.reset_buf.zero_()
        self.time_out_buf.zero_()
    env.check_termination = MethodType(no_reset, env)
    flat.reset_to_fixed_state(env)
    return env, class_to_dict(cfg)


def split_policy(policy, obs):
    parts = policy.estimator(obs)
    vel, z = parts[:, :3], F.normalize(parts[:, 3:19], dim=-1)
    return torch.cat((obs[:, :45], vel), dim=-1), z


def rollout(args, policy, cases, modes, reference=None):
    env, cfg = make_env(args, cases, modes)
    group = len(cases) * args.replicas
    commands = torch.tensor([c[1] for c in cases for _ in range(args.replicas)] * len(modes),
                            device=env.device)
    rng = np.random.RandomState(args.seed + 71)
    records = {k: [] for k in ("obs", "latent", "action", "state", "phase")}
    constants = {}
    banks = {}
    if reference is not None:
        warm = (reference["phase"] == 0) & (np.arange(len(reference["phase"])) * env.dt >= 2.)
        stand_mean = reference["latent"][warm].reshape(-1, 16).mean(axis=0)
        stand_constant = stand_mean / max(np.linalg.norm(stand_mean), 1e-12)
        for case in range(len(cases)):
            sl = slice(case * args.replicas, (case + 1) * args.replicas)
            for phase in (1, 2):
                bank = reference["latent"][reference["phase"] == phase, sl].reshape(-1, 16)
                banks[case, phase] = bank
            mean = banks[case, 1].mean(axis=0)
            constants[case] = mean / max(np.linalg.norm(mean), 1e-12)
    try:
        with torch.inference_mode():
            for step in range(round(19. / env.dt)):
                t = step * env.dt
                phase = 0 if t < 3. else (1 if t < 13. else 2)
                env.commands.zero_()
                if phase == 1:
                    env.commands[:, :3] = commands
                # No second history roll: step() constructs the next frame.
                env.obs_buf[:, :3] = env.commands[:, :3] * env.commands_scale
                obs = env.get_observations().clone()
                base, z = split_policy(policy, obs)
                supplied = z.clone()
                if phase and reference is not None:
                    for m, mode in enumerate(modes):
                        for case in range(len(cases)):
                            sl = slice(m * group + case * args.replicas,
                                       m * group + (case + 1) * args.replicas)
                            if mode == "shuffle":
                                bank = banks[case, phase]
                                donor = bank[rng.randint(len(bank), size=args.replicas)]
                                supplied[sl] = torch.as_tensor(donor, device=env.device)
                            elif mode == "constant":
                                supplied[sl] = torch.as_tensor(constants[case], device=env.device)
                            elif mode == "stand_constant":
                                supplied[sl] = torch.as_tensor(stand_constant, device=env.device)
                            elif mode == "zero":
                                supplied[sl] = 0.
                action = policy.actor(torch.cat((base, supplied), dim=-1))
                if step == 0:
                    assert torch.allclose(action, policy(obs), atol=1e-6), "Split inference differs from JIT"
                _, _, reward, _, _, _, _ = env.step(action)
                if not torch.isfinite(env.obs_buf).all() or not torch.isfinite(action).all():
                    raise RuntimeError("Non-finite simulation or policy")
                ground = env._get_base_heights()
                tilt = torch.acos(torch.clamp(-env.projected_gravity[:, 2], -1., 1.))
                state = torch.cat((env.root_states[:, :3], env.base_lin_vel[:, :2],
                                   env.base_ang_vel[:, 2:3], ground[:, None], tilt[:, None],
                                   reward[:, None]), dim=-1)
                records["obs"].append(obs.cpu().numpy())
                records["latent"].append(z.cpu().numpy())
                records["action"].append(action.cpu().numpy())
                records["state"].append(state.cpu().numpy())
                records["phase"].append(phase)
                if step % 250 == 0:
                    print(args.terrain, list(modes), "t=", round(t, 2), flush=True)
        data = {k: np.asarray(v) for k, v in records.items()}
        data["dt"] = np.array(env.dt)
        return data, cfg
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


def offline(policy, data, cases, replicas, seed):
    rows = []
    rng = np.random.RandomState(seed)
    warm = (data["phase"] == 0) & (np.arange(len(data["phase"])) * float(data["dt"]) >= 2.)
    stand_mean = torch.tensor(data["latent"][warm].reshape(-1, 16).mean(axis=0))
    stand_mean = F.normalize(stand_mean, dim=0)
    with torch.inference_mode():
        for i, (name, _) in enumerate(cases):
            sl = slice(i * replicas, (i + 1) * replicas)
            obs = torch.tensor(data["obs"][data["phase"] == 1, sl].reshape(-1, 270))
            base, z = split_policy(policy, obs)
            normal = policy.actor(torch.cat((base, z), dim=-1))
            assert torch.allclose(normal, policy(obs), atol=1e-6)
            mean = F.normalize(z.mean(dim=0, keepdim=True), dim=-1).expand_as(z)
            donors = torch.tensor(rng.permutation(len(z)), dtype=torch.long)
            for mode, replacement in (("shuffle", z[donors]), ("constant", mean),
                                      ("stand_constant", stand_mean.expand_as(z)),
                                      ("zero", torch.zeros_like(z))):
                changed = policy.actor(torch.cat((base, replacement), dim=-1))
                delta = (changed - normal).numpy()
                rms = float(np.sqrt(np.mean(delta ** 2)))
                rows.append({"case": name, "mode": mode, "samples": len(z),
                             "action_delta_rms": rms,
                             "action_delta_p95_abs": float(np.percentile(np.abs(delta), 95)),
                             "normal_action_rms": float(normal.square().mean().sqrt()),
                             "delta_relative_to_normal_rms": rms / max(float(normal.square().mean().sqrt()), 1e-12),
                             "joint_target_delta_rms_rad": rms * .25,
                             "latent_mean_dimension_std": float(z.std(dim=0).mean()),
                             "latent_mean_norm": float(z.mean(dim=0).norm())})
    return rows


def closed_metrics(data, cases, modes, replicas):
    rows = []
    for m, mode in enumerate(modes):
        for c, (name, command) in enumerate(cases):
            for r in range(replicas):
                idx = m * len(cases) * replicas + c * replicas + r
                whole = data["state"][:, idx]
                for phase in (1, 2):
                    mask = data["phase"] == phase
                    s = whole[mask]
                    first = int(np.flatnonzero(mask)[0])
                    target = np.array(command) if phase == 1 else np.zeros(3)
                    actual = s[:, [3, 4, 5]]
                    rows.append({"case": name, "mode": mode, "replica": r,
                        "phase": "active" if phase == 1 else "stand_after",
                        "vx_rmse": float(np.sqrt(np.mean((actual[:, 0] - target[0]) ** 2))),
                        "yaw_rmse": float(np.sqrt(np.mean((actual[:, 2] - target[2]) ** 2))),
                        "xy_speed_mean": float(np.linalg.norm(actual[:, :2], axis=1).mean()),
                        "xy_displacement_m": float(np.linalg.norm(s[-1, :2] - whole[max(0, first-1), :2])),
                        "x_progress_m": float(s[-1, 0] - whole[max(0, first-1), 0]),
                        "final_world_x_m": float(s[-1, 0]),
                        "height_min_m": float(s[:, 6].min()),
                        "tilt_rms_deg": float(np.rad2deg(np.sqrt(np.mean(s[:, 7] ** 2)))),
                        "reward_mean": float(s[:, 8].mean()),
                        "fell": bool(((s[:, 6] < .13) | (s[:, 7] > np.deg2rad(60))).any()),
                        "stairs_completed": bool(name == "stairs_forward" and s[-1, 0] >= 5.
                                                 and not ((s[:, 6] < .13) | (s[:, 7] > np.deg2rad(60))).any())})
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    if args.terrain not in ("flat", "stairs") or args.replicas < 1:
        raise ValueError("terrain must be flat/stairs; replicas >= 1")
    if args.rl_device != "cpu":
        raise ValueError("This reproducible diagnostic currently uses CPU inference")
    torch.set_num_threads(1)
    path = Path(args.policy).resolve()
    policy = torch.jit.load(str(path), map_location="cpu").eval()
    cases = STAIR_CASES if args.terrain == "stairs" else FLAT_CASES
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    # Keep simulator size, actor placement and lane precision identical across
    # reference and intervention runs. Different environment counts can change
    # CPU PhysX contact trajectories, especially on risers.
    full_ref, _ = rollout(args, policy, cases, ("normal",) * len(MODES))
    n = len(cases) * args.replicas
    ref = {k: (v[:, :n] if k in ("obs", "latent", "action", "state") else v)
           for k, v in full_ref.items()}
    sensitivity = offline(policy, ref, cases, args.replicas, args.seed)
    data, cfg = rollout(args, policy, cases, MODES, ref)
    metrics = closed_metrics(data, cases, MODES, args.replicas)
    # Normal reference and the paired normal group must agree closely. This
    # checks initial pairing despite differences in the total environment count.
    ref_position = ref["state"][:, :, :3] - ref["state"][:1, :, :3]
    repeat_position = data["state"][:, :n, :3] - data["state"][:1, :n, :3]
    repeat_error = float(np.max(np.abs(ref_position - repeat_position)))
    if repeat_error > .001:
        raise RuntimeError("Normal reference did not reproduce: position error {} m".format(repeat_error))
    meta = {"policy": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "terrain": args.terrain, "replicas": args.replicas, "seed": args.seed,
            "modes": MODES, "cases": cases, "schedule_s": [3., 10., 6.],
            "state_columns": ["x", "y", "z", "vx", "vy", "wz", "ground_height", "tilt_rad", "reward"],
            "reference_repeat_max_position_error_m": repeat_error,
            "config": cfg, "offline": sensitivity, "closed_loop": metrics,
            "limitations": ["Interventions break observation-latent consistency; zero is outside the unit sphere.",
                            "Three small initial-condition perturbations are not independent training seeds.",
                            "This tests dependence, not semantic terrain classification."]}
    (out / "summary.json").write_text(json.dumps(meta, indent=2))
    write_csv(out / "offline.csv", sensitivity)
    write_csv(out / "closed_loop.csv", metrics)
    np.savez_compressed(out / "reference.npz", **ref)
    np.savez_compressed(out / "interventions.npz", **data)
    print("REFERENCE REPEAT ERROR", repeat_error, flush=True)
    print("OFFLINE", json.dumps(sensitivity), flush=True)
    print("Saved", out, flush=True)


if __name__ == "__main__":
    main()

"""Low-memory terrain latent sampling and CPU t-SNE (run collect/plot separately).

collect --checkpoint PATH --output DIR --sim_device cuda:0 --pipeline gpu --headless
plot --output DIR
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

NAMES = ("flat", "rough", "slope_up", "slope_down", "stairs_up", "stairs_down")


def collect():
    # Isaac Gym must precede torch imports. Plotting never imports either.
    import isaacgym
    from isaacgym import gymutil, gymtorch, terrain_utils
    from types import MethodType
    import torch
    from scipy.ndimage import gaussian_filter
    from legged_gym.envs import LeggedRobot
    from legged_gym.utils import task_registry, class_to_dict, set_seed
    from rsl_rl.modules import HIMActorCritic
    import play_minich_flat_height as flat

    args = gymutil.parse_arguments(description=__doc__, custom_parameters=[
        {"name": "--checkpoint", "type": str, "required": True},
        {"name": "--output", "type": str, "required": True},
        {"name": "--headless", "action": "store_true"},
        {"name": "--episodes", "type": int, "default": 4},
        {"name": "--seconds", "type": float, "default": 9.},
        {"name": "--seed", "type": int, "default": 20261016},
    ])
    if args.episodes < 4 or args.seconds < 5:
        raise ValueError("Use >=4 episodes and >=5 seconds for matched-speed holdout")
    args.sim_device = ("cuda:" + str(args.compute_device_id)) if args.sim_device_type == "cuda" else "cpu"
    args.num_envs = 12
    torch.set_num_threads(1)
    set_seed(args.seed)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint).resolve()
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    cfg, train_cfg = task_registry.get_cfgs("minich")
    flat.configure_flat_sequence_env(cfg)
    cfg.env.num_envs = args.num_envs
    cfg.env.episode_length_s = args.seconds + 5
    cfg.rewards.log_phase_rewards = False
    cfg.commands.in_place_turn_probability = 0.
    cfg.commands.resampling_time = args.seconds + 10
    for key in dir(cfg.domain_rand):
        if isinstance(getattr(cfg.domain_rand, key), bool):
            setattr(cfg.domain_rand, key, False)
    cfg.asset.terminate_after_contacts_on = ["trunk"]
    cfg.terrain.mesh_type = "trimesh"
    cfg.terrain.horizontal_scale = .05
    cfg.terrain.vertical_scale = .005
    cfg.terrain.border_size = 2.
    cfg.sim.physx.max_gpu_contact_pairs = 2**18
    cfg.sim.physx.default_buffer_size_multiplier = 1

    class TerrainRobot(LeggedRobot):
        def create_sim(self):
            self.up_axis_idx = 2
            self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id,
                                           self.physics_engine, self.sim_params)
            h, v, b = cfg.terrain.horizontal_scale, cfg.terrain.vertical_scale, cfg.terrain.border_size
            x = np.arange(round(14 / h)) * h - b
            y = np.arange(round((args.num_envs * 3 + 2 * b) / h)) * h - b
            distance = np.clip(x - 1., 0., 7.)
            rng = np.random.RandomState(args.seed)
            heights = np.zeros((len(x), len(y)), dtype=np.float64)
            for lane in range(args.num_envs):
                kind = lane // 2
                iy = (y >= lane * 3) & (y < (lane + 1) * 3)
                if kind == 1:
                    rough = gaussian_filter(rng.uniform(-1, 1, (len(x), iy.sum())), sigma=2)
                    rough *= .025 / max(np.abs(rough).max(), 1e-9)
                    rough[x < 1.] = 0
                    heights[:, iy] = rough
                elif kind in (2, 3):
                    heights[:, iy] = ((1 if kind == 2 else -1) * .12 * distance)[:, None]
                elif kind in (4, 5):
                    profile = np.where(x >= 1., np.floor(distance / .30) + 1, 0.) * .04
                    heights[:, iy] = ((1 if kind == 4 else -1) * profile)[:, None]
            raw = np.rint(heights / v).astype(np.int16)
            vertices, triangles = terrain_utils.convert_heightfield_to_trimesh(raw, h, v, .75)
            self.terrain = SimpleNamespace(cfg=cfg.terrain, vertices=vertices,
                triangles=triangles, heightsamples=raw, tot_rows=len(x), tot_cols=len(y))
            np.savez_compressed(out / "terrain.npz", heights=raw, x=x, y=y, vertical_scale=v)
            self._create_trimesh()
            self._create_envs()

        def _get_env_origins(self):
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device)
            self.env_origins[:, 1] = torch.arange(self.num_envs, device=self.device) * 3 + 1.5

    task_registry.register("minich_terrain_tsne", TerrainRobot, cfg, train_cfg)
    env, _ = task_registry.make_env("minich_terrain_tsne", args=args, env_cfg=cfg)
    flat.install_fixed_reset(env)
    fixed_dof_reset = env._reset_dofs

    def jittered_dof_reset(self, ids):
        fixed_dof_reset(ids)
        self.dof_pos[ids] += .015 * (2 * torch.rand((len(ids), self.num_dof), device=self.device) - 1)
        ids32 = ids.to(torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(ids32), len(ids))

    env._reset_dofs = MethodType(jittered_dof_reset, env)
    policy = HIMActorCritic(cfg.env.num_observations, cfg.env.num_privileged_obs,
        cfg.env.num_one_step_observations, cfg.env.num_actions, **class_to_dict(train_cfg.policy)).to(env.device)
    saved = torch.load(str(checkpoint), map_location=env.device)
    policy.load_state_dict(saved["model_state_dict"], strict=True)
    del saved
    policy.eval()
    records = {key: [] for key in ("z", "label", "episode", "env_id", "time", "command", "velocity", "position")}
    rollouts = []
    try:
        with torch.inference_mode():
            for episode in range(args.episodes):
                flat.reset_to_fixed_state(env)
                # Two speeds, each repeated in train and held-out episodes.
                speed = (.6, .9)[episode % 2]
                alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
                for step in range(round(args.seconds / env.dt)):
                    env.commands.zero_()
                    env.commands[:, 0] = speed if step * env.dt >= 1. else 0.
                    env.obs_buf[:, :3] = env.commands[:, :3] * env.commands_scale
                    obs = env.get_observations()
                    _, z = policy.estimator(obs)
                    valid = alive & (env.root_states[:, 0] > 1.6) & (env.root_states[:, 0] < 6.)
                    valid &= (env.root_states[:, 1] - env.env_origins[:, 1]).abs() < .8
                    valid &= -env.projected_gravity[:, 2] > .85
                    valid &= env._get_base_heights() > .16
                    if step % 5 == 0 and step * env.dt > 2.:
                        ids = valid.nonzero().flatten()
                        if len(ids):
                            values = dict(z=z[ids], label=ids // 2, env_id=ids,
                                episode=torch.full_like(ids, episode),
                                time=torch.full((len(ids),), step * env.dt, device=env.device),
                                command=env.commands[ids, :3], velocity=env.base_lin_vel[ids],
                                position=env.root_states[ids, :3])
                            for key, value in values.items():
                                records[key].append(value.cpu().numpy().copy())
                    action = policy.act_inference(obs)
                    if not torch.isfinite(z).all() or not torch.isfinite(action).all():
                        raise RuntimeError("Non-finite latent/action")
                    _, _, _, done, _, _, _ = env.step(action)
                    alive &= ~done.bool()
                    if step % 200 == 0:
                        print(f"episode={episode} time={step * env.dt:.1f}s alive={int(alive.sum())}", flush=True)
                rollouts.append(dict(episode=episode, speed=speed, alive=int(alive.sum())))
        if not records["z"]:
            raise RuntimeError("No valid terrain-interior samples")
        data = {key: np.concatenate(value) for key, value in records.items()}
        np.savez_compressed(out / "samples.npz", **data)
        asset = Path(cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=str(Path(__file__).resolve().parents[2])))
        meta = dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
            urdf=str(asset), urdf_sha256=hashlib.sha256(asset.read_bytes()).hexdigest(),
            seed=args.seed, terrain_names=NAMES, sim_device=env.device,
            gpu_physx=bool(env.sim_params.physx.use_gpu), gpu_pipeline=bool(env.sim_params.use_gpu_pipeline),
            headless=args.headless, num_envs=env.num_envs, dt=env.dt,
            samples_per_terrain={name: int((data["label"] == i).sum()) for i, name in enumerate(NAMES)},
            rollouts=rollouts, config=class_to_dict(cfg),
            cuda_peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0,
            terrain_parameters=dict(start_x=1., sample_x=[1.6, 6.], slope=.12, stair_height=.04,
                                    stair_depth=.30, rough_max_abs_height=.025),
            initial_joint_jitter_rad=.015,
            limitations=["One checkpoint, synthetic mild terrains, two replicas per terrain.",
                         "Latent mixes terrain response, posture and motion; coloring is ground truth.",
                         "Samples exclude reset, fallen robots, boundaries and initial flat approach."])
        (out / "metadata.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta["samples_per_terrain"]), flush=True)
    finally:
        if env.viewer is not None:
            env.gym.destroy_viewer(env.viewer)
        env.gym.destroy_sim(env.sim)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits

    parser = argparse.ArgumentParser(description="CPU plots and episode-held-out linear probe")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    data = np.load(out / "samples.npz")
    z, labels = data["z"], data["label"]
    if not np.isfinite(z).all() or not np.allclose(np.linalg.norm(z, axis=1), 1., atol=1e-4):
        raise ValueError("Expected finite unit-length latent")
    # Equal class counts keep visual density and held-out scoring comparable.
    rng = np.random.RandomState(42)
    count = min(600, min(int((labels == i).sum()) for i in range(len(NAMES))))
    if count < 20:
        raise ValueError("Too few samples for at least one terrain; inspect sampling")
    ids = np.concatenate([rng.choice(np.flatnonzero(labels == i), count, replace=False) for i in range(len(NAMES))])
    colors = plt.get_cmap("tab10")
    terrain = np.load(out / "terrain.npz")
    fig, ax = plt.subplots(figsize=(9, 4))
    for i, name in enumerate(NAMES):
        center = np.argmin(abs(terrain["y"] - (2 * i * 3 + 1.5)))
        ax.plot(terrain["x"], terrain["heights"][:, center] * terrain["vertical_scale"],
                color=colors(i), label=name)
    ax.axvspan(1.6, 6., alpha=.1, color="green", label="Sample region")
    ax.set(xlim=(-.5, 7.), xlabel="World x (m)", ylabel="Ground height (m)")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "terrain_profiles.png", dpi=180)
    plt.close(fig)
    embeddings = {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    with threadpool_limits(limits=2):
        for row, seed in enumerate((42, 7)):
            for col, perplexity in enumerate((10, 30, 50)):
                init = "pca" if row == 0 else "random"
                xy = TSNE(n_components=2, perplexity=perplexity, init=init, learning_rate=200.,
                          random_state=seed, n_iter=1200, n_jobs=2).fit_transform(z[ids])
                embeddings[f"seed{seed}_p{perplexity}"] = xy
                ax = axes[row, col]
                for i, name in enumerate(NAMES):
                    mask = labels[ids] == i
                    ax.scatter(*xy[mask].T, s=5, alpha=.5, color=colors(i), label=name, rasterized=True)
                ax.set_title(f"perplexity={perplexity}, init={init}, seed={seed}")
                ax.set_xticks([])
                ax.set_yticks([])
        axes[0, 0].legend(markerscale=3, fontsize=8)
        fig.suptitle(f"HIM 16-D latent, equal samples: {count} per terrain")
        fig.tight_layout()
        fig.savefig(out / "terrain_tsne.png", dpi=180)
        plt.close(fig)
        xy = embeddings["seed42_p30"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, values, title in zip(axes,
                (data["command"][ids, 0], data["velocity"][ids, 0], data["episode"][ids]),
                ("Command vx (m/s)", "Actual vx (m/s)", "Episode")):
            artist = ax.scatter(*xy.T, c=values, s=6, cmap="viridis", alpha=.7)
            ax.set_title(title)
            fig.colorbar(artist, ax=ax)
        fig.tight_layout()
        fig.savefig(out / "terrain_tsne_controls.png", dpi=180)
        plt.close(fig)
        # Never train/test on randomly split adjacent frames or t-SNE coordinates.
        episodes = data["episode"]
        train = episodes < 2
        test = episodes >= 2
        probe = LogisticRegression(C=1e4, max_iter=3000, class_weight="balanced")
        probe.fit(z[train], labels[train])
        prediction = probe.predict(z[test])
        score = balanced_accuracy_score(labels[test], prediction)
        matrix = confusion_matrix(labels[test], prediction, labels=np.arange(len(NAMES)), normalize="true")
    np.savez_compressed(out / "embeddings.npz", sample_ids=ids, **embeddings)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(NAMES)), NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(NAMES)), NAMES)
    ax.set(xlabel="Predicted terrain", ylabel="True terrain", title=f"16-D linear probe, held-out episodes: {score:.1%}")
    for i in range(len(NAMES)):
        for j in range(len(NAMES)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if matrix[i, j] > .5 else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out / "terrain_probe.png", dpi=180)
    plt.close(fig)
    metrics = dict(balanced_accuracy=score, chance=1 / len(NAMES), samples_per_class_in_tsne=count,
        train_episodes=np.unique(episodes[train]).tolist(), test_episodes=np.unique(episodes[test]).tolist(),
        latent_mean_dimension_std=float(z.std(axis=0).mean()),
        latent_mean_norm=float(np.linalg.norm(z.mean(axis=0))), confusion_matrix=matrix.tolist(),
        probe="LogisticRegression C=10000, no per-dimension scaling, raw unit latent",
        limitation="Held-out episodes reuse terrain geometry; this does not test unseen terrain generalization.")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    import sys
    mode = sys.argv.pop(1) if len(sys.argv) > 1 else ""
    if mode == "collect":
        collect()
    elif mode == "plot":
        plot()
    else:
        raise SystemExit(__doc__)

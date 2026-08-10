# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
"""固定平地回放 Mini Cheetah HIM 策略，并输出仿真器原始足端支撑力。"""

import math
from types import MethodType

import isaacgym
from isaacgym import gymtorch
import numpy as np
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

# 高于训练目标高度；用于让模型在平地上方生成，再由重力平稳落到支撑姿态，
# 避免以足端/小腿穿入地面的状态开始仿真。
INITIAL_BASE_HEIGHT_M = 0.36
TASK_NAME = "minich"
LOAD_RUN = "Aug06_16-16-21_smoke_minich"
CHECKPOINT = 1000
ACTION_SCALE = 0.25
DEFAULT_POSE_HOLD_S = 1.0
FORWARD_SPEED_M_S = 1.0
TURN_YAW_RATE_RAD_S = 1.5
STAND_BEFORE_S = 2.0
FORWARD_S = 0.0
TURN_S = 6.0
STAND_AFTER_S = 6.0
ENABLE_TERMINATION_RESET = False
FOOT_FORCE_PRINT_INTERVAL_S = 0.1

# 与参考脚本相同的初始化相机视角。
CAMERA_OFFSET = np.array([-2.5, -2.5, 1.5], dtype=np.float64)
CAMERA_LOOKAT_OFFSET = np.array([0.0, 0.0, 0.2], dtype=np.float64)
LEG_ORDER = ("FL", "FR", "RL", "RR")


def scheduled_phase(step, env_dt):
    stand_before_steps = int(round(STAND_BEFORE_S / env_dt))
    forward_end_step = stand_before_steps + int(round(FORWARD_S / env_dt))
    turn_end_step = forward_end_step + int(round(TURN_S / env_dt))
    if step < stand_before_steps:
        return "stand_before"
    if step < forward_end_step:
        return "forward"
    if step < turn_end_step:
        return "turn"
    return "stand_after"


def scheduled_command(step, env_dt):
    phase = scheduled_phase(step, env_dt)
    if phase == "forward":
        return (FORWARD_SPEED_M_S, 0.0, 0.0)
    if phase == "turn":
        return (0.0, 0.0, TURN_YAW_RATE_RAD_S)
    return (0.0, 0.0, 0.0)


def total_play_steps(env_dt):
    return int(round((STAND_BEFORE_S + FORWARD_S + TURN_S + STAND_AFTER_S) / env_dt))


def configure_flat_sequence_env(env_cfg):
    """仅修改回放实例；训练配置和 checkpoint 不受影响。"""
    env_cfg.env.num_envs = 1
    env_cfg.env.episode_length_s = max(20.0, DEFAULT_POSE_HOLD_S + STAND_BEFORE_S + FORWARD_S + TURN_S + STAND_AFTER_S + 1.0)
    env_cfg.init_state.pos[2] = INITIAL_BASE_HEIGHT_M
    env_cfg.terrain.mesh_type = "plane"
    env_cfg.terrain.measure_heights = True
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.domain_rand.randomize_link_mass = False
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.commands.resampling_time = env_cfg.env.episode_length_s + 1.0
    env_cfg.commands.ranges.lin_vel_x = [0.0, 0.0]
    env_cfg.commands.ranges.lin_vel_y = [0.0, 0.0]
    env_cfg.commands.ranges.ang_vel_yaw = [0.0, 0.0]
    env_cfg.control.action_scale = ACTION_SCALE
    env_cfg.asset.terminate_after_contacts_on = []


def set_command(env, command):
    env.commands.zero_()
    env.commands[:, :3] = torch.as_tensor(command, device=env.device, dtype=env.commands.dtype)


def install_fixed_reset(env):
    """照参考脚本固定 reset，避免回放重新随机化初始姿态。"""
    def reset_dofs_fixed(self, env_ids):
        if len(env_ids) == 0:
            return
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids),
        )

    def reset_root_states_fixed(self, env_ids):
        if len(env_ids) == 0:
            return
        self.root_states[env_ids] = self.base_init_state
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 7:13] = 0.0
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids),
        )

    env._reset_dofs = MethodType(reset_dofs_fixed, env)
    env._reset_root_states = MethodType(reset_root_states_fixed, env)


def reset_to_fixed_state(env):
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.reset_idx(env_ids)
    set_command(env, (0.0, 0.0, 0.0))
    env.actions.zero_()
    env.last_actions.zero_()
    env.last_dof_vel.zero_()
    env.last_root_vel.zero_()
    env.compute_observations()
    return env.get_observations()


def install_termination_override(env):
    """与参考脚本一致：关闭回放中的自动 reset。"""
    if ENABLE_TERMINATION_RESET:
        return

    def check_termination_without_reset(self):
        self.reset_buf.zero_()
        self.time_out_buf = torch.zeros_like(self.reset_buf, dtype=torch.bool)

    env.check_termination = MethodType(check_termination_without_reset, env)


def aim_camera_at_robot(env):
    if env.viewer is None:
        return
    base_position = env.root_states[0, :3].detach().cpu().numpy()
    env.set_camera(base_position + CAMERA_OFFSET, base_position + CAMERA_LOOKAT_OFFSET)


class FootContactForcePrinter:
    """照参考脚本直接读取 net_contact_force_tensor 的足端 Fz。"""

    def __init__(self, env, interval_s):
        if interval_s <= 0.0:
            raise ValueError("FOOT_FORCE_PRINT_INTERVAL_S must be positive")
        self.interval_steps = max(1, int(round(interval_s / env.dt)))
        body_names = env.gym.get_actor_rigid_body_names(env.envs[0], env.actor_handles[0])
        foot_indices = env.feet_indices.detach().cpu().tolist()
        ordered_indices = []
        for leg in LEG_ORDER:
            matches = [index for index in foot_indices if body_names[index].startswith(f"{leg}_")]
            if len(matches) != 1:
                raise ValueError(f"expected one foot body for {leg}, found {matches}")
            ordered_indices.append(matches[0])
        self.foot_indices = torch.tensor(ordered_indices, device=env.device, dtype=torch.long)

    @torch.no_grad()
    def sample(self, env, step, phase):
        if step % self.interval_steps != 0:
            return
        env.gym.refresh_net_contact_force_tensor(env.sim)
        forces = env.contact_forces[0, self.foot_indices]
        support_forces = torch.clamp(forces[:, env.up_axis_idx], min=0.0)
        values = support_forces.detach().cpu().tolist()
        print(
            f"[足端地面支撑力] t={(step + 1) * env.dt:.2f}s, phase={phase} [N]; "
            + ", ".join(f"{leg}={force:.3f}" for leg, force in zip(LEG_ORDER, values))
            + f", total={sum(values):.3f}",
            flush=True,
        )


def run_default_pose_hold(env):
    hold_steps = int(round(DEFAULT_POSE_HOLD_S / env.dt))
    zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    set_command(env, (0.0, 0.0, 0.0))
    print(f"Default-pose hold: {DEFAULT_POSE_HOLD_S:.1f}s, {hold_steps} policy steps")
    for _ in range(hold_steps):
        env.step(zero_actions)


def play(args):
    args.task = TASK_NAME
    args.num_envs = 1
    args.load_run = LOAD_RUN
    args.checkpoint = CHECKPOINT
    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    configure_flat_sequence_env(env_cfg)
    train_cfg.runner.resume = True
    train_cfg.runner.load_run = LOAD_RUN
    train_cfg.runner.checkpoint = CHECKPOINT

    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    install_fixed_reset(env)
    obs = reset_to_fixed_state(env)
    aim_camera_at_robot(env)
    run_default_pose_hold(env)

    runner, _ = task_registry.make_alg_runner(env=env, name=TASK_NAME, args=args, train_cfg=train_cfg)
    obs = reset_to_fixed_state(env)
    install_termination_override(env)
    aim_camera_at_robot(env)
    force_printer = FootContactForcePrinter(env, FOOT_FORCE_PRINT_INTERVAL_S)
    policy = runner.get_inference_policy(device=env.device)
    play_steps = total_play_steps(env.dt)
    print(
        f"Flat viewer: task={TASK_NAME}, run={LOAD_RUN}, checkpoint={CHECKPOINT}; "
        f"camera_offset={CAMERA_OFFSET.tolist()}, steps={play_steps}",
        flush=True,
    )

    for step in range(play_steps):
        phase = scheduled_phase(step, env.dt)
        set_command(env, scheduled_command(step, env.dt))
        env.compute_observations()
        obs = env.get_observations()
        actions = policy(obs.detach())
        # HIMLoco LeggedRobot 返回：obs、privileged_obs、reward、done、extras、termination_ids、termination_obs。
        obs, _, _, _, _, _, _ = env.step(actions.detach())
        force_printer.sample(env, step, phase)


if __name__ == "__main__":
    play(get_args())

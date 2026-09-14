# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
"""固定平地回放 Mini Cheetah HIM 策略，可输出基座高度、姿态和足端支撑力。"""

import math
from pathlib import Path
from types import MethodType
from xml.etree import ElementTree

import isaacgym
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
import numpy as np
import torch

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

# 高于训练目标高度；用于让模型在平地上方生成，再由重力平稳落到支撑姿态，
# 避免以足端/小腿穿入地面的状态开始仿真。
INITIAL_BASE_HEIGHT_M = 0.36
TASK_NAME = "minich"
# 最右侧机器人始终加载 rough_minich 日志目录中最新训练的最新 checkpoint。
# 前两个元组为手动指定项：(显示名称, run 目录名, checkpoint 编号)。
# 请按需替换为实际存在的 run 与 checkpoint；两个项允许暂时指向同一模型。
POLICY_SPECS = (
    ("manual_1", "Sep12_10-10-23_coupled_hip_test", -1),
    ("manual_2", "Sep12_10-10-23_coupled_hip_test", -1),
    ("latest", -1, -1),
)
ENV_SPACING_M = 1.0
ACTION_SCALE = 0.25
DEFAULT_POSE_HOLD_S = 1.0
FORWARD_SPEED_M_S = 1.0#前进速度
TURN_YAW_RATE_RAD_S = 1.5
STAND_BEFORE_S = 6.0#关节模式
FORWARD_S = 3.0#前进时间
TURN_S = 20.0#旋转时间
STAND_AFTER_S = 10.0#后站立时间
ENABLE_TERMINATION_RESET = False
HEIGHT_PRINT_INTERVAL_S = 1.0
# 设为 0.0 关闭基座 IMU 角度打印；正数为打印间隔（仿真秒），输出单位为度。
IMU_PRINT_INTERVAL_S = 1.0
# 设为 0.0 关闭足端支撑力打印；设为正数时按该间隔打印四足 Fz。
FOOT_FORCE_PRINT_INTERVAL_S = 0.0

# 初始化相机对准最右侧（latest）策略的机器狗，而不是三台机器狗的中点。
# 相机距离缩短，使初始画面只包含该机器狗；负 x/负 y 表示从左后上方看。
CAMERA_OFFSET = np.array([-1.35, -1.35, 0.85], dtype=np.float64)
CAMERA_LOOKAT_OFFSET = np.array([0.0, 0.0, 0.18], dtype=np.float64)
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
    env_cfg.env.num_envs = len(POLICY_SPECS)
    env_cfg.env.env_spacing = ENV_SPACING_M
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
    # 某些已安装/旧版本配置没有该字段；回放时显式补齐，确保与训练使用
    # 的固定周期步态保持一致，同时避免直接访问属性时报 AttributeError。
    env_cfg.rewards.raibert_gait_period_s = getattr(
        env_cfg.rewards, "raibert_gait_period_s", 0.6
    )


def set_command(env, command):
    env.commands.zero_()
    env.commands[:, :3] = torch.as_tensor(command, device=env.device, dtype=env.commands.dtype)


def set_line_origins(env):
    """在单一 Isaac Gym 仿真中，将三台机器人沿 x 轴固定排开。"""
    if env.num_envs != len(POLICY_SPECS):
        raise ValueError("num_envs must match the number of policy specifications")
    env.env_origins.zero_()
    env.env_origins[:, 0] = torch.arange(env.num_envs, device=env.device) * ENV_SPACING_M


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


def aim_camera_at_latest_robot(env):
    """初始化时只对准相机视角最右侧的 latest 策略机器狗。"""
    if env.viewer is None:
        return
    latest_robot_index = len(POLICY_SPECS) - 1
    if POLICY_SPECS[latest_robot_index][0] != "latest":
        raise ValueError("the rightmost POLICY_SPECS entry must be the latest policy")
    latest_position = env.root_states[latest_robot_index, :3].detach().cpu().numpy()
    env.set_camera(
        latest_position + CAMERA_OFFSET,
        latest_position + CAMERA_LOOKAT_OFFSET,
    )


@torch.no_grad()
def get_base_heights_above_ground(env):
    """返回每台机器狗的世界高度、相对地面高度和目标偏差。"""
    env.gym.refresh_actor_root_state_tensor(env.sim)
    world_heights = env.root_states[:, 2].detach().cpu().tolist()
    ground_heights = env._get_base_heights().detach().cpu().tolist()
    target_height = float(env.cfg.rewards.base_height_target)
    return [
        (world_height, ground_height, ground_height - target_height)
        for world_height, ground_height in zip(world_heights, ground_heights)
    ]


@torch.no_grad()
def get_base_imu_angles_deg(env):
    """将基座世界系四元数 (x, y, z, w) 转为 ZYX 欧拉角 roll/pitch/yaw（度）。

    使用仿真姿态真值；roll/yaw 范围为 [-180, 180]，pitch 为 [-90, 90]。
    """
    env.gym.refresh_actor_root_state_tensor(env.sim)
    x, y, z, w = env.root_states[:, 3:7].unbind(dim=-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), min=-1.0, max=1.0))
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (torch.stack((roll, pitch, yaw), dim=-1) * (180.0 / math.pi)).cpu().tolist()


class FootContactForcePrinter:
    """直接读取 Isaac Gym net-contact tensor 的四足向上支撑力。"""

    def __init__(self, env, interval_s):
        if interval_s <= 0.0:
            raise ValueError("FOOT_FORCE_PRINT_INTERVAL_S must be positive when enabled")
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
        forces = env.contact_forces[:, self.foot_indices]
        support_forces = torch.clamp(forces[:, :, env.up_axis_idx], min=0.0)
        robot_forces = []
        for policy_spec, values in zip(POLICY_SPECS, support_forces.detach().cpu().tolist()):
            robot_forces.append(
                f"{policy_spec[0]}: "
                + ", ".join(f"{leg}={force:.3f}" for leg, force in zip(LEG_ORDER, values))
                + f", total={sum(values):.3f}"
            )
        print(
            f"[足端地面支撑力] t={(step + 1) * env.dt:.2f}s, phase={phase} [N]; "
            + " | ".join(robot_forces),
            flush=True,
        )


class LatestPolicyRewardMeter:
    """统计最右侧 latest 机器人的每阶段奖励。"""

    def __init__(self, env, robot_index):
        self.robot_index = robot_index
        self.reward_names = tuple(env.episode_sums.keys())
        self.phases = ("stand_before", "forward", "turn", "stand_after")
        self.phase = self.phases[0]
        self.stats = {
            phase: {"steps": 0, "terms": torch.zeros(len(self.reward_names), device=env.device), "total": 0.0}
            for phase in self.phases
        }
        self.raibert_stats = {
            phase: {"steps": 0, "total": 0.0} for phase in self.phases
        }

    def set_phase(self, phase):
        self.phase = phase

    @torch.no_grad()
    def sample(self, term_rewards, total_reward):
        stats = self.stats[self.phase]
        stats["steps"] += 1
        stats["terms"] += term_rewards
        stats["total"] += float(total_reward)
        if "raibert_heuristic" in self.reward_names:
            index = self.reward_names.index("raibert_heuristic")
            self.raibert_stats[self.phase]["steps"] += 1
            self.raibert_stats[self.phase]["total"] += float(term_rewards[index])

    def print_summary(self):
        print("\n最新策略 Raibert 奖励统计：")
        for phase in self.phases:
            stats = self.raibert_stats[phase]
            mean = stats["total"] / stats["steps"] if stats["steps"] else 0.0
            print(f"{phase:13s}: total={stats['total']:.6f}, mean/step={mean:.8f}")
        print("\n最新策略（最右侧机器人）回放奖励汇总：")
        print("phase         steps  total reward  mean/step")
        for phase in self.phases:
            stats = self.stats[phase]
            if stats["steps"]:
                print(f"{phase:13s} {stats['steps']:5d} {stats['total']:13.6f} {stats['total'] / stats['steps']:10.8f}")
        print("\nphase         reward term                         total  mean/step")
        for phase in self.phases:
            stats = self.stats[phase]
            if stats["steps"]:
                for name, value in zip(self.reward_names, stats["terms"].detach().cpu().tolist()):
                    if name == "raibert_heuristic" and phase in ("stand_before", "stand_after"):
                        raibert = self.raibert_stats[phase]
                        if raibert["steps"]:
                            value = raibert["total"]
                    print(f"{phase:13s} {name:35s} {value:12.6f} {value / stats['steps']:10.8f}")


def install_latest_policy_reward_meter(env):
    """只记录最右侧 latest 策略在原始奖励计算产生的增量。"""
    robot_index = len(POLICY_SPECS) - 1
    if POLICY_SPECS[robot_index][0] != "latest":
        raise ValueError("the rightmost POLICY_SPECS entry must be the latest policy")
    meter = LatestPolicyRewardMeter(env, robot_index)
    original_compute_reward = env.compute_reward

    def compute_reward_and_measure(self):
        before = torch.stack([self.episode_sums[name][robot_index] for name in meter.reward_names]).clone()
        original_compute_reward()
        after = torch.stack([self.episode_sums[name][robot_index] for name in meter.reward_names])
        meter.sample(after - before, self.rew_buf[robot_index])

    env.compute_reward = MethodType(compute_reward_and_measure, env)
    return meter


def run_default_pose_hold(env):
    hold_steps = int(round(DEFAULT_POSE_HOLD_S / env.dt))
    zero_actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
    set_command(env, (0.0, 0.0, 0.0))
    print(f"Default-pose hold: {DEFAULT_POSE_HOLD_S:.1f}s, {hold_steps} policy steps")
    for _ in range(hold_steps):
        env.step(zero_actions)


def get_raibert_foot_body_xy(env):
    """以训练 Raibert 项的 [FL, FR, RL, RR] 顺序返回当前足端机身系 XY。"""
    env.gym.refresh_actor_root_state_tensor(env.sim)
    env.gym.refresh_rigid_body_state_tensor(env.sim)
    foot_world_relative = (
        env.feet_pos[:, env.raibert_foot_order, :]
        - env.root_states[:, :3].unsqueeze(1)
    )
    foot_body = quat_rotate_inverse(
        env.base_quat.repeat_interleave(4, dim=0),
        foot_world_relative.reshape(-1, 3),
    ).view(env.num_envs, 4, 3)
    return foot_body[:, :, :2]


def get_raibert_desired_foot_xy(env, command=None):
    """按训练中的固定周期 Raibert 项计算名义足端目标。"""
    half_length = 0.5 * env.cfg.rewards.raibert_stance_length
    half_width = 0.5 * env.cfg.rewards.raibert_stance_width
    x_nom = torch.tensor(
        [half_length, half_length, -half_length, -half_length], device=env.device
    )
    y_nom = torch.tensor(
        [half_width, -half_width, half_width, -half_width], device=env.device
    )
    return torch.stack((x_nom, y_nom), dim=-1)


def print_raibert_default_pose_comparison(env, default_foot_body_xy):
    """输出当前阶段首条指令的 Raibert 目标与默认关节姿态足端的偏差。"""
    command = scheduled_command(
        int(round(STAND_BEFORE_S / env.dt)), env.dt
    )
    desired_foot_xy = get_raibert_desired_foot_xy(env)
    errors = default_foot_body_xy - desired_foot_xy.unsqueeze(0)
    squared_errors = torch.sum(torch.square(errors), dim=-1)
    print("\nRaibert 落足点平均误差（STAND_BEFORE 结束；单位 m）：")
    gait_period_s = getattr(env.cfg.rewards, "raibert_gait_period_s", 0.6)
    print(
        f"固定步态周期: {gait_period_s:.3f} s; "
        f"下一阶段指令: vx={command[0]:.3f} m/s, "
        f"vy={command[1]:.3f} m/s, wz={command[2]:.3f} rad/s"
    )
    reward_scale = env.cfg.rewards.scales.raibert_heuristic
    for policy_spec, foot_errors, foot_squared_errors in zip(
        POLICY_SPECS,
        errors.detach().cpu().tolist(),
        squared_errors.detach().cpu().tolist(),
    ):
        mean_error = math.sqrt(sum(foot_squared_errors) / len(foot_squared_errors))
        raw_penalty = sum(foot_squared_errors)
        print(
            f"{policy_spec[0]}: 平均误差={mean_error:.6f}, "
            f"原始惩罚={raw_penalty:.6f}, 奖励贡献={reward_scale * raw_penalty:.6f}"
        )


def load_policies(env, args, train_cfg):
    """为同一个向量化环境逐一构建并加载三个独立 HIM 推理策略。"""
    policies = []
    for label, load_run, checkpoint in POLICY_SPECS:
        train_cfg.runner.resume = True
        train_cfg.runner.load_run = load_run
        train_cfg.runner.checkpoint = checkpoint
        runner, _ = task_registry.make_alg_runner(
            env=env, name=TASK_NAME, args=args, train_cfg=train_cfg
        )
        policies.append(runner.get_inference_policy(device=env.device))
        print(
            f"已加载策略 {label}: run={load_run}, checkpoint={checkpoint}",
            flush=True,
        )
    return policies


def play(args):
    args.task = TASK_NAME
    args.num_envs = len(POLICY_SPECS)
    # make_alg_runner 读取 args 中的覆盖值；策略加载统一由 POLICY_SPECS 控制。
    args.load_run = None
    args.checkpoint = None
    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK_NAME)
    configure_flat_sequence_env(env_cfg)

    env, _ = task_registry.make_env(name=TASK_NAME, args=args, env_cfg=env_cfg)
    set_line_origins(env)
    install_fixed_reset(env)
    reset_to_fixed_state(env)
    aim_camera_at_latest_robot(env)
    run_default_pose_hold(env)
    default_foot_body_xy = get_raibert_foot_body_xy(env).clone()

    policies = load_policies(env, args, train_cfg)
    obs = reset_to_fixed_state(env)
    install_termination_override(env)
    reward_meter = install_latest_policy_reward_meter(env)
    aim_camera_at_latest_robot(env)
    play_steps = total_play_steps(env.dt)
    height_interval_steps = (
        max(1, int(round(HEIGHT_PRINT_INTERVAL_S / env.dt)))
        if HEIGHT_PRINT_INTERVAL_S > 0.0 else None
    )
    imu_interval_steps = (
        max(1, int(round(IMU_PRINT_INTERVAL_S / env.dt)))
        if IMU_PRINT_INTERVAL_S > 0.0 else None
    )
    force_printer = (
        FootContactForcePrinter(env, FOOT_FORCE_PRINT_INTERVAL_S)
        if FOOT_FORCE_PRINT_INTERVAL_S > 0.0 else None
    )
    target_height = float(env.cfg.rewards.base_height_target)
    print(
        f"Flat viewer: task={TASK_NAME}, robots={env.num_envs}, spacing={ENV_SPACING_M:.1f}m; "
        f"camera_offset={CAMERA_OFFSET.tolist()}, steps={play_steps}",
        flush=True,
    )
    if HEIGHT_PRINT_INTERVAL_S > 0.0:
        print(
            f"基座高度打印间隔: {HEIGHT_PRINT_INTERVAL_S:.3f}s, "
            f"目标高度: {target_height:.3f}m",
            flush=True,
        )
    else:
        print("基座高度打印: 已关闭（HEIGHT_PRINT_INTERVAL_S=0.0）", flush=True)
    if imu_interval_steps is None:
        print("基座 IMU 角度打印: 已关闭（IMU_PRINT_INTERVAL_S=0.0）", flush=True)
    else:
        print(f"基座 IMU 角度打印间隔: {IMU_PRINT_INTERVAL_S:.3f}s，单位: 度", flush=True)
    if force_printer is None:
        print("足端支撑力打印: 已关闭（FOOT_FORCE_PRINT_INTERVAL_S=0.0）", flush=True)
    else:
        print(f"足端支撑力打印间隔: {FOOT_FORCE_PRINT_INTERVAL_S:.3f}s", flush=True)

    for step in range(play_steps):
        phase = scheduled_phase(step, env.dt)
        reward_meter.set_phase(phase)
        set_command(env, scheduled_command(step, env.dt))
        env.compute_observations()
        obs = env.get_observations()
        # 每个 checkpoint 仅为与其索引相同的一台机器狗产生动作。
        actions = torch.cat(
            [policy(obs[index:index + 1].detach()) for index, policy in enumerate(policies)],
            dim=0,
        )
        # HIMLoco LeggedRobot 返回：obs、privileged_obs、reward、done、extras、termination_ids、termination_obs。
        obs, _, _, _, _, _, _ = env.step(actions.detach())
        if step + 1 == int(round(STAND_BEFORE_S / env.dt)):
            print_raibert_default_pose_comparison(env, default_foot_body_xy)
        if height_interval_steps is not None and step % height_interval_steps == 0:
            height_lines = []
            for policy_spec, (world_height, ground_height, height_error) in zip(
                POLICY_SPECS, get_base_heights_above_ground(env)
            ):
                height_lines.append(
                    f"{policy_spec[0]}: 世界={world_height:.4f}m, "
                    f"相对地面={ground_height:.4f}m, 相对目标={height_error:+.4f}m"
                )
            print(
                f"[基座高度] t={(step + 1) * env.dt:7.2f}s; " + " | ".join(height_lines),
                flush=True,
            )
        if imu_interval_steps is not None and step % imu_interval_steps == 0:
            imu_lines = [
                f"{policy_spec[0]}: roll={roll:+.2f}°, pitch={pitch:+.2f}°, yaw={yaw:+.2f}°"
                for policy_spec, (roll, pitch, yaw) in zip(
                    POLICY_SPECS, get_base_imu_angles_deg(env)
                )
            ]
            print(
                f"[基座 IMU 角度] t={(step + 1) * env.dt:7.2f}s; " + " | ".join(imu_lines),
                flush=True,
            )
        if force_printer is not None:
            force_printer.sample(env, step, phase)

    reward_meter.print_summary()


if __name__ == "__main__":
    play(get_args())

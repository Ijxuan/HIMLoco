# SPDX-License-Identifier: BSD-3-Clause
"""在独立 Isaac Gym 仿真中测量 Mini Cheetah 默认姿态的足端坐标。"""

import argparse
from pathlib import Path

from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_rotate_inverse
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.minich.minich_config import MiniChRoughCfg


LEG_ORDER = ("FL", "FR", "RL", "RR")
FOOT_BODY_NAMES = tuple(f"{leg}_foot" for leg in LEG_ORDER)
BASE_BODY_NAME = "base"


def parse_args():
    parser = argparse.ArgumentParser(
        description="用固定基座 Isaac Gym actor 测量默认关节姿态的足端坐标"
    )
    parser.add_argument("--compute-device-id", type=int, default=0)
    parser.add_argument("--graphics-device-id", type=int, default=-1)
    return parser.parse_args()


def make_sim(gym, compute_device_id, graphics_device_id):
    sim_params = gymapi.SimParams()
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    sim_params.use_gpu_pipeline = False
    sim_params.physx.use_gpu = compute_device_id >= 0

    sim = gym.create_sim(
        max(compute_device_id, 0),
        graphics_device_id,
        gymapi.SIM_PHYSX,
        sim_params,
    )
    if sim is None:
        raise RuntimeError("无法创建 Isaac Gym 仿真")
    return sim


def load_fixed_base_asset(gym, sim):
    urdf_path = Path(LEGGED_GYM_ROOT_DIR) / "resources/robots/mini_cheetah/urdf/mini_cheetah.urdf"
    options = gymapi.AssetOptions()
    options.fix_base_link = True
    options.disable_gravity = True
    options.collapse_fixed_joints = False
    options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

    asset = gym.load_asset(sim, str(urdf_path.parent), urdf_path.name, options)
    if asset is None:
        raise RuntimeError(f"无法加载 URDF: {urdf_path}")
    return asset, urdf_path


def set_default_joint_state(gym, env, actor, asset):
    dof_names = gym.get_asset_dof_names(asset)
    configured_angles = MiniChRoughCfg.init_state.default_joint_angles
    missing = [name for name in dof_names if name not in configured_angles]
    extra = [name for name in configured_angles if name not in dof_names]
    if missing or extra:
        raise ValueError(f"默认角度与 URDF DOF 不一致: missing={missing}, extra={extra}")

    dof_states = np.zeros(len(dof_names), dtype=gymapi.DofState.dtype)
    dof_states["pos"] = np.asarray(
        [configured_angles[name] for name in dof_names], dtype=np.float32
    )
    dof_states["vel"] = 0.0
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)
    return dof_names, dof_states["pos"].copy()


def measure_foot_positions(gym, sim, env, actor):
    body_names = gym.get_actor_rigid_body_names(env, actor)
    missing = [name for name in (BASE_BODY_NAME, *FOOT_BODY_NAMES) if name not in body_names]
    if missing:
        raise ValueError(f"URDF 缺少所需刚体: {missing}; available={body_names}")

    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)

    root_states = gymtorch.wrap_tensor(gym.acquire_actor_root_state_tensor(sim))
    rigid_body_states = gymtorch.wrap_tensor(gym.acquire_rigid_body_state_tensor(sim))
    foot_indices = torch.tensor(
        [gym.find_actor_rigid_body_index(env, actor, name, gymapi.DOMAIN_SIM) for name in FOOT_BODY_NAMES],
        dtype=torch.long,
        device=rigid_body_states.device,
    )
    if torch.any(foot_indices < 0):
        raise RuntimeError(f"无法取得足端刚体索引: {foot_indices.tolist()}")

    base_position_world = root_states[0, 0:3]
    base_quaternion_world = root_states[0, 3:7]
    feet_position_world = rigid_body_states[foot_indices, 0:3]
    feet_position_body = quat_rotate_inverse(
        base_quaternion_world.unsqueeze(0).repeat(len(FOOT_BODY_NAMES), 1),
        feet_position_world - base_position_world.unsqueeze(0),
    )
    return feet_position_world.clone(), feet_position_body.clone()


def print_report(urdf_path, dof_names, dof_positions, feet_world, feet_body):
    print(f"URDF: {urdf_path}")
    print("Isaac Gym DOF 顺序与默认角度 [rad]:")
    for name, position in zip(dof_names, dof_positions):
        print(f"  {name:20s} {position:+.6f}")

    print("\n足端坐标 [m]（顺序 FL, FR, RL, RR）:")
    for leg, world, body in zip(
        LEG_ORDER, feet_world.cpu().tolist(), feet_body.cpu().tolist()
    ):
        print(
            f"  {leg}: world=({world[0]:+.6f}, {world[1]:+.6f}, {world[2]:+.6f}), "
            f"body=({body[0]:+.6f}, {body[1]:+.6f}, {body[2]:+.6f})"
        )

    center = torch.mean(feet_body, dim=0)
    front_center = torch.mean(feet_body[:2], dim=0)
    rear_center = torch.mean(feet_body[2:], dim=0)
    stance_length = front_center[0] - rear_center[0]
    left_center = torch.mean(feet_body[[0, 2]], dim=0)
    right_center = torch.mean(feet_body[[1, 3]], dim=0)
    stance_width = left_center[1] - right_center[1]

    print("\n默认站姿足端几何统计 [m]:")
    print(f"  四足几何中心 = ({center[0]:+.6f}, {center[1]:+.6f}, {center[2]:+.6f})")
    print(f"  前足中心 x   = {front_center[0]:+.6f}")
    print(f"  后足中心 x   = {rear_center[0]:+.6f}")
    print(f"  前后跨度     = {stance_length:+.6f}")
    print(f"  左右跨度     = {stance_width:+.6f}")
    print("\n若零速度 Raibert 目标要匹配该默认姿态，建议至少表达：")
    print(f"  stance_center_x = {center[0]:+.6f}")
    print(f"  stance_center_y = {center[1]:+.6f}")
    print(f"  stance_length   = {stance_length:+.6f}")
    print(f"  stance_width    = {stance_width:+.6f}")


def main():
    args = parse_args()
    gym = gymapi.acquire_gym()
    sim = make_sim(gym, args.compute_device_id, args.graphics_device_id)
    try:
        asset, urdf_path = load_fixed_base_asset(gym, sim)
        env = gym.create_env(
            sim,
            gymapi.Vec3(-1.0, -1.0, 0.0),
            gymapi.Vec3(1.0, 1.0, 2.0),
            1,
        )
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
        actor = gym.create_actor(env, asset, pose, "minich_default_pose", 0, 0)
        if actor < 0:
            raise RuntimeError("无法创建固定基座 actor")

        gym.prepare_sim(sim)
        dof_names, dof_positions = set_default_joint_state(gym, env, actor, asset)
        feet_world, feet_body = measure_foot_positions(gym, sim, env, actor)
        print_report(urdf_path, dof_names, dof_positions, feet_world, feet_body)
    finally:
        gym.destroy_sim(sim)


if __name__ == "__main__":
    main()

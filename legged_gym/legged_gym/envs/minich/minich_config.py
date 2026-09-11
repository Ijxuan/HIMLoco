from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class MiniChRoughCfg(LeggedRobotCfg):
    """标准 12 自由度 Mini Cheetah rough-terrain 配置。"""

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        # 12 joint states are retained, while the two front/rear HAA pairs
        # share one action each: 10 policy actions drive 12 physical DOFs.
        num_one_step_observations = 43
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = num_one_step_observations + 3 + 3 + 187
        num_privileged_obs = num_one_step_privileged_obs
        num_actions = 10
        episode_length_s = 20

    class commands(LeggedRobotCfg.commands):
        in_place_turn_probability = 0.1

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.30]
        default_joint_angles = {
            "FL_hip_joint": 0.1,
            "RL_hip_joint": 0.1,
            "FR_hip_joint": -0.1,
            "RR_hip_joint": -0.1,
            "FL_thigh_joint": -0.8,
            "RL_thigh_joint": -0.8,
            "FR_thigh_joint": -0.8,
            "RR_thigh_joint": -0.8,
            "FL_calf_joint": 1.62,
            "RL_calf_joint": 1.62,
            "FR_calf_joint": 1.62,
            "RR_calf_joint": 1.62,
        }

    class control(LeggedRobotCfg.control):
        control_type = "P"
        stiffness = {"joint": 17.0, "calf_joint": 34.0}
        damping = {"joint": 0.4, "calf_joint": 0.8}
        action_scale = 0.25
        decimation = 4
        # Policy action order:
        # front HAA, FL HFE, FL KFE, FR HFE, FR KFE,
        # rear HAA, RL HFE, RL KFE, RR HFE, RR KFE.
        # Mini-Cheetah's right HAA axis is mirrored, hence the -1 signs.
        action_to_dof = [
            [(0, 1.0), (3, -1.0)],
            1, 2, 4, 5,
            [(6, 1.0), (9, -1.0)],
            7, 8, 10, 11,
        ]

    class asset(LeggedRobotCfg.asset):
        file = "{LEGGED_GYM_ROOT_DIR}/resources/robots/mini_cheetah/urdf/mini_cheetah.urdf"
        name = "minich"
        foot_name = "foot"
        penalize_contacts_on = []
        terminate_after_contacts_on = ["trunk"]
        collapse_fixed_joints = False
        self_collisions = 1
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = -30.0
            foot_clearance = 0
            action_rate = -0.01
            smoothness = -0.01
            feet_air_time = 0.0
            collision = 0.0
            termination = 0.0
            torques = 0.0
            dof_vel = 0.0
            dof_pos_limits = 0.0
            dof_vel_limits = 0.0
            torque_limits = 0.0
            raibert_heuristic = -10.0

        base_height_target = 0.26
        clearance_height_target = 0.04
        # Isaac Gym 固定基座、默认关节角下测得的四足矩形参数。
        raibert_stance_center_x = -0.011010
        raibert_stance_center_y = 0.0
        raibert_stance_length = 0.38
        raibert_stance_width = 0.276335
        raibert_prediction_time = 0.25
        raibert_contact_force_threshold = 1.0
        raibert_vx_command_threshold = 0.05
        raibert_yaw_command_threshold = 0.05

    class domain_rand(LeggedRobotCfg.domain_rand):
        # Keep the reference project's robot-level randomization defaults.
        randomize_link_mass = False
        randomize_restitution = False


class MiniChRoughCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ""
        experiment_name = "rough_minich"

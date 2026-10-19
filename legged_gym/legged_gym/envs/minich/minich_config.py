from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class MiniChRoughCfg(LeggedRobotCfg):
    """标准 12 自由度 Mini Cheetah rough-terrain 配置。"""

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        num_one_step_observations = 45
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = 45 + 3 + 3 + 187
        num_privileged_obs = num_one_step_privileged_obs
        num_actions = 12
        episode_length_s = 20

    class terrain(LeggedRobotCfg.terrain):
        long_stairs_enabled = True
        long_stairs_fraction = 0.15  # 3 of the default 20 curriculum columns.
        long_stairs_start_level = 2  # Zero-based; rows 0 and 1 remain flat.
        long_stairs_height = 0.08
        long_stairs_depth = 1.20
        long_stairs_count = 2
        long_stairs_spawn_range = 0.25
        long_stairs_success_hold_s = 0.20

    class commands(LeggedRobotCfg.commands):
        in_place_turn_probability = 0.1
        in_place_turn_min_yaw_rate = 0.2
        in_place_turn_duration_s = 10.0
        enable_turn_to_stand = True
        turn_to_stand_probability = 0.1

        class ranges(LeggedRobotCfg.commands.ranges):
            ang_vel_yaw = [-1.0, 1.0]

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
        stiffness = {"joint": 17.0, "calf_joint": 24.0}
        damping = {"joint": 0.4, "calf_joint": 0.8}
        action_scale = 0.25
        decimation = 4

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
        log_phase_rewards = True
        class scales(LeggedRobotCfg.rewards.scales):
            tracking_lin_vel = 1.0
            tracking_ang_vel = 1.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -1.2  # Terrain-relative angular cost; retune after training.
            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = -30.0
            step_retraction = 0.05  # Total credit <= 0.05 per foot/latched edge.
            step_retraction_stall = -0.20
            step_clearance = -0.25  # Bounded per-foot deficit; initial training weight.
            foot_clearance = 0
            action_rate = -0.01
            smoothness = -0.01
            feet_air_time = 0.0
            turn_lift_frequency = 0.25  # Disabled in the Sep14 scale baseline.
            turn_air_time_overdue = -1.0
            collision = 0.0
            termination = 0.0
            torques = 0.0
            dof_vel = 0.0
            dof_pos_limits = 0.0
            dof_vel_limits = 0.0
            torque_limits = 0.0
            raibert_heuristic = -10.0
            hip_mirror = -1.0
            hip_pos_penalty = -1.0

        # Relative XY tracking during translation; keep the original near-zero branch.
        relative_tracking_lin_vel = True
        tracking_lin_vel_sigma = 0.25  # Independent of angular tracking_sigma.
        tracking_lin_vel_min_speed = 0.3  # m/s; bound low-speed sensitivity.
        tracking_lin_vel_max_speed = 1.0  # m/s; do not relax high-speed tracking.
        tracking_lin_vel_command_threshold = 0.1  # m/s; inclusive moving gate.

        # Privileged stair-edge reward; actor/export inputs stay unchanged.
        step_retraction_distance = 0.03
        step_retraction_trigger_distance = 0.05
        step_retraction_force = 5.0
        step_retraction_confirm_time = 0.04
        step_retraction_timeout = 0.40
        step_retraction_progress_epsilon = 0.0002
        step_clearance_lookahead = 0.15
        step_clearance_sample_spacing = 0.01
        step_clearance_margin = 0.03
        step_clearance_foot_radius = 0.02  # URDF foot collision sphere.
        step_clearance_min_rise = 0.035
        step_clearance_support_tolerance = 0.012
        step_clearance_release_distance = 0.03
        step_clearance_min_speed = 0.10

        terrain_relative_orientation = True
        orientation_window_half_length = 0.4  # 0.8 m fore/aft fitting window.
        orientation_window_half_width = 0.2   # 0.4 m lateral fitting window.
        orientation_deadband_deg = 5.0
        orientation_filter_tau = 0.15        # World-frame normal low-pass, seconds.
        orientation_max_tilt_deg = 35.0      # Reference cap, not a termination gate.

        base_height_target = 0.26
        hip_pos_deadband_deg = 3.0  # Relative to each hip's default standing angle.
        # Turn lifts: >=1 cm sole clearance, >=0.5 Hz per foot in a 4 s window.
        turn_lift_min_clearance = 0.01
        turn_lift_min_frequency = 0.5
        turn_lift_window_s = 4.0
        turn_lift_foot_radius = 0.02
        # Retain continuous overdue-flight cost beyond 0.30 s.
        # Target/min remain the fallback formula for tasks without turn_lift settings.
        feet_air_time_target = 0.25
        feet_air_time_min = 0.20
        feet_air_time_max = 0.30
        clearance_height_target = 0.04
        # Isaac Gym 固定基座、默认关节角下测得的四足矩形参数。
        raibert_stance_center_x = -0.011010
        raibert_stance_center_y = 0.0
        raibert_stance_length = 0.38
        raibert_stance_width = 0.276335
        raibert_prediction_time = 0.25
        raibert_min_air_steps = 2  # Reject one-frame dropouts (dt=0.02s).
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

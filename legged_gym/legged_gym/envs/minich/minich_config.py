from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO


class MiniChRoughCfg(LeggedRobotCfg):
    """标准 12 自由度 Mini Cheetah rough-terrain 配置。"""

    class env(LeggedRobotCfg.env):
        num_envs = 4096
        stair_gait_clock = True
        num_one_step_observations = 47
        num_observations = num_one_step_observations * 6
        num_one_step_privileged_obs = num_one_step_observations + 3 + 3 + 187
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
        curriculum = False  # Former high-speed group now learns low-speed gait.
        gait_training_fraction = 0.20
        gait_training_speed_range = [0.20, 0.50]  # Magnitude; equal forward/backward sampling.
        in_place_turn_probability = 0.1
        in_place_turn_min_yaw_rate = 0.2
        in_place_turn_duration_s = 10.0
        enable_turn_to_stand = True
        turn_to_stand_probability = 0.1
        stair_step_stand_probability = 0.30  # At resampling on tilted terrain, hold zero 2-5 s.

        class ranges(LeggedRobotCfg.commands.ranges):
            lin_vel_x = [-1.0, 1.0]  # 前后速度，m/s
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
            foot_vertical_force = -1e-4  # -5e-5 ～ -2e-4Initial N^-2 weight; substep mean, then control dt.
            base_height = -30.0
            step_retraction = 0.1  # Signed per-foot event budget, maximum at 1 cm.后摆
            step_retraction_stall = 0.0  # Unified signed event score replaces process penalties.
            step_clearance = 0  # Bounded per-foot deficit; initial training weight.
            swing_peak_height = 0.1  # Per foot/swing, only the gait-training group.抬腿
            stair_step = 0.0  # Legacy event scorer; periodic dense costs now teach the gait.
            stair_step_overdue = 0.0
            stair_gait_force = -0.10  # 建议起点 -0.10，待验证范围 [-0.16, -0.08]。
            stair_gait_height = -0.25  # 建议起点 -0.25，待验证范围 [-0.40, -0.20]。
            stair_gait_velocity = -0.025  # 建议起点 -0.025，待验证范围 [-0.04, -0.02]。
            stair_gait_pair_deficit = -0.10  # 对角组取较差脚缺额；建议起点 -0.10，待验证范围 [-0.16, -0.08]。
            foot_clearance = 0
            action_rate = -0.01
            smoothness = -0.01
            feet_air_time = 0.0
            turn_lift_frequency = 0.25  # Disabled in the Sep14 scale baseline.
            turn_air_time_overdue = -1.0
            collision = 0.0
            termination = 0.0
            torques = -2e-4
            dof_vel = 0.0
            dof_pos_limits = 0.0
            dof_vel_limits = 0.0
            torque_limits = 0.0
            raibert_heuristic = -10.0
            hip_mirror = -1.0
            hip_pos_penalty = -1.0

        # Existing dense costs average four feet; pair deficit sums two groups.
        # All costs are integrated with dt by the reward framework.
        # Clock/costs run on tilted terrain at ALL commands, including zero/yaw.
        # Flat terrain disables this clock; normal velocity tracking remains.
        stair_gait_frequency = 1.5  # Hz; independent of commanded translation.
        stair_gait_duty_factor = 0.60  # Allows brief four-foot support at exchanges.
        stair_gait_height = 0.035  # Local sole clearance, not world Z.
        stair_gait_force_sigma = 25.0  # N, total foot contact-force norm.
        stair_gait_velocity_sigma = 0.15  # m/s, world foot-speed norm.
        stair_step_on_deg = 5.0
        stair_step_off_deg = 3.0
        # Legacy zero-command event diagnostics only; NOT stair_gait gates.
        stair_step_linear_threshold = 0.1
        stair_step_yaw_threshold = 0.1
        stair_step_min_height = 0.02
        stair_step_target_height = 0.035
        stair_step_max_height = 0.06
        stair_step_sync_time = 0.08
        stair_step_confirm_time = 0.04
        stair_step_min_air_time = 0.20
        stair_step_max_air_time = 0.30
        stair_step_overdue_time = 0.70

        # Relative XY tracking during translation; keep the original near-zero branch.
        relative_tracking_lin_vel = True
        tracking_lin_vel_sigma = 0.25  # Independent of angular tracking_sigma.
        tracking_lin_vel_min_speed = 0.3  # m/s; bound low-speed sensitivity.
        tracking_lin_vel_max_speed = 1.0  # m/s; do not relax high-speed tracking.
        tracking_lin_vel_command_threshold = 0.1  # m/s; inclusive moving gate.

        # Training-only swing/edge rewards; actor/export inputs stay unchanged.
        # Straight swings: 0.5--1.5 cm; subtract cumulative reverse body travel.
        step_retraction_distance = 0.005  # Minimum effective retraction.
        step_retraction_target_distance = 0.010
        step_retraction_max_distance = 0.015  # Maximum WORLD retraction.
        step_retraction_reference_time = 0.30
        step_retraction_timeout = 0.40
        step_retraction_min_forward_speed = 0.10
        step_retraction_max_yaw_rate = 0.05
        step_retraction_stance_confirm_time = 0.06
        step_retraction_swing_confirm_time = 0.04
        step_retraction_landing_confirm_time = 0.04
        step_retraction_contact_force = 1.0
        step_retraction_landing_penalty_time = 0.30
        step_clearance_lookahead = 0.15
        step_clearance_sample_spacing = 0.01
        step_clearance_margin = 0.03
        step_clearance_foot_radius = 0.02  # URDF foot collision sphere.
        step_clearance_min_rise = 0.035
        step_clearance_support_tolerance = 0.012
        step_clearance_release_distance = 0.03
        step_clearance_min_speed = 0.10

        swing_peak_height_min = 0.025
        swing_peak_height_target = 0.05
        swing_peak_height_max = 0.08
        swing_peak_height_min_air_time = 0.08
        swing_peak_height_timeout = 0.50
        swing_peak_height_stance_time = 0.06
        swing_peak_height_landing_time = 0.04

        terrain_relative_orientation = True
        # World-Z foot force, sampled each 5 ms while the policy remains at 20 ms.
        # Positive target uphill angle: <=5 deg flat, >=10 deg stairs, blend between.
        foot_vertical_force_flat_threshold = 130.0  # N per foot
        foot_vertical_force_stair_threshold = 160.0  # N per foot
        foot_vertical_force_flat_angle_deg = 5.0
        foot_vertical_force_stair_angle_deg = 10.0
        foot_vertical_force_min_forward_speed = 0.1  # m/s; stand/backward/sideways excluded
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

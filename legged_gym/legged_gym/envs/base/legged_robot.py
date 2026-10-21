# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, envs
from time import time
from warnings import WarningMessage
import numpy as np
import os
import math

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from torch import Tensor
from typing import Tuple, Dict

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs.base.base_task import BaseTask
from legged_gym.utils.terrain import Terrain
from legged_gym.utils.step_clearance import StepClearance
from legged_gym.utils.swing_peak_height import SwingPeakHeight
from legged_gym.utils.step_retraction import StepRetraction
from legged_gym.utils.math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float
from legged_gym.utils.helpers import class_to_dict
from .legged_robot_config import LeggedRobotCfg

class LeggedRobot(BaseTask):
    COMMAND_NORMAL = 0
    COMMAND_TURN = 1
    COMMAND_STAND = 2
    COMMAND_TURN_TO_STAND = 3

    def __init__(self, cfg: LeggedRobotCfg, sim_params, physics_engine, sim_device, headless):
        """ Parses the provided config file,
            calls create_sim() (which creates, simulation, terrain and environments),
            initilizes pytorch buffers used during training

        Args:
            cfg (Dict): Environment config file
            sim_params (gymapi.SimParams): simulation parameters
            physics_engine (gymapi.SimType): gymapi.SIM_PHYSX (must be PhysX)
            device_type (string): 'cuda' or 'cpu'
            device_id (int): 0, 1, ...
            headless (bool): Run without rendering if True
        """
        self.cfg = cfg
        self.sim_params = sim_params
        self.height_samples = None
        self.debug_viz = False
        self.init_done = False
        self._parse_cfg(self.cfg)
        super().__init__(self.cfg, sim_params, physics_engine, sim_device, headless)
        self.num_one_step_obs = self.cfg.env.num_one_step_observations
        self.num_one_step_privileged_obs = self.cfg.env.num_one_step_privileged_obs
        self.history_length = int(self.num_obs / self.num_one_step_obs)

        if not self.headless:
            self.set_camera(self.cfg.viewer.pos, self.cfg.viewer.lookat)
        self._init_buffers()
        self._prepare_reward_function()
        self.init_done = True

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)

        self.delayed_actions = self.actions.clone().view(self.num_envs, 1, self.num_actions).repeat(1, self.cfg.control.decimation, 1)
        delay_steps = torch.randint(0, self.cfg.control.decimation, (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.delay:
            for i in range(self.cfg.control.decimation):
                self.delayed_actions[:, i] = self.last_actions + (self.actions - self.last_actions) * (i >= delay_steps)
        # step physics and render each frame
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.delayed_actions[:, _]).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
        termination_ids, termination_priveleged_obs = self.post_physics_step()

        # return clipped obs, clipped states (None), rewards, dones and infos
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras, termination_ids, termination_priveleged_obs

    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations 
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self._post_physics_step_callback()
        if getattr(self.cfg.rewards, "terrain_relative_orientation", False):
            self._update_orientation_reference()

        if hasattr(self, "step_clearance") or hasattr(self, "step_retraction"):
            command = torch.cat((self.commands[:, :2], torch.zeros_like(self.commands[:, :1])), dim=1)
            command_world = quat_apply_yaw(self.base_quat, command)[:, :2]
        if hasattr(self, "step_clearance"):
            self.step_clearance.update(self.feet_pos, self.contact_forces[:, self.feet_indices],
                                       command_world, self._sample_step_heights)
        if hasattr(self, "step_retraction") or hasattr(self, "swing_peak_height"):
            sole = self.feet_pos[..., 2] - self.cfg.rewards.step_clearance_foot_radius
            ground = self._sample_step_heights(self.feet_pos[..., :2])
            support = ((sole - ground).abs() <= self.cfg.rewards.step_clearance_support_tolerance) & (
                self.contact_forces[:, self.feet_indices, 2] > self.cfg.rewards.step_retraction_contact_force)
        if hasattr(self, "swing_peak_height"):
            enabled = (self._gait_training_mask()
                       & (self.commands[:, 0].abs() >= self.cfg.rewards.step_retraction_min_forward_speed)
                       & (self.commands[:, 2].abs() < self.cfg.rewards.step_retraction_max_yaw_rate))
            self.swing_peak_height.update(sole - ground, support, self.commands, enabled)
        if hasattr(self, "step_retraction"):
            self.step_retraction.update(self.feet_pos, self.commands, command_world, support,
                                        self.root_states[:, :2])
            if hasattr(self, "step_clearance"):
                self.step_clearance.per_foot_cost.masked_fill_(self.step_retraction.stage == StepRetraction.RETRACT, 0.)
                self.step_clearance.cost[:] = self.step_clearance.per_foot_cost.sum(-1)

        if hasattr(self, "long_stair_success"):
            self._update_long_stair_success()

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        termination_privileged_obs = self.compute_termination_observations(env_ids)
        self.reset_idx(env_ids)
        self.compute_observations() # in some cases a simulation step might be required to refresh some obs (for example body positions)


        self.disturbance[:, :, :] = 0.0
        self.last_last_actions[:] = self.last_actions[:]
        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

        return env_ids, termination_privileged_obs

    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1., dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        long_stair_rate = None
        if hasattr(self, "long_stair_success"):
            selected = env_ids[self.long_stair_mask[env_ids] & (self.terrain_levels[env_ids] >= self.cfg.terrain.long_stairs_start_level)]
            if len(selected):
                long_stair_rate = self.long_stair_success[selected].float().mean()
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self.update_command_curriculum(env_ids)
        
        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self._resample_commands(env_ids)

        # reset buffers
        self.base_quat[env_ids] = self.root_states[env_ids, 3:7]
        self.base_lin_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 7:10])
        self.base_ang_vel[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.root_states[env_ids, 10:13])
        self.projected_gravity[env_ids] = quat_rotate_inverse(self.base_quat[env_ids], self.gravity_vec[env_ids])
        if hasattr(self, "orientation_reference_initialized"):
            self.orientation_reference_initialized[env_ids] = False
            self.orientation_target_normal[env_ids] = -self.gravity_vec[env_ids]
        if hasattr(self, "step_clearance"):
            self.step_clearance.reset(env_ids)
        if hasattr(self, "step_retraction"):
            self.step_retraction.reset(env_ids)
        if hasattr(self, "swing_peak_height"):
            self.swing_peak_height.reset(env_ids)
        if hasattr(self, "long_stair_success"):
            self.long_stair_success[env_ids] = False
            self.long_stair_hold[env_ids] = 0.
        self.actions[env_ids] = 0.
        self.disturbance[env_ids] = 0.
        self.obs_history_reset_pending[env_ids] = True
        self.last_actions[env_ids] = 0.
        self.last_last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.feet_air_time_penalty_active[env_ids] = False
        if hasattr(self, "turn_lift_ages"):
            self._reset_turn_lift_buffers(env_ids)
            self.turn_air_steps[env_ids] = 0
            self.turn_lift_frequency_reward[env_ids] = 0.
            self.turn_air_overdue_cost[env_ids] = 0.
            self.last_contacts[env_ids] = False
        self.raibert_air_steps[env_ids] = 0
        self.raibert_last_contacts[env_ids] = False
        self.raibert_contact_initialized[env_ids] = False
        self.reset_buf[env_ids] = 1

        # update height measurements
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        
         #reset randomized prop
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors[env_ids] = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (len(env_ids), 1), device=self.device)
        self.refresh_actor_rigid_shape_props(env_ids)
        
        # fill extras
        self.extras["episode"] = {}
        if long_stair_rate is not None:
            self.extras["episode"]["long_stairs_success"] = long_stair_rate
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids] / torch.clip(self.episode_length_buf[env_ids], min=1) / self.dt)
            self.episode_sums[key][env_ids] = 0.
        gait = self._gait_training_mask() if hasattr(self, 'swing_peak_height') or hasattr(self, 'step_retraction') else None
        for attr, prefix in [('swing_peak_height', 'gait_height'), ('step_retraction', 'gait_retraction')]:
            if not hasattr(self, attr):
                continue
            stats = getattr(self, attr).stats
            groups = [(prefix, env_ids[gait[env_ids]])]
            if attr == 'step_retraction':
                groups.append(('retraction_other', env_ids[~gait[env_ids]]))
            for group, ids in groups:
                for foot, name in enumerate(self.feet_names):
                    totals = {key: value[ids, foot].sum() for key, value in stats.items()}
                    if attr == 'swing_peak_height':
                        scale = self.cfg.rewards.scales.swing_peak_height
                        totals['weighted_score_sum'] = totals['score_sum'] * scale
                        totals['positive_reward_sum'] = totals['positive_sum'] * scale
                        totals['negative_reward_sum'] = totals['negative_sum'] * scale
                    else:
                        scale = self.cfg.rewards.scales.step_retraction_stall
                        for component in ('lag', 'over', 'landing'):
                            totals[component + '_reward_sum'] = totals[component + '_integral'] * scale
                        totals['weighted_score_sum'] = (totals['lag_reward_sum'] + totals['over_reward_sum']
                            + totals['landing_reward_sum'] + totals['score_sum'] * self.cfg.rewards.scales.step_retraction)
                        totals['positive_reward_sum'] = totals['positive_sum'] * self.cfg.rewards.scales.step_retraction
                        totals['negative_reward_sum'] = totals['negative_sum'] * self.cfg.rewards.scales.step_retraction
                    for metric, value in totals.items():
                        self.extras['episode'][f'{group}/{name}/{metric}'] = value
            for values in stats.values():
                values[env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
            for name, mask in self.stair_terrain_masks.items():
                self.extras["episode"]["terrain_level_" + name] = self.terrain_levels[mask].float().mean()
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        self.episode_length_buf[env_ids] = 0
    
    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        if hasattr(self, "turn_lift_ages"):
            self._update_turn_lift_state()
        self.rew_buf[:] = 0.
        phase_logger = getattr(self, "phase_reward_logger", None)
        phase_rewards = {} if phase_logger is not None else None
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
            if phase_rewards is not None:
                phase_rewards[name] = rew
        if phase_logger is not None:
            clipped = ((self.rew_buf < 0.) if self.cfg.rewards.only_positive_rewards
                       else torch.zeros_like(self.rew_buf, dtype=torch.bool))
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew
            if phase_rewards is not None:
                phase_rewards["termination"] = rew
        if phase_logger is not None:
            phase_logger.update(self.commands, phase_rewards, self.rew_buf,
                                clipped.to(self.rew_buf.dtype))

    def pop_phase_reward_logs(self):
        logger = getattr(self, "phase_reward_logger", None)
        result = logger.pop() if logger is not None else {}
        lift_logger = getattr(self, "turn_lift_logger", None)
        if lift_logger is not None:
            result.update(lift_logger.pop())
        return result
    
    def compute_observations(self):
        """ Computes observations
        """
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        self.obs_buf = torch.cat((current_obs[:, :self.num_one_step_obs], self.obs_buf[:, :-self.num_one_step_obs]), dim=-1)
        self.privileged_obs_buf = torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)

        # Repeat the same (possibly noisy) first frame, never the previous episode.
        reset_ids = self.obs_history_reset_pending.nonzero(as_tuple=False).flatten()
        self.obs_buf[reset_ids] = current_obs[reset_ids, :self.num_one_step_obs].repeat(1, self.history_length)
        privileged_history_length = self.num_privileged_obs // self.num_one_step_privileged_obs
        self.privileged_obs_buf[reset_ids] = current_obs[reset_ids, :self.num_one_step_privileged_obs].repeat(1, privileged_history_length)
        self.obs_history_reset_pending[reset_ids] = False

    def get_current_obs(self):
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        return current_obs
        
    def compute_termination_observations(self, env_ids):
        """ Computes observations
        """
        current_obs = torch.cat((   self.commands[:, :3] * self.commands_scale,
                                    self.base_ang_vel  * self.obs_scales.ang_vel,
                                    self.projected_gravity,
                                    (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                                    self.dof_vel * self.obs_scales.dof_vel,
                                    self.actions
                                    ),dim=-1)
        # add noise if needed
        if self.add_noise:
            current_obs += (2 * torch.rand_like(current_obs) - 1) * self.noise_scale_vec[0:(9 + 3 * self.num_actions)]

        # add perceptive inputs if not blind
        current_obs = torch.cat((current_obs, self.base_lin_vel * self.obs_scales.lin_vel, self.disturbance[:, 0, :]), dim=-1)
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements 
            heights += (2 * torch.rand_like(heights) - 1) * self.noise_scale_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions+187)]
            current_obs = torch.cat((current_obs, heights), dim=-1)

        return torch.cat((current_obs[:, :self.num_one_step_privileged_obs], self.privileged_obs_buf[:, :-self.num_one_step_privileged_obs]), dim=-1)[env_ids]
        
            
    def create_sim(self):
        """ Creates simulation, terrain and evironments
        """
        self.up_axis_idx = 2 # 2 for z, 1 for y -> adapt gravity accordingly
        self.sim = self.gym.create_sim(self.sim_device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        mesh_type = self.cfg.terrain.mesh_type
        if mesh_type in ['heightfield', 'trimesh']:
            self.terrain = Terrain(self.cfg.terrain, self.num_envs)
        if mesh_type=='plane':
            self._create_ground_plane()
        elif mesh_type=='heightfield':
            self._create_heightfield()
        elif mesh_type=='trimesh':
            self._create_trimesh()
        elif mesh_type is not None:
            raise ValueError("Terrain mesh type not recognised. Allowed types are [None, plane, heightfield, trimesh]")
        self._create_envs()

    def set_camera(self, position, lookat):
        """ Set camera position and direction
        """
        cam_pos = gymapi.Vec3(position[0], position[1], position[2])
        cam_target = gymapi.Vec3(lookat[0], lookat[1], lookat[2])
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    #------------- Callbacks --------------
    def _process_rigid_shape_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the rigid shape properties of each environment.
            Called During environment creation.
            Base behavior: randomizes the friction of each environment

        Args:
            props (List[gymapi.RigidShapeProperties]): Properties of each shape of the asset
            env_id (int): Environment id

        Returns:
            [List[gymapi.RigidShapeProperties]]: Modified rigid shape properties
        """
        if self.cfg.domain_rand.randomize_friction:
            if env_id==0:
                # prepare friction randomization
                friction_range = self.cfg.domain_rand.friction_range
                self.friction_coeffs = torch_rand_float(friction_range[0], friction_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].friction = self.friction_coeffs[env_id]

        if self.cfg.domain_rand.randomize_restitution:
            if env_id==0:
                # prepare restitution randomization
                restitution_range = self.cfg.domain_rand.restitution_range
                self.restitution_coeffs = torch_rand_float(restitution_range[0], restitution_range[1], (self.num_envs,1), device=self.device)

            for s in range(len(props)):
                props[s].restitution = self.restitution_coeffs[env_id]

        return props
    
    def refresh_actor_rigid_shape_props(self, env_ids):
        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.friction_range[0], self.cfg.domain_rand.friction_range[1], (len(env_ids), 1), device=self.device)
        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs[env_ids] = torch_rand_float(self.cfg.domain_rand.restitution_range[0], self.cfg.domain_rand.restitution_range[1], (len(env_ids), 1), device=self.device)
        
        for env_id in env_ids:
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[env_id], 0)

            for i in range(len(rigid_shape_props)):
                rigid_shape_props[i].friction = self.friction_coeffs[env_id, 0]
                rigid_shape_props[i].restitution = self.restitution_coeffs[env_id, 0]

            self.gym.set_actor_rigid_shape_properties(self.envs[env_id], 0, rigid_shape_props)

    def _process_dof_props(self, props, env_id):
        """ Callback allowing to store/change/randomize the DOF properties of each environment.
            Called During environment creation.
            Base behavior: stores position, velocity and torques limits defined in the URDF

        Args:
            props (numpy.array): Properties of each DOF of the asset
            env_id (int): Environment id

        Returns:
            [numpy.array]: Modified DOF properties
        """
        if env_id==0:
            self.dof_pos_limits = torch.zeros(self.num_dof, 2, dtype=torch.float, device=self.device, requires_grad=False)
            self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
            for i in range(len(props)):
                self.dof_pos_limits[i, 0] = props["lower"][i].item()
                self.dof_pos_limits[i, 1] = props["upper"][i].item()
                self.dof_vel_limits[i] = props["velocity"][i].item()
                self.torque_limits[i] = props["effort"][i].item()
                # soft limits
                m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
                r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
                self.dof_pos_limits[i, 0] = m - 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
                self.dof_pos_limits[i, 1] = m + 0.5 * r * self.cfg.rewards.soft_dof_pos_limit
        return props

    def _process_rigid_body_props(self, props, env_id):
        # if env_id==0:
        #     sum = 0
        #     for i, p in enumerate(props):
        #         sum += p.mass
        #         print(f"Mass of body {i}: {p.mass} (before randomization)")
        #     print(f"Total mass {sum} (before randomization)")
        # randomize base mass
        if self.cfg.domain_rand.randomize_payload_mass:
            props[0].mass = self.default_rigid_body_mass[0] + self.payload[env_id, 0]
            
        if self.cfg.domain_rand.randomize_com_displacement:
            props[0].com = gymapi.Vec3(self.com_displacement[env_id, 0], self.com_displacement[env_id, 1], self.com_displacement[env_id, 2])

        if self.cfg.domain_rand.randomize_link_mass:
            rng = self.cfg.domain_rand.link_mass_range
            for i in range(1, len(props)):
                scale = np.random.uniform(rng[0], rng[1])
                props[i].mass = scale * self.default_rigid_body_mass[i]

        return props
    
    def _post_physics_step_callback(self):
        """ Callback called before computing terminations, rewards, and observations
            Default behaviour: Compute ang vel command based on target and heading, compute measured terrain heights and randomly push robots
        """
        # 
        if getattr(self.cfg.commands, "enable_turn_to_stand", False):
            self._update_command_stages()
        else:
            env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt)==0).nonzero(as_tuple=False).flatten()
            self._resample_commands(env_ids)
        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            heading_yaw_commands = torch.clip(0.5*wrap_to_pi(self.commands[:, 3] - heading), -2., 2.)
            if getattr(self.cfg.commands, "enable_turn_to_stand", False):
                normal = self.command_stage == self.COMMAND_NORMAL
                self.commands[:, 2] = torch.where(normal, heading_yaw_commands, self.commands[:, 2])
            else:
                self.commands[:, 2] = torch.where(self.in_place_turn_buf, self.commands[:, 2], heading_yaw_commands)

        # The low-speed gait group never receives lateral translation targets.
        self.commands[self._gait_training_mask(), 1] = 0.
        if self.cfg.commands.heading_command:
            self.commands[self._gait_training_mask(), 2] = 0.
        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
        if self.cfg.domain_rand.push_robots and  (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()
        if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()
        if getattr(self.cfg.commands, "enable_turn_to_stand", False):
            self.commands[self.command_stage == self.COMMAND_STAND, :3] = 0.

    def _command_duration_steps(self, duration_range):
        """Validate a duration interval and return inclusive policy-step bounds."""
        if len(duration_range) != 2:
            raise ValueError("Command duration range must contain two values")
        lower, upper = duration_range
        if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower <= upper):
            raise ValueError("Command duration range must be finite, positive and ordered")
        # Sim dt may originate from a float32. Snap near-integer boundaries
        # before ceil/floor, including fixed-duration intervals such as 10 s.
        def step_boundary(seconds):
            value = seconds / self.dt
            nearest = round(value)
            return nearest if math.isclose(value, nearest, rel_tol=1e-7, abs_tol=1e-7) else value
        bounds = math.ceil(step_boundary(lower)), math.floor(step_boundary(upper))
        if bounds[0] > bounds[1]:
            raise ValueError("Command duration range contains no positive policy step")
        return bounds

    def _sample_command_duration(self, count, bounds):
        return torch.randint(bounds[0], bounds[1] + 1, (count,), device=self.device)

    def _update_command_stages(self):
        """Advance independent command clocks without resetting physical/history state."""
        self.command_steps_remaining -= 1
        expired = (self.command_steps_remaining <= 0).nonzero(as_tuple=False).flatten()
        # The scenario is selected at command sampling, not at turn expiry.
        enter_stand = self.command_stage[expired] == self.COMMAND_TURN_TO_STAND
        stand_ids = expired[enter_stand]
        self._resample_commands(expired[~enter_stand])
        self.command_stage[stand_ids] = self.COMMAND_STAND
        self.command_steps_remaining[stand_ids] = self._sample_command_duration(
            len(stand_ids), self.stand_duration_steps)
        self.in_place_turn_buf[stand_ids] = False
        self.commands[stand_ids, :3] = 0.

    def _gait_training_mask(self):
        fraction = getattr(self.cfg.commands, "gait_training_fraction", 0.)
        return torch.arange(self.num_envs, device=self.device) < self.num_envs * fraction

    def _resample_commands(self, env_ids):
        """ Randommly select commands of some environments

        Args:
            env_ids (List[int]): Environments ids for which new commands are needed
        """
        if len(env_ids) == 0:
            return
        self.commands[env_ids, 0] = torch_rand_float(-1.0, 1.0, (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        draw = torch.rand(len(env_ids), device=self.device)
        long_turn = draw < self.cfg.commands.in_place_turn_probability
        if getattr(self.cfg.commands, "enable_turn_to_stand", False):
            short_turn = (
                (draw >= self.cfg.commands.in_place_turn_probability)
                & (draw < self.cfg.commands.in_place_turn_probability + self.cfg.commands.turn_to_stand_probability)
            )
        else:
            short_turn = torch.zeros_like(long_turn)
        gait = self._gait_training_mask()[env_ids]
        long_turn &= ~gait
        short_turn &= ~gait
        self.in_place_turn_buf[env_ids] = long_turn | short_turn
        in_place_turn_env_ids = env_ids[self.in_place_turn_buf[env_ids]]
        self.commands[in_place_turn_env_ids, :2] = 0.
        self.commands[in_place_turn_env_ids, 2] = self._sample_in_place_turn_yaw(
            len(in_place_turn_env_ids))

        if getattr(self.cfg.commands, "gait_training_fraction", 0.) > 0:
            gait_ids = env_ids[gait]
            lo, hi = self.cfg.commands.gait_training_speed_range
            speed = torch_rand_float(lo, hi, (len(gait_ids), 1), device=self.device).squeeze(1)
            sign = torch.where(torch.rand(len(gait_ids), device=self.device) < .5, -1., 1.)
            # Explicit zero-range playback/test configurations remain zero-command.
            if self.command_ranges["lin_vel_x"] == [0., 0.]:
                speed.zero_()
            self.commands[gait_ids, 0] = speed * sign
            self.commands[gait_ids, 1:3] = 0.
        else:
            high_vel_env_ids = env_ids[(env_ids < (self.num_envs * 0.2)) & ~self.in_place_turn_buf[env_ids]]
            self.commands[high_vel_env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(high_vel_env_ids), 1), device=self.device).squeeze(1)
            self.commands[high_vel_env_ids, 1:2] *= (torch.norm(self.commands[high_vel_env_ids, 0:1], dim=1) < 1.0).unsqueeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= ((torch.norm(self.commands[env_ids, :2], dim=1) > 0.2) | gait).unsqueeze(1)
        if getattr(self.cfg.commands, "enable_turn_to_stand", False):
            self.command_stage[env_ids] = torch.where(
                self.in_place_turn_buf[env_ids], self.COMMAND_TURN, self.COMMAND_NORMAL)
            self.command_stage[env_ids[short_turn]] = self.COMMAND_TURN_TO_STAND
            self.command_steps_remaining[env_ids] = int(self.cfg.commands.resampling_time / self.dt)
            self.command_steps_remaining[env_ids[long_turn]] = self.long_turn_duration_steps
            short_turn_ids = env_ids[short_turn]
            self.command_steps_remaining[short_turn_ids] = self._sample_command_duration(
                len(short_turn_ids), self.turn_duration_steps)

    def _sample_in_place_turn_yaw(self, count):
        """Uniformly sample the yaw range excluding the minimum-magnitude gap."""
        low, high = self.command_ranges["ang_vel_yaw"]
        minimum = getattr(self.cfg.commands, "in_place_turn_min_yaw_rate", 0.0)
        if not math.isfinite(minimum) or minimum < 0:
            raise ValueError("in_place_turn_min_yaw_rate must be finite and nonnegative")
        # Keep legacy sampling and explicitly zeroed diagnostic ranges supported.
        if minimum == 0 or (low == 0 and high == 0):
            return torch_rand_float(low, high, (count, 1), device=self.device).squeeze(1)
        negative_high = min(high, -minimum)
        positive_low = max(low, minimum)
        negative_width = max(0.0, negative_high - low)
        positive_width = max(0.0, high - positive_low)
        if negative_width + positive_width <= 0:
            raise ValueError("ang_vel_yaw range must contain a nonempty interval outside the turn gap")
        sample = torch_rand_float(0.0, negative_width + positive_width,
                                  (count, 1), device=self.device).squeeze(1)
        return torch.where(
            sample < negative_width,
            torch.clamp(low + sample, max=negative_high),
            torch.clamp(positive_low + sample - negative_width, min=positive_low, max=high))

    def _compute_torques(self, actions):
        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, [0, 3, 6, 9]] *=self.cfg.control.hip_reduction
        self.joint_pos_target = self.default_dof_pos + actions_scaled

        control_type = self.cfg.control.control_type
        if control_type=="P":
            torques = self.p_gains * self.Kp_factors * (self.joint_pos_target - self.dof_pos) - self.d_gains * self.Kd_factors * self.dof_vel
        elif control_type=="V":
            torques = self.p_gains*(actions_scaled - self.dof_vel) - self.d_gains*(self.dof_vel - self.last_dof_vel)/self.sim_params.dt
        elif control_type=="T":
            torques = actions_scaled
        else:
            raise NameError(f"Unknown controller type: {control_type}")
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _reset_dofs(self, env_ids):
        """ Resets DOF position and velocities of selected environmments
        Positions are randomly selected within 0.5:1.5 x default positions.
        Velocities are set to zero.

        Args:
            env_ids (List[int]): Environemnt ids
        """
        self.dof_pos[env_ids] = self.default_dof_pos * torch_rand_float(0.5, 1.5, (len(env_ids), self.num_dof), device=self.device)
        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if hasattr(self, "long_stair_mask"):
            ids = env_ids[self.long_stair_mask[env_ids]]
            if len(ids):
                self.root_states[ids, :2] = self.env_origins[ids, :2] + self.base_init_state[:2] + torch_rand_float(
                    -self.cfg.terrain.long_stairs_spawn_range, self.cfg.terrain.long_stairs_spawn_range,
                    (len(ids), 2), device=self.device)
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def _push_robots(self):
        """ Random pushes the robots. Emulates an impulse by setting a randomized base velocity. 
        """
        max_vel = self.cfg.domain_rand.max_push_vel_xy
        self.root_states[:, 7:9] = torch_rand_float(-max_vel, max_vel, (self.num_envs, 2), device=self.device) # lin vel x/y
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

    def _disturbance_robots(self):
        """ Random add disturbance force to the robots.
        """
        disturbance = torch_rand_float(self.cfg.domain_rand.disturbance_range[0], self.cfg.domain_rand.disturbance_range[1], (self.num_envs, 3), device=self.device)
        self.disturbance[:, 0, :] = disturbance
        self.gym.apply_rigid_body_force_tensors(self.sim, forceTensor=gymtorch.unwrap_tensor(self.disturbance), space=gymapi.CoordinateSpace.LOCAL_SPACE)

    def _update_long_stair_success(self):
        """Require all feet on one outer landing, upright and supported for 0.2 s."""
        cfg = self.cfg.terrain
        enabled = self.long_stair_mask & (self.terrain_levels >= cfg.long_stairs_start_level)
        local = self.feet_pos - self.env_origins[:, None, :]
        radius = self.cfg.rewards.step_clearance_foot_radius
        threshold = cfg.long_stairs_count * cfg.long_stairs_depth + radius + cfg.horizontal_scale
        same_landing = (local[..., 0] >= threshold).all(1) | (local[..., 0] <= -threshold).all(1)
        inside = (local[..., 0].abs() < cfg.terrain_length / 2 - radius).all(1) & (local[..., 1].abs() < cfg.terrain_width / 2 - radius).all(1)
        high = (local[..., 2] >= cfg.long_stairs_count * cfg.long_stairs_height + radius - .02).all(1)
        supported = (self.contact_forces[:, self.feet_indices, 2] > 1.).sum(1) >= 2
        valid = enabled & same_landing & inside & high & supported & (self.projected_gravity[:, 2] < -.7)
        self.long_stair_hold[:] = torch.where(valid, self.long_stair_hold + self.dt, torch.zeros_like(self.long_stair_hold))
        self.long_stair_success |= self.long_stair_hold >= cfg.long_stairs_success_hold_s

    def _update_terrain_curriculum(self, env_ids):
        """ Implements the game-inspired curriculum.

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # Implement Terrain curriculum
        if not self.init_done:
            # don't change on initial reset
            return
        distance = torch.norm(self.root_states[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        # robots that walked far enough progress to harder terains
        move_up = distance > self.terrain.env_length / 2
        # robots that walked less than half of their required distance go to simpler terrains
        move_down = (distance < torch.norm(self.commands[env_ids, :2], dim=1)*self.max_episode_length_s*0.5) * ~move_up

        # In-place turning neither advances nor demotes the terrain level after a timeout.
        # A non-timeout termination is treated as a fall and demotes the terrain level.
        in_place_turn = (
            (torch.norm(self.commands[env_ids, :2], dim=1) < 0.1)
            & (torch.abs(self.commands[env_ids, 2]) >= 0.2)
        )
        fell = ~self.time_out_buf[env_ids]
        move_up = torch.where(in_place_turn, torch.zeros_like(move_up), move_up)
        move_down = torch.where(in_place_turn, fell, move_down)

        if hasattr(self, "long_stair_success"):
            special = self.long_stair_mask[env_ids] & (self.terrain_levels[env_ids] >= self.cfg.terrain.long_stairs_start_level)
            success = self.long_stair_success[env_ids] & ~fell
            move_up = torch.where(special, success, move_up)
            # Failure to climb under a translating command demotes; standing and
            # turning timeouts hold the level, while falls always demote.
            failed = fell | ((torch.norm(self.commands[env_ids, :2], dim=1) >= .1) & ~success)
            move_down = torch.where(special, failed, move_down)
        self.terrain_levels[env_ids] += 1 * move_up - 1 * move_down
        # Robots that solve the last level are sent to a random one
        self.terrain_levels[env_ids] = torch.where(self.terrain_levels[env_ids]>=self.max_terrain_level,
                                                   torch.randint_like(self.terrain_levels[env_ids], self.max_terrain_level),
                                                   torch.clip(self.terrain_levels[env_ids], 0)) # (the minumum level is zero)
        self.env_origins[env_ids] = self.terrain_origins[self.terrain_levels[env_ids], self.terrain_types[env_ids]]
    
    def update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        if getattr(self.cfg.commands, "gait_training_fraction", 0.) > 0:
            return  # No high-speed group consumes the expanding command range.
        low_vel_env_ids = (env_ids > (self.num_envs * 0.2))
        high_vel_env_ids = (env_ids < (self.num_envs * 0.2))
        low_vel_env_ids = env_ids[low_vel_env_ids.nonzero(as_tuple=True)]
        high_vel_env_ids = env_ids[high_vel_env_ids.nonzero(as_tuple=True)]
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if (torch.mean(self.episode_sums["tracking_lin_vel"][low_vel_env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]) and (torch.mean(self.episode_sums["tracking_lin_vel"][high_vel_env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]):
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.2, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.2, 0., self.cfg.commands.max_curriculum)


    def _get_noise_scale_vec(self, cfg):
        """ Sets a vector used to scale the noise added to the observations.
            [NOTE]: Must be adapted when changing the observations structure

        Args:
            cfg (Dict): Environment config file

        Returns:
            [torch.Tensor]: Vector of scales used to multiply a uniform distribution in [-1, 1]
        """
        # noise_vec = torch.zeros_like(self.obs_buf[0])\
        if self.cfg.terrain.measure_heights:
            noise_vec = torch.zeros(9 + 3*self.num_actions + 187, device=self.device)
        else:
            noise_vec = torch.zeros(9 + 3*self.num_actions, device=self.device)
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[0:3] = 0. # commands
        noise_vec[3:6] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[6:9] = noise_scales.gravity * noise_level
        noise_vec[9:(9 + self.num_actions)] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[(9 + self.num_actions):(9 + 2 * self.num_actions)] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[(9 + 2 * self.num_actions):(9 + 3 * self.num_actions)] = 0. # previous actions
        if self.cfg.terrain.measure_heights:
            noise_vec[(9 + 3 * self.num_actions):(9 + 3 * self.num_actions + 187)] = noise_scales.height_measurements* noise_level * self.obs_scales.height_measurements
        #noise_vec[232:] = 0
        return noise_vec

    #----------------------------------------
    def _init_buffers(self):
        """ Initialize torch tensors which will contain simulation states and processed quantities
        """
        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.base_quat = self.root_states[:, 3:7]
        self.feet_pos = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 7:10]

        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(self.num_envs, -1, 3) # shape: num_envs, num_bodies, xyz axis

        # initialize some data used later on
        self.common_step_counter = 0
        self.obs_history_reset_pending = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.extras = {}
        self.noise_scale_vec = self._get_noise_scale_vec(self.cfg)
        self.gravity_vec = to_torch(get_axis_params(-1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.forward_vec = to_torch([1., 0., 0.], device=self.device).repeat((self.num_envs, 1))
        self.torques = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.p_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.d_gains = torch.zeros(self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_last_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device, requires_grad=False)
        self.last_dof_vel = torch.zeros_like(self.dof_vel)
        self.last_root_vel = torch.zeros_like(self.root_states[:, 7:13])
        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, dtype=torch.float, device=self.device, requires_grad=False) # x vel, y vel, yaw vel, heading
        self.in_place_turn_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        if getattr(self.cfg.commands, "enable_turn_to_stand", False):
            self.command_stage = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.command_steps_remaining = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.commands_scale = torch.tensor([self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel], device=self.device, requires_grad=False,) # TODO change this
        self.feet_air_time = torch.zeros(self.num_envs, self.feet_indices.shape[0], dtype=torch.float, device=self.device, requires_grad=False)
        self.feet_air_time_penalty_active = torch.zeros_like(self.feet_air_time, dtype=torch.bool)
        self.last_contacts = torch.zeros(self.num_envs, len(self.feet_indices), dtype=torch.bool, device=self.device, requires_grad=False)
        if getattr(self.cfg.rewards, "turn_lift_min_clearance", None) is not None:
            self._init_turn_lift_buffers()
        if (not isinstance(self.cfg.rewards.raibert_min_air_steps, int)
                or self.cfg.rewards.raibert_min_air_steps < 1):
            raise ValueError("raibert_min_air_steps must be a positive integer")
        self.raibert_air_steps = torch.zeros(self.num_envs, 4, dtype=torch.long, device=self.device)
        self.raibert_last_contacts = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device, requires_grad=False)
        self.raibert_contact_initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self.base_lin_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity = quat_rotate_inverse(self.base_quat, self.gravity_vec)
        if self.cfg.terrain.measure_heights:
            self.height_points = self._init_height_points()
        self.measured_heights = self._get_heights()
        if getattr(self.cfg.rewards, "terrain_relative_orientation", False):
            self._init_orientation_reference()
        if getattr(self.cfg.rewards.scales, "swing_peak_height", 0.) != 0.:
            self.swing_peak_height = SwingPeakHeight(self.num_envs, len(self.feet_indices), self.device, self.cfg.rewards, self.dt)
        retraction_enabled = any(getattr(self.cfg.rewards.scales, name, 0.) != 0. for name in ("step_retraction", "step_retraction_stall"))
        if getattr(self.cfg.rewards.scales, "step_clearance", 0.) != 0.:
            if self.cfg.terrain.mesh_type not in ("plane", "heightfield", "trimesh"):
                raise ValueError("step_clearance requires terrain heights")
            self.step_clearance = StepClearance(self.num_envs, len(self.feet_indices), self.device, self.cfg.rewards)
        if retraction_enabled:
            if self.cfg.terrain.mesh_type not in ("plane", "heightfield", "trimesh"):
                raise ValueError("step_retraction requires support-height sampling")
            self.step_retraction = StepRetraction(self.num_envs, len(self.feet_indices), self.device, self.cfg.rewards, self.dt)
        self.base_height_points = self._init_base_height_points()

        # joint positions offsets and PD gains
        self.default_dof_pos = torch.zeros(self.num_dof, dtype=torch.float, device=self.device, requires_grad=False)
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            angle = self.cfg.init_state.default_joint_angles[name]
            self.default_dof_pos[i] = angle
            found = False
            for dof_name in self.cfg.control.stiffness.keys():
                if dof_name in name:
                    self.p_gains[i] = self.cfg.control.stiffness[dof_name]
                    self.d_gains[i] = self.cfg.control.damping[dof_name]
                    found = True
            if not found:
                self.p_gains[i] = 0.
                self.d_gains[i] = 0.
                if self.cfg.control.control_type in ["P", "V"]:
                    print(f"PD gain of joint {name} were not defined, setting them to zero")
        self.default_dof_pos = self.default_dof_pos.unsqueeze(0)
        
        
        #randomize kp, kd, motor strength
        self.Kp_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.Kd_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.motor_strength_factors = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        self.disturbance = torch.zeros(self.num_envs, self.num_bodies, 3, dtype=torch.float, device=self.device, requires_grad=False)
        
        if self.cfg.domain_rand.randomize_kp:
            self.Kp_factors = torch_rand_float(self.cfg.domain_rand.kp_range[0], self.cfg.domain_rand.kp_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_kd:
            self.Kd_factors = torch_rand_float(self.cfg.domain_rand.kd_range[0], self.cfg.domain_rand.kd_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_motor_strength:
            self.motor_strength_factors = torch_rand_float(self.cfg.domain_rand.motor_strength_range[0], self.cfg.domain_rand.motor_strength_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        #store friction and restitution
        self.friction_coeffs = torch.ones(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.restitution_coeffs = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)


    def _prepare_reward_function(self):
        """ Prepares a list of reward functions, whcih will be called to compute the total reward.
            Looks for self._reward_<REWARD_NAME>, where <REWARD_NAME> are names of all non zero reward scales in the cfg.
        """
        if (hasattr(self, "turn_lift_ages")
                and self.reward_scales.get("feet_air_time", 0.) != 0.):
            raise ValueError("Use turn_lift_frequency/turn_air_time_overdue with turn-lift buffers; disable feet_air_time")
        # remove zero scales + multiply non-zero ones by dt
        for key in list(self.reward_scales.keys()):
            scale = self.reward_scales[key]
            if scale==0:
                self.reward_scales.pop(key) 
            else:
                self.reward_scales[key] *= self.dt
        # prepare list of functions
        self.reward_functions = []
        self.reward_names = []
        for name, scale in self.reward_scales.items():
            if name=="termination":
                continue
            self.reward_names.append(name)
            name = '_reward_' + name
            self.reward_functions.append(getattr(self, name))

        # reward episode sums
        self.episode_sums = {name: torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
                             for name in self.reward_scales.keys()}
        self.phase_reward_logger = None
        logging_cfg = getattr(getattr(self, "cfg", None), "rewards", None)
        if getattr(logging_cfg, "log_phase_rewards", False):
            from legged_gym.utils.phase_reward_logger import PhaseRewardLogger
            self.phase_reward_logger = PhaseRewardLogger(
                self.reward_scales.keys(), self.dt, self.device)

    def _create_ground_plane(self):
        """ Adds a ground plane to the simulation, sets friction and restitution based on the cfg.
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.cfg.terrain.static_friction
        plane_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        plane_params.restitution = self.cfg.terrain.restitution
        self.gym.add_ground(self.sim, plane_params)
    
    def _create_heightfield(self):
        """ Adds a heightfield terrain to the simulation, sets parameters based on the cfg.
        """
        hf_params = gymapi.HeightFieldParams()
        hf_params.column_scale = self.terrain.cfg.horizontal_scale
        hf_params.row_scale = self.terrain.cfg.horizontal_scale
        hf_params.vertical_scale = self.terrain.cfg.vertical_scale
        hf_params.nbRows = self.terrain.tot_cols
        hf_params.nbColumns = self.terrain.tot_rows 
        hf_params.transform.p.x = -self.terrain.cfg.border_size 
        hf_params.transform.p.y = -self.terrain.cfg.border_size
        hf_params.transform.p.z = 0.0
        hf_params.static_friction = self.cfg.terrain.static_friction
        hf_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        hf_params.restitution = self.cfg.terrain.restitution

        self.gym.add_heightfield(self.sim, self.terrain.heightsamples, hf_params)
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_trimesh(self):
        """ Adds a triangle mesh terrain to the simulation, sets parameters based on the cfg.
        # """
        tm_params = gymapi.TriangleMeshParams()
        tm_params.nb_vertices = self.terrain.vertices.shape[0]
        tm_params.nb_triangles = self.terrain.triangles.shape[0]

        tm_params.transform.p.x = -self.terrain.cfg.border_size 
        tm_params.transform.p.y = -self.terrain.cfg.border_size
        tm_params.transform.p.z = 0.0
        tm_params.static_friction = self.cfg.terrain.static_friction
        tm_params.dynamic_friction = self.cfg.terrain.dynamic_friction
        tm_params.restitution = self.cfg.terrain.restitution
        self.gym.add_triangle_mesh(self.sim, self.terrain.vertices.flatten(order='C'), self.terrain.triangles.flatten(order='C'), tm_params)   
        self.height_samples = torch.tensor(self.terrain.heightsamples).view(self.terrain.tot_rows, self.terrain.tot_cols).to(self.device)

    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])
            
        self.default_rigid_body_mass = torch.zeros(self.num_bodies, dtype=torch.float, device=self.device, requires_grad=False)

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        
        self.payload = torch.zeros(self.num_envs, 1, dtype=torch.float, device=self.device, requires_grad=False)
        self.com_displacement = torch.zeros(self.num_envs, 3, dtype=torch.float, device=self.device, requires_grad=False)
        if self.cfg.domain_rand.randomize_payload_mass:
            self.payload = torch_rand_float(self.cfg.domain_rand.payload_mass_range[0], self.cfg.domain_rand.payload_mass_range[1], (self.num_envs, 1), device=self.device)
        if self.cfg.domain_rand.randomize_com_displacement:
            self.com_displacement = torch_rand_float(self.cfg.domain_rand.com_displacement_range[0], self.cfg.domain_rand.com_displacement_range[1], (self.num_envs, 3), device=self.device)
            
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            start_pose.p = gymapi.Vec3(*pos)
                
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            
            if i == 0:
                for j in range(len(body_props)):
                    self.default_rigid_body_mass[j] = body_props[j].mass
                    
            body_props = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)

        self.feet_names = feet_names
        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        # Raibert uses an explicit [FL, FR, RL, RR] order. Existing rewards
        # continue using the original Isaac Gym feet_indices order.
        raibert_feet_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        raibert_indices = []
        raibert_order = []
        for foot_name in raibert_feet_names:
            matches = [name for name in feet_names if name == foot_name]
            if len(matches) != 1:
                raise ValueError(f"expected exactly one body named {foot_name}, found {matches}")
            raibert_indices.append(self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.actor_handles[0], foot_name))
            raibert_order.append(feet_names.index(foot_name))
        self.raibert_foot_indices = torch.tensor(raibert_indices, dtype=torch.long, device=self.device)
        self.raibert_foot_order = torch.tensor(raibert_order, dtype=torch.long, device=self.device)

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])
            

    def _get_env_origins(self):
        """ Sets environment origins. On rough terrain the origins are defined by the terrain platforms.
            Otherwise create a grid.
        """
        if self.cfg.terrain.mesh_type in ["heightfield", "trimesh"]:
            self.custom_origins = True
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # put robots at the origins defined by the terrain
            max_init_level = self.cfg.terrain.max_init_terrain_level
            if not self.cfg.terrain.curriculum: max_init_level = self.cfg.terrain.num_rows - 1
            self.terrain_levels = torch.randint(0, max_init_level+1, (self.num_envs,), device=self.device)
            self.terrain_types = torch.div(torch.arange(self.num_envs, device=self.device), (self.num_envs/self.cfg.terrain.num_cols), rounding_mode='floor').to(torch.long)
            self.max_terrain_level = self.cfg.terrain.num_rows
            self.terrain_origins = torch.from_numpy(self.terrain.env_origins).to(self.device).to(torch.float)
            self.env_origins[:] = self.terrain_origins[self.terrain_levels, self.terrain_types]
            self.stair_terrain_masks = {}
            if self.cfg.terrain.curriculum:
                # Match Terrain.curiculum/make_terrain, including its sign convention:
                # choices below proportions[2] generate negative-height (down) stairs.
                ordinary_cols = self.cfg.terrain.num_cols - getattr(self.terrain, "long_stair_columns", 0)
                choices = np.arange(self.cfg.terrain.num_cols) / ordinary_cols + 0.001
                long_columns = np.arange(self.cfg.terrain.num_cols) >= ordinary_cols
                if long_columns.any():
                    self.long_stair_mask = torch.as_tensor(long_columns, device=self.device)[self.terrain_types]
                    self.long_stair_success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                    self.long_stair_hold = torch.zeros(self.num_envs, device=self.device)
                proportions = self.terrain.proportions
                stair_columns = (choices >= proportions[1]) & (choices < proportions[3])
                for name, columns in (
                    ("stairs", stair_columns),
                    ("stairs_up", stair_columns & (choices >= proportions[2])),
                    ("stairs_down", stair_columns & (choices < proportions[2])),
                    ("long_stairs", long_columns),
                ):
                    mask = torch.as_tensor(columns, device=self.device)[self.terrain_types]
                    # Fixed masks across resets; omit absent types rather than log NaN.
                    if mask.any().item():
                        self.stair_terrain_masks[name] = mask
        else:
            self.custom_origins = False
            self.env_origins = torch.zeros(self.num_envs, 3, device=self.device, requires_grad=False)
            # create a grid of robots
            num_cols = np.floor(np.sqrt(self.num_envs))
            num_rows = np.ceil(self.num_envs / num_cols)
            xx, yy = torch.meshgrid(torch.arange(num_rows), torch.arange(num_cols))
            spacing = self.cfg.env.env_spacing
            self.env_origins[:, 0] = spacing * xx.flatten()[:self.num_envs]
            self.env_origins[:, 1] = spacing * yy.flatten()[:self.num_envs]
            self.env_origins[:, 2] = 0.

    def _parse_cfg(self, cfg):
        self.dt = self.cfg.control.decimation * self.sim_params.dt
        if getattr(cfg.commands, "enable_turn_to_stand", False):
            if not math.isfinite(self.dt) or self.dt <= 0:
                raise ValueError("Policy step duration must be finite and positive")
            probabilities = (cfg.commands.in_place_turn_probability, cfg.commands.turn_to_stand_probability)
            if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
                raise ValueError("Command scenario probability must be in [0, 1]")
            if sum(probabilities) > 1:
                raise ValueError("Command scenario probability sum must be <= 1")
            if not math.isfinite(cfg.commands.resampling_time) or int(cfg.commands.resampling_time / self.dt) < 1:
                raise ValueError("resampling_time must contain at least one policy step")
            self.turn_duration_steps = self._command_duration_steps(cfg.commands.turn_duration_range_s)
            self.stand_duration_steps = self._command_duration_steps(cfg.commands.stand_duration_range_s)
            long_duration = cfg.commands.in_place_turn_duration_s
            self.long_turn_duration_steps = self._command_duration_steps([long_duration, long_duration])[0]
        fraction = getattr(cfg.commands, "gait_training_fraction", 0.)
        if not math.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("Invalid gait_training_fraction")
        if fraction > 0:
            lo, hi = cfg.commands.gait_training_speed_range
            if not (math.isfinite(lo) and math.isfinite(hi) and .2 <= lo <= hi):
                raise ValueError("Gait speeds must be finite, ordered and >= 0.2 m/s")
        self.obs_scales = self.cfg.normalization.obs_scales
        self.reward_scales = class_to_dict(self.cfg.rewards.scales)
        self.command_ranges = class_to_dict(self.cfg.commands.ranges)
        if self.cfg.terrain.mesh_type not in ['heightfield', 'trimesh']:
            self.cfg.terrain.curriculum = False
        self.max_episode_length_s = self.cfg.env.episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.dt)

        self.cfg.domain_rand.push_interval = np.ceil(self.cfg.domain_rand.push_interval_s / self.dt)

    def _draw_debug_vis(self):
        """ Draws visualizations for dubugging (slows down simulation a lot).
            Default behaviour: draws height measurement points
        """
        # draw height lines
        if not self.terrain.cfg.measure_heights:
            return
        self.gym.clear_lines(self.viewer)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        sphere_geom = gymutil.WireframeSphereGeometry(0.02, 4, 4, None, color=(1, 1, 0))
        for i in range(self.num_envs):
            base_pos = (self.root_states[i, :3]).cpu().numpy()
            heights = self.measured_heights[i].cpu().numpy()
            height_points = quat_apply_yaw(self.base_quat[i].repeat(heights.shape[0]), self.height_points[i]).cpu().numpy()
            for j in range(heights.shape[0]):
                x = height_points[j, 0] + base_pos[0]
                y = height_points[j, 1] + base_pos[1]
                z = heights[j]
                sphere_pose = gymapi.Transform(gymapi.Vec3(x, y, z), r=None)
                gymutil.draw_lines(sphere_geom, self.gym, self.viewer, self.envs[i], sphere_pose) 

    def _init_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_height_points, 3)
        """
        y = torch.tensor(self.cfg.terrain.measured_points_y, device=self.device, requires_grad=False)
        x = torch.tensor(self.cfg.terrain.measured_points_x, device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points
    
    def _init_base_height_points(self):
        """ Returns points at which the height measurments are sampled (in base frame)

        Returns:
            [torch.Tensor]: Tensor of shape (num_envs, self.num_base_height_points, 3)
        """
        y = torch.tensor([-0.2, -0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15, 0.2], device=self.device, requires_grad=False)
        x = torch.tensor([-0.15, -0.1, -0.05, 0., 0.05, 0.1, 0.15], device=self.device, requires_grad=False)
        grid_x, grid_y = torch.meshgrid(x, y)

        self.num_base_height_points = grid_x.numel()
        points = torch.zeros(self.num_envs, self.num_base_height_points, 3, device=self.device, requires_grad=False)
        points[:, :, 0] = grid_x.flatten()
        points[:, :, 1] = grid_y.flatten()
        return points

    def _get_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return torch.zeros(self.num_envs, self.num_height_points, device=self.device, requires_grad=False)
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_height_points), self.height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_height_points), self.height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)

        return heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
    
    def _get_base_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.root_states[:, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = quat_apply_yaw(self.base_quat[env_ids].repeat(1, self.num_base_height_points), self.base_height_points[env_ids]) + (self.root_states[env_ids, :3]).unsqueeze(1)
        else:
            points = quat_apply_yaw(self.base_quat.repeat(1, self.num_base_height_points), self.base_height_points) + (self.root_states[:, :3]).unsqueeze(1)


        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        heights = torch.min(heights1, heights2)
        heights = torch.min(heights, heights3)
        # heights = (heights1 + heights2 + heights3) / 3

        base_height =  heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - base_height, dim=1)

        return base_height
    
    def _get_feet_heights(self, env_ids=None):
        """ Samples heights of the terrain at required points around each robot.
            The points are offset by the base's position and rotated by the base's yaw

        Args:
            env_ids (List[int], optional): Subset of environments for which to return the heights. Defaults to None.

        Raises:
            NameError: [description]

        Returns:
            [type]: [description]
        """
        if self.cfg.terrain.mesh_type == 'plane':
            return self.feet_pos[:, :, 2].clone()
        elif self.cfg.terrain.mesh_type == 'none':
            raise NameError("Can't measure height with terrain mesh type 'none'")

        if env_ids:
            points = self.feet_pos[env_ids].clone()
        else:
            points = self.feet_pos.clone()

        points += self.terrain.cfg.border_size
        points = (points/self.terrain.cfg.horizontal_scale).long()
        px = points[:, :, 0].view(-1)
        py = points[:, :, 1].view(-1)
        px = torch.clip(px, 0, self.height_samples.shape[0]-2)
        py = torch.clip(py, 0, self.height_samples.shape[1]-2)

        heights1 = self.height_samples[px, py]
        heights2 = self.height_samples[px+1, py]
        heights3 = self.height_samples[px, py+1]
        # heights = torch.min(heights1, heights2)
        # heights = torch.min(heights, heights3)
        heights = (heights1 + heights2 + heights3) / 3

        heights = heights.view(self.num_envs, -1) * self.terrain.cfg.vertical_scale

        feet_height =  self.feet_pos[:, :, 2] - heights

        return feet_height

    #------------ reward functions----------------
    def _reward_hip_mirror(self):
        """Pairwise hip mirror error in radians, gated by in-place turn commands."""
        in_place_turn = (
            (torch.norm(self.commands[:, :2], dim=1) < 0.1)
            & (torch.abs(self.commands[:, 2]) >= 0.2)
        )
        # DOF order: FL, FR, RL, RR; each leg has hip, thigh, calf.
        # Mirrored ab/adduction has opposite signed joint angles.
        front_error = torch.abs(self.dof_pos[:, 0] + self.dof_pos[:, 3])
        rear_error = torch.abs(self.dof_pos[:, 6] + self.dof_pos[:, 9])
        return (front_error + rear_error) * in_place_turn

    def _reward_hip_pos_penalty(self):
        """Squared hip deviation beyond the default-pose deadband during turns."""
        in_place_turn = (
            (torch.norm(self.commands[:, :2], dim=1) < 0.1)
            & (torch.abs(self.commands[:, 2]) >= 0.2)
        )
        # FL, FR, RL, RR ab/adduction joints, relative to their own defaults.
        hip_indices = [0, 3, 6, 9]
        deviation = torch.abs(
            self.dof_pos[:, hip_indices] - self.default_dof_pos[:, hip_indices]
        )
        excess = torch.clamp(
            deviation - self.cfg.rewards.hip_pos_deadband_deg * (np.pi / 180.0),
            min=0.0,
        )
        return torch.sum(torch.square(excess), dim=1) * in_place_turn

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        if self.cfg.rewards.relative_tracking_lin_vel:
            cfg = self.cfg.rewards
            command_speed = torch.norm(self.commands[:, :2], dim=1)
            speed_scale = command_speed.clamp(
                min=cfg.tracking_lin_vel_min_speed, max=cfg.tracking_lin_vel_max_speed)
            # One XY scale preserves lateral-error penalties even when vy_cmd is zero.
            moving_reward = torch.exp(
                -lin_vel_error / (cfg.tracking_lin_vel_sigma * speed_scale.square()))
            standing_reward = torch.exp(-lin_vel_error / cfg.tracking_lin_vel_sigma)
            return torch.where(
                command_speed >= cfg.tracking_lin_vel_command_threshold,
                moving_reward, standing_reward)
        return torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw) 
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _init_orientation_reference(self):
        """Precompute a weighted plane fit in a yaw-aligned, level sampling frame."""
        cfg = self.cfg.rewards
        if not self.cfg.terrain.measure_heights:
            raise ValueError("Terrain-relative orientation requires measure_heights=True")
        length, width = cfg.orientation_window_half_length, cfg.orientation_window_half_width
        values = [length, width, cfg.orientation_filter_tau,
                  cfg.orientation_deadband_deg, cfg.orientation_max_tilt_deg]
        if (not all(math.isfinite(v) for v in values) or min(length, width) <= 0
                or cfg.orientation_filter_tau < 0
                or not 0 <= cfg.orientation_deadband_deg < cfg.orientation_max_tilt_deg < 90):
            raise ValueError("Invalid terrain-relative orientation settings")
        xy = self.height_points[0, :, :2]
        mask = (xy[:, 0].abs() <= length + 1e-6) & (xy[:, 1].abs() <= width + 1e-6)
        self.orientation_height_indices = mask.nonzero(as_tuple=False).flatten()
        xy = xy[mask]
        design = torch.cat((xy, torch.ones_like(xy[:, :1])), dim=1)
        if len(xy) < 3 or torch.linalg.matrix_rank(design).item() < 3:
            raise ValueError("Orientation window needs non-collinear height samples")
        weight = torch.exp(-0.5 * ((xy[:, 0] / length)**2 + (xy[:, 1] / width)**2))
        weighted_transpose = design.T * weight
        self.orientation_plane_fit = torch.linalg.solve(
            weighted_transpose @ design, weighted_transpose)
        self.orientation_target_normal = -self.gravity_vec.clone()
        self.orientation_reference_initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device)

    def _update_orientation_reference(self):
        """Update once after fresh terrain sampling, never from a reward/obs query.

        Fit z=a*x+b*y+c around the nominal support region, not moving feet.
        Fully follow the fitted plane (alpha=1); smooth normals in WORLD frame
        so a yaw turn cannot rotate a stale local-frame reference incorrectly.
        """
        heights = self.measured_heights[:, self.orientation_height_indices]
        # Center heights for numerical stability at high world elevations.
        heights = heights - heights.mean(dim=1, keepdim=True)
        slopes = (heights @ self.orientation_plane_fit.T)[:, :2]
        limit = math.tan(math.radians(self.cfg.rewards.orientation_max_tilt_deg))
        slopes = slopes * (limit / slopes.norm(dim=1, keepdim=True).clamp_min(1e-8)).clamp(max=1.)
        local_normal = torch.cat((-slopes, torch.ones_like(slopes[:, :1])), dim=1)
        local_normal = torch.nn.functional.normalize(local_normal, dim=1)
        raw_normal = quat_apply_yaw(self.base_quat, local_normal)
        tau = self.cfg.rewards.orientation_filter_tau
        gain = 1. if tau == 0 else -math.expm1(-self.dt / tau)
        filtered = torch.nn.functional.normalize(
            self.orientation_target_normal + gain * (raw_normal - self.orientation_target_normal), dim=1)
        self.orientation_target_normal[:] = torch.where(
            self.orientation_reference_initialized[:, None], filtered, raw_normal)
        self.orientation_reference_initialized[:] = True

    def _reward_orientation(self):
        if not getattr(self.cfg.rewards, "terrain_relative_orientation", False):
            return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        # Body Z dot target normal. atan2 handles both aligned and upside-down
        # states without the near-zero sensitivity of acos.
        target_body = quat_rotate_inverse(self.base_quat, self.orientation_target_normal)
        angle = torch.atan2(target_body[:, :2].norm(dim=1), target_body[:, 2])
        excess = (angle - math.radians(self.cfg.rewards.orientation_deadband_deg)).clamp_min(0.)
        return excess.square()
    
    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_joint_power(self):
        #Penalize high power
        return torch.sum(torch.abs(self.dof_vel) * torch.abs(self.torques), dim=1)

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self._get_base_heights()
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _sample_step_heights(self, xy):
        """Unsmoothed privileged heights; grid resolution limits edge accuracy."""
        if self.cfg.terrain.mesh_type == "plane":
            return torch.zeros(xy.shape[:-1], device=self.device)
        cfg = self.terrain.cfg
        grid = (xy + cfg.border_size) / cfg.horizontal_scale
        indices = torch.floor(grid).long()
        x = indices[..., 0].clamp(0, self.height_samples.shape[0] - 2)
        y = indices[..., 1].clamp(0, self.height_samples.shape[1] - 2)
        h00 = self.height_samples[x, y] * cfg.vertical_scale
        h10 = self.height_samples[x + 1, y] * cfg.vertical_scale
        h01 = self.height_samples[x, y + 1] * cfg.vertical_scale
        h11 = self.height_samples[x + 1, y + 1] * cfg.vertical_scale
        fraction = (grid - indices).clamp(0, 1)
        u, v = fraction.unbind(-1)
        smooth = (1-u)*(1-v)*h00 + u*(1-v)*h10 + (1-u)*v*h01 + u*v*h11
        variation = torch.stack((h10-h00, h01-h00, h11-h10, h11-h01), -1).abs().amax(-1)
        # Smooth ramps should not appear as a sequence of discrete risers.
        steep = variation > cfg.slope_treshold * cfg.horizontal_scale
        # Standard Terrain moves the LOW vertex to the high vertex to make
        # vertical risers. The lower height fills the steep cell on either
        # approach direction. Custom playback meshes retain their own grid.
        edge_height = torch.minimum(torch.minimum(h00, h10), torch.minimum(h01, h11)) if isinstance(self.terrain, Terrain) else h00
        return torch.where(steep, edge_height, smooth)

    def _reward_step_retraction(self):
        return self.step_retraction.progress_reward

    def _reward_step_retraction_stall(self):
        return self.step_retraction.stall_cost

    def _reward_step_clearance(self):
        # Updated once per control step; reward queries are side-effect free.
        return self.step_clearance.cost

    def _reward_swing_peak_height(self):
        return self.swing_peak_height.score

    def _reward_foot_clearance(self):
        cur_footpos_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footpos_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        cur_footvel_translated = self.feet_vel - self.root_states[:, 7:10].unsqueeze(1)
        footvel_in_body_frame = torch.zeros(self.num_envs, len(self.feet_indices), 3, device=self.device)
        for i in range(len(self.feet_indices)):
            footpos_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footpos_translated[:, i, :])
            footvel_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footvel_translated[:, i, :])
        
        height_error = torch.square(footpos_in_body_frame[:, :, 2] - self.cfg.rewards.clearance_height_target).view(self.num_envs, -1)
        foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(self.num_envs, -1)
        return torch.sum(height_error * foot_leteral_vel, dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_smoothness(self):
        # second order smoothness
        return torch.sum(torch.square(self.actions - self.last_actions - self.last_actions + self.last_last_actions), dim=1)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.sum(torch.square(self.torques), dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_termination(self):
        # Terminal reward / penalty
        return self.reset_buf * ~self.time_out_buf
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)

    def _reward_dof_vel_limits(self):
        # Penalize dof velocities too close to the limit
        # clip to max error = 1 rad/s per joint to avoid huge penalties
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)

    def _reward_torque_limits(self):
        # penalize torques too close to the limit
        return torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)

    def _init_turn_lift_buffers(self):
        cfg = self.cfg.rewards
        values = (cfg.turn_lift_min_clearance, cfg.turn_lift_min_frequency,
                  cfg.turn_lift_window_s, cfg.turn_lift_foot_radius)
        if any(not np.isfinite(v) or v <= 0 for v in values):
            raise ValueError("Turn lift parameters must be positive and finite")
        count = cfg.turn_lift_min_frequency * cfg.turn_lift_window_s
        if abs(count - round(count)) > 1e-6 or count < 1:
            raise ValueError("Turn lift window * frequency must be a positive integer")
        steps = cfg.turn_lift_window_s / self.dt
        if abs(steps - round(steps)) > 1e-4 or steps < 1:
            raise ValueError("Turn lift window must contain an integer number of control steps")
        if not 0. <= cfg.feet_air_time_min < cfg.feet_air_time_target < cfg.feet_air_time_max:
            raise ValueError("Expected air-time min < target < max")
        self.turn_lift_window_steps = int(round(steps))
        shape = self.feet_air_time.shape
        # Integer ages avoid cumulative float error at the open left boundary.
        self.turn_lift_ages = torch.full((*shape, int(round(count))),
            self.turn_lift_window_steps, device=self.device, dtype=torch.long)
        self.turn_lift_peak = torch.zeros_like(self.feet_air_time)
        self.turn_lift_eligible = torch.zeros_like(self.feet_air_time, dtype=torch.bool)
        self.turn_lift_elapsed = torch.zeros(shape[0], device=self.device, dtype=torch.long)
        self.turn_air_steps = torch.zeros_like(self.feet_air_time, dtype=torch.long)
        self.turn_lift_frequency_reward = torch.zeros(shape[0], device=self.device)
        self.turn_air_overdue_cost = torch.zeros(shape[0], device=self.device)
        self.turn_lift_logger = None
        if getattr(cfg, "log_phase_rewards", False):
            from legged_gym.utils.phase_reward_logger import TurnLiftLogger
            self.turn_lift_logger = TurnLiftLogger(self.dt, self.device)

    def _reset_turn_lift_buffers(self, env_ids):
        self.turn_lift_ages[env_ids] = self.turn_lift_window_steps
        self.turn_lift_peak[env_ids] = 0.
        self.turn_lift_eligible[env_ids] = False
        self.turn_lift_elapsed[env_ids] = 0

    def _update_turn_lift_state(self):
        """Advance contact history once per reward step; reward accessors are pure."""
        cfg = self.cfg.rewards
        turning = ((torch.norm(self.commands[:, :2], dim=1) < 0.1)
                   & (self.commands[:, 2].abs() >= 0.2))
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        filtered = contact | self.last_contacts
        self.last_contacts = contact
        previous_steps = self.turn_air_steps.clone()
        first_contact = (previous_steps > 0) & filtered
        self.turn_air_steps += 1

        # Preserve incremental overdue cost, including the landing tick and
        # any carry-over flight that first became overdue during a turn.
        previous_overdue = torch.clamp(previous_steps * self.dt - cfg.feet_air_time_max, min=0.)
        current_overdue = torch.clamp(self.turn_air_steps * self.dt - cfg.feet_air_time_max, min=0.)
        overdue_seconds = current_overdue - previous_overdue
        self.feet_air_time_penalty_active |= (turning.unsqueeze(1)
            & (overdue_seconds > 0.) & ((previous_steps > 0) | ~filtered))
        paid_seconds = overdue_seconds * self.feet_air_time_penalty_active
        self.turn_air_overdue_cost = paid_seconds.sum(dim=1) / (
            cfg.feet_air_time_max - cfg.feet_air_time_target)

        self._reset_turn_lift_buffers(~turning)
        self.turn_lift_elapsed = torch.where(turning,
            (self.turn_lift_elapsed + 1).clamp(max=self.turn_lift_window_steps),
            self.turn_lift_elapsed)
        self.turn_lift_ages.add_(1).clamp_(max=self.turn_lift_window_steps)
        self.turn_lift_eligible |= (previous_steps == 0) & ~filtered & turning.unsqueeze(1)
        clearance = self._get_feet_heights() - cfg.turn_lift_foot_radius
        self.turn_lift_peak = torch.where(self.turn_lift_eligible & ~filtered,
            torch.maximum(self.turn_lift_peak, clearance), self.turn_lift_peak)
        valid = (first_contact & self.turn_lift_eligible & turning.unsqueeze(1)
                 & (self.turn_lift_peak >= cfg.turn_lift_min_clearance - 1e-6))
        shifted = torch.cat((torch.zeros_like(self.turn_lift_ages[..., :1]),
                             self.turn_lift_ages[..., :-1]), dim=-1)
        self.turn_lift_ages = torch.where(valid.unsqueeze(-1), shifted, self.turn_lift_ages)
        counts = (self.turn_lift_ages < self.turn_lift_window_steps).sum(dim=-1)
        quota = self.turn_lift_ages.shape[-1]
        coverage = counts.float() / quota
        ramp = self.turn_lift_elapsed.float() / self.turn_lift_window_steps
        self.turn_lift_frequency_reward = (2. * coverage.mean(dim=1) - 1.) * ramp * turning
        if self.turn_lift_logger is not None:
            self.turn_lift_logger.update(self.commands, turning,
                self.turn_lift_elapsed == self.turn_lift_window_steps,
                valid, counts >= quota, paid_seconds)
        self.turn_lift_peak.masked_fill_(filtered, 0.)
        self.turn_lift_eligible &= ~filtered
        self.feet_air_time_penalty_active &= ~filtered
        self.turn_air_steps *= ~filtered
        self.feet_air_time[:] = self.turn_air_steps * self.dt

    def _reward_turn_lift_frequency(self):
        """Ramped, per-foot capped window coverage in [-1, 1]."""
        return self.turn_lift_frequency_reward

    def _reward_turn_air_time_overdue(self):
        """Positive overdue cost, weighted negatively in the task configuration."""
        return self.turn_air_overdue_cost

    def _reward_feet_air_time(self):
        # Score normal/short flights at touchdown. With a target interval,
        # settle the extra cost beyond the upper bound incrementally.
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        previous_air_time = self.feet_air_time.clone()
        self.feet_air_time += self.dt
        target = getattr(self.cfg.rewards, "feet_air_time_target", None)
        if target is None:
            air_time_score = self.feet_air_time - 0.5
            overdue_cost = torch.zeros_like(self.feet_air_time)
        else:
            lower = self.cfg.rewards.feet_air_time_min
            upper = self.cfg.rewards.feet_air_time_max
            if not 0.0 <= lower < target < upper:
                raise ValueError("Expected 0 <= feet_air_time_min < feet_air_time_target < feet_air_time_max")
            # Keep the original touchdown scores up to upper. A long flight
            # gets zero touchdown score: its negative part is already paid.
            touchdown_time = torch.clamp(self.feet_air_time, max=upper)
            width = torch.where(touchdown_time < target, target - lower, upper - target)
            air_time_score = 1.0 - torch.abs(touchdown_time - target) / width
            # C(T)=max(T-upper,0)/(upper-target). Only pay C(T_new)-C(T_old),
            # including the final timer increment on the touchdown step.
            overdue_cost = (
                torch.clamp(self.feet_air_time - upper, min=0.)
                - torch.clamp(previous_air_time - upper, min=0.)
            ) / (upper - target)
        # Touchdown scores retain the original command gate. Once a flight
        # pays overdue cost while turning, keep paying until filtered contact.
        in_place_turn = (
            (torch.norm(self.commands[:, :2], dim=1) < 0.1)
            & (torch.abs(self.commands[:, 2]) >= 0.2)
        )
        if target is not None:
            self.feet_air_time_penalty_active |= (
                in_place_turn.unsqueeze(1) & (overdue_cost > 0.)
                & ((previous_air_time > 0.) | ~contact_filt)
            )
        rew_airTime = torch.sum(
            air_time_score * first_contact * in_place_turn.unsqueeze(1)
            - overdue_cost * self.feet_air_time_penalty_active, dim=1
        )
        self.feet_air_time_penalty_active &= ~contact_filt
        self.feet_air_time *= ~contact_filt
        return rew_airTime
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
        
    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) -  self.cfg.rewards.max_contact_force).clip(min=0.), dim=1)

    def _get_raibert_desired_foot_xy(self, commands=None):
        """Return [env, FL/FR/RL/RR, XY] targets in the full body frame."""
        if commands is None:
            commands = self.commands
        half_length = 0.5 * self.cfg.rewards.raibert_stance_length
        half_width = 0.5 * self.cfg.rewards.raibert_stance_width
        stance_center_x = self.cfg.rewards.raibert_stance_center_x
        stance_center_y = self.cfg.rewards.raibert_stance_center_y
        x_nom = torch.tensor([half_length, half_length, -half_length, -half_length], device=self.device) + stance_center_x
        y_nom = torch.tensor([half_width, -half_width, half_width, -half_width], device=self.device) + stance_center_y
        vx_cmd = commands[:, 0:1]
        wz_cmd = commands[:, 2:3]
        prediction_time = self.cfg.rewards.raibert_prediction_time
        return torch.stack((
            x_nom.unsqueeze(0) + 0.5 * prediction_time * (vx_cmd - wz_cmd * y_nom.unsqueeze(0)),
            y_nom.unsqueeze(0) + 0.5 * prediction_time * (wz_cmd * x_nom.unsqueeze(0)),
        ), dim=-1)

    def _reward_raibert_heuristic(self):
        """Penalize XY error on contact after a confirmed low-force interval.

        Targets use current commands in the full body frame (including roll/pitch).
        Confirmation suppresses brief force dropouts without delaying touchdown.
        """
        contact = self.contact_forces[:, self.raibert_foot_indices, 2] > self.cfg.rewards.raibert_contact_force_threshold
        min_air_steps = self.cfg.rewards.raibert_min_air_steps
        touchdown = (contact & (self.raibert_air_steps >= min_air_steps)
                     & self.raibert_contact_initialized.unsqueeze(1))
        self.raibert_air_steps = torch.where(contact, torch.zeros_like(self.raibert_air_steps),
            (self.raibert_air_steps + 1).clamp(max=min_air_steps))
        self.raibert_last_contacts = contact
        self.raibert_contact_initialized.fill_(True)

        command_active = (
            (torch.abs(self.commands[:, 0]) > self.cfg.rewards.raibert_vx_command_threshold)
            | (torch.abs(self.commands[:, 2]) > self.cfg.rewards.raibert_yaw_command_threshold)
        )
        touchdown &= command_active.unsqueeze(1)
        desired_xy = self._get_raibert_desired_foot_xy()

        foot_world_relative = self.feet_pos[:, self.raibert_foot_order, :] - self.root_states[:, :3].unsqueeze(1)
        foot_body = quat_rotate_inverse(
            self.base_quat.repeat_interleave(4, dim=0),
            foot_world_relative.reshape(-1, 3),
        ).view(self.num_envs, 4, 3)
        error = torch.sum(torch.square(foot_body[:, :, :2] - desired_xy), dim=-1)
        return torch.sum(error * touchdown, dim=1)

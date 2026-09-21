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

import numpy as np
import os
import inspect
from pathlib import Path
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import export_policy_as_jit, get_args, task_registry
import torch


JIT_EXPORT_INTERVAL = 1000


def print_training_code_status(env, ppo_runner, stage):
    """Report loaded code, not just the version of this entry-point script."""
    project_root = Path(__file__).resolve().parents[3]
    expected_env = project_root / 'legged_gym/legged_gym/envs/base/legged_robot.py'
    expected_runner = project_root / 'rsl_rl/rsl_rl/runners/him_on_policy_runner.py'
    print('\n' + '=' * 72, flush=True)
    print(f'[训练代码核验] {stage}', flush=True)
    print(f'训练入口：{Path(__file__).resolve()}', flush=True)
    for label, method, expected in (
        ('环境重置', env.reset_idx, expected_env),
        ('观测计算', env.compute_observations, expected_env),
        ('训练循环', ppo_runner.learn, expected_runner),
    ):
        actual = Path(inspect.getfile(method)).resolve()
        status = '正确：当前项目' if actual == expected.resolve() else '警告：不是当前项目预期文件'
        print(f'{label}：{actual}\n  [{status}]', flush=True)

    try:
        reset_source = inspect.getsource(env.reset_idx)
        observation_source = inspect.getsource(env.compute_observations)
        history_change_found = (
            hasattr(env, 'obs_history_reset_pending')
            and 'self.obs_history_reset_pending[env_ids] = True' in reset_source
            and 'self.obs_history_reset_pending[reset_ids] = False' in observation_source
            and 'current_obs[reset_ids, :self.num_one_step_obs].repeat(1, self.history_length)' in observation_source
        )
    except (OSError, TypeError):
        history_change_found = False
    print('最近改动：物理重置后，用同一个有效首帧填满观测历史；随后正常滚动。', flush=True)
    if history_change_found:
        print(f'[已检测到改动] 实际环境具有历史重置标记，源码包含首帧重复逻辑；历史长度={env.history_length} 帧。', flush=True)
    else:
        print('[警告] 未能确认该改动！请勿认为当前训练已使用新的历史重置逻辑。', flush=True)
    print('说明：这是加载路径与源码检查，不是训练效果验证；训练期间请勿修改源码。', flush=True)
    print('=' * 72 + '\n', flush=True)


def export_policy(ppo_runner, filename):
    export_dir = os.path.join(ppo_runner.log_dir, 'exported', 'policies')
    export_policy_as_jit(ppo_runner.alg.actor_critic, export_dir, filename=filename)
    export_path = os.path.join(export_dir, filename)
    print(f'Exported TorchScript policy: {export_path}')


def train(args, headless=True):
    args.headless = headless
    args.resume = False
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    remaining_iterations = train_cfg.runner.max_iterations
    init_at_random_ep_len = True
    try:
        print_training_code_status(env, ppo_runner, '第一轮采样之前')
        while remaining_iterations > 0:
            iterations = min(JIT_EXPORT_INTERVAL, remaining_iterations)
            ppo_runner.learn(
                num_learning_iterations=iterations,
                init_at_random_ep_len=init_at_random_ep_len,
            )
            if ppo_runner.current_learning_iteration == 1000:
                print_training_code_status(env, ppo_runner, '已完成第 1000 轮训练')
            export_policy(
                ppo_runner,
                f'model_{ppo_runner.current_learning_iteration}.jit',
            )
            remaining_iterations -= iterations
            init_at_random_ep_len = False
    finally:
        export_policy(ppo_runner, 'latest.jit')

if __name__ == '__main__':
    args = get_args()
    train(args, headless=True)

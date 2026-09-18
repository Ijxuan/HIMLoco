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
from datetime import datetime

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import export_policy_as_jit, get_args, task_registry
import torch


JIT_EXPORT_INTERVAL = 1000


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
        while remaining_iterations > 0:
            iterations = min(JIT_EXPORT_INTERVAL, remaining_iterations)
            ppo_runner.learn(
                num_learning_iterations=iterations,
                init_at_random_ep_len=init_at_random_ep_len,
            )
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

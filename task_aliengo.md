# AlienGo 任务配置与训练方法总结

这篇文档由gpt-5.6-sol-max总结，价值2.26元

本文基于当前仓库代码总结 `aliengo` 任务的观测、奖励/惩罚、策略网络、HIM 估计器、随机化和训练方法。对应主要文件：

- `legged_gym/legged_gym/envs/aliengo/aliengo_config.py`
- `legged_gym/legged_gym/envs/base/legged_robot_config.py`
- `legged_gym/legged_gym/envs/base/legged_robot.py`
- `rsl_rl/rsl_rl/modules/him_actor_critic.py`
- `rsl_rl/rsl_rl/modules/him_estimator.py`
- `rsl_rl/rsl_rl/runners/him_on_policy_runner.py`
- `rsl_rl/rsl_rl/algorithms/him_ppo.py`

## 1. 任务和控制周期

- 机器人：Unitree AlienGo。
- 动力学仿真：Isaac Gym PhysX，GPU pipeline。
- 仿真步长：`0.005 s`。
- 控制 decimation：`4`。
- 策略控制周期：`0.005 * 4 = 0.02 s`，即 `50 Hz`。
- 每个 episode：`20 s`，约 `1000` 个策略步。
- 动作维度：`12`，对应 AlienGo 的 12 个关节。
- 控制器：P 控制器。
- PD 参数：关节刚度 `Kp=40`，阻尼 `Kd=2`。
- 动作缩放：`0.5`。
- 策略输出先作为关节目标增量，随后通过 PD 控制器转换为力矩，并裁剪到 URDF 力矩限制内。

## 2. 策略输入维度

### 2.1 单步原始观测：45 维

代码中 `num_one_step_observations = 45`。单步观测在 `compute_observations()` 中按以下顺序拼接：

| 序号 | 组成                                      |         维度 | 说明                                                                                                                     |
| ---- | ----------------------------------------- | -----------: | ------------------------------------------------------------------------------------------------------------------------ |
| 1    | 线速度命令`commands[:, :3]`             |            3 | `v_x, v_y, v_yaw`；实际经过 `commands_scale` 缩放。由于 `heading_command=True`，第 3 个量由 heading 误差计算得到。 |
| 2    | 机体角速度`base_ang_vel`                |            3 | 机体坐标系下的`x,y,z` 角速度，乘 `ang_vel` scale `0.25`。                                                          |
| 3    | 投影重力`projected_gravity`             |            3 | 重力方向在机体坐标系的投影。                                                                                             |
| 4    | 关节位置误差`dof_pos - default_dof_pos` |           12 | 相对默认关节角，乘`dof_pos` scale `1.0`。                                                                            |
| 5    | 关节速度`dof_vel`                       |           12 | 乘`dof_vel` scale `0.05`。                                                                                           |
| 6    | 上一步动作`actions`                     |           12 | 上一个策略动作。                                                                                                         |
|      | **合计**                            | **45** | `3+3+3+12+12+12=45`。                                                                                                  |

单步观测中的噪声：

- 命令 3 维不加噪声。
- 角速度 3 维：均匀噪声幅度 `0.2 * 1.0 * 0.25 = 0.05`。
- 投影重力 3 维：均匀噪声幅度 `0.05`。
- 关节位置 12 维：均匀噪声幅度 `0.01 * 1.0 * 1.0 = 0.01`。
- 关节速度 12 维：均匀噪声幅度 `1.5 * 1.0 * 0.05 = 0.075`。
- 上一步动作 12 维：不加噪声。
- 观测噪声通过 `2 * rand - 1` 生成，即在对应的正负幅度范围内均匀采样。

### 2.2 额外输入和单步完整观测：238 维

单步基础观测 45 维之后，还拼接：

| 组成                             |                     维度 | 是否进入 actor 的单步输入 |
| -------------------------------- | -----------------------: | ------------------------- |
| 机体线速度`base_lin_vel`       |                        3 | 否，属于特权信息          |
| 外部扰动力`disturbance[:,0,:]` |                        3 | 否，属于特权信息          |
| 周围地形高度测量                 |                      187 | 否，属于特权信息          |
| **合计**                   | **45+3+3+187=238** |                           |

高度测量点为 `17 * 11 = 187` 个，覆盖约 `1.0 m * 1.6 m` 的局部区域。高度值先裁剪到 `[-1,1]`，再乘高度观测 scale `5.0`，并添加幅度 `0.1 * 5.0 = 0.5` 的均匀噪声。

### 2.3 历史观测：270 维

配置中：

```text
num_observations = 45 * 6 = 270
history_size = 6
```

环境把连续 6 个时刻的 45 维单步基础观测堆叠为策略观测：

```text
obs_history: [batch, 270] = 6 帧 * 45 维
```

虽然环境内部同时维护 `privileged_obs_buf`，其单步维度为 `238`，但 AlienGo 配置中：

```text
num_privileged_obs = 238
```

因此 critic 的输入是单步 238 维特权观测，而不是 6 帧历史。

## 3. HIM 估计器和 actor 输入

HIM 估计器的输入是 6 帧历史观测：`270` 维。

估计器 encoder：

```text
270 -> 128 -> 64 -> 19
```

最后 19 维拆分为：

- 速度估计 `vel_hat`：3 维。
- 隐变量 `z`：16 维，并做 L2 normalization。

策略 actor 实际输入为：

```text
当前单步观测 45
+ 估计速度 vel_hat 3
+ HIM latent z 16
= 64 维
```

注意：actor 不直接读取 270 维历史，也不直接读取 238 维 privileged observation；历史由 HIM encoder 压缩为速度和 latent 后再进入 actor。

## 4. 网络结构和参数量

以下参数量按当前 AlienGo 配置、全连接层 bias 和可训练动作噪声参数计算。

### 4.1 Actor

结构：

```text
64 -> Linear(512) -> ELU
   -> Linear(256) -> ELU
   -> Linear(128) -> ELU
   -> Linear(12)
```

参数量：

| 层                   |            参数量 |
| -------------------- | ----------------: |
| `64 -> 512`        |        `33,280` |
| `512 -> 256`       |       `131,328` |
| `256 -> 128`       |        `32,896` |
| `128 -> 12`        |         `1,548` |
| **Actor 合计** | **199,052** |

Actor 输出的是 12 维高斯分布的均值 `action_mean`。训练时从：

```text
Normal(action_mean, action_std)
```

中采样 12 维动作。`action_std` 是 12 个可训练标量，初始值都是 `1.0`。

因此 actor 模块加动作噪声参数后为：

```text
199,052 + 12 = 199,064 个可训练参数
```

推理时 `act_inference()` 直接输出 12 维均值动作，不进行随机采样。

### 4.2 Critic

结构：

```text
238 -> Linear(512) -> ELU
    -> Linear(256) -> ELU
    -> Linear(128) -> ELU
    -> Linear(1)
```

参数量：

| 层                    |            参数量 |
| --------------------- | ----------------: |
| `238 -> 512`        |       `122,368` |
| `512 -> 256`        |       `131,328` |
| `256 -> 128`        |        `32,896` |
| `128 -> 1`          |           `129` |
| **Critic 合计** | **286,721** |

Critic 输出一个标量状态价值 $V(s)$。

### 4.3 HIM Encoder

结构：

```text
270 -> Linear(128) -> ELU
    -> Linear(64) -> ELU
    -> Linear(19)
```

参数量：

| 层                     |           参数量 |
| ---------------------- | ---------------: |
| `270 -> 128`         |       `34,688` |
| `128 -> 64`          |        `8,256` |
| `64 -> 19`           |        `1,235` |
| **Encoder 合计** | **44,179** |

输出 3 维速度估计和 16 维 latent。

### 4.4 HIM Target Network

结构：

```text
45 -> Linear(128) -> ELU
   -> Linear(64) -> ELU
   -> Linear(16)
```

参数量：

| 层                    |           参数量 |
| --------------------- | ---------------: |
| `45 -> 128`         |        `5,888` |
| `128 -> 64`         |        `8,256` |
| `64 -> 16`          |        `1,040` |
| **Target 合计** | **15,184** |

Target network 用于对下一时刻观测提取 latent target，参与 contrastive swap loss。

### 4.5 Prototype Embedding

```text
Embedding(32, 16)
```

参数量：`32 * 16 = 512`。

### 4.6 总参数量

按模块统计：

| 模块                   |            参数量 |
| ---------------------- | ----------------: |
| Actor，不含 action std |       `199,052` |
| Critic                 |       `286,721` |
| HIM Encoder            |        `44,179` |
| HIM Target             |        `15,184` |
| Prototype              |           `512` |
| Action std             |            `12` |
| **总计**         | **545,660** |

所有这些模块都注册在 `HIMActorCritic` 中，并由同一个 Adam 优化器管理；HIM encoder、target 和 prototype 也会通过估计器优化器更新。代码中的 actor 更新步骤使用 `self.actor_critic.parameters()`，因此 target/prototype 也属于模型参数集合，但估计器有单独的 encoder/target/prototype 优化过程。

## 5. 策略网络输出和动作执行

策略网络最终输出：

```text
12 维动作
```

动作对应 12 个关节的目标位置增量。执行流程为：

1. actor 输出 12 维动作均值，训练时从高斯分布采样。
2. 动作裁剪到 `[-100, 100]`。
3. 动作乘 `action_scale=0.5`。
4. 加到默认关节角上，形成目标关节位置。
5. 通过 P 控制器计算力矩：

$$
\tau = K_p (q_{target}-q) - K_d \dot q
$$

6. `Kp`、`Kd` 还会乘以每个环境的随机化因子。
7. 最终力矩裁剪到 URDF 的 torque limits。
8. 每个策略动作保持 4 个仿真子步。

## 6. 当前实际启用的奖励和惩罚

`_prepare_reward_function()` 会删除权重为 0 的项目，并将非零权重乘以控制周期 `dt=0.02`。AlienGo 当前真正参与 reward 的项目如下：

| 名称                 |        系数 | 类型 | 实际含义                                                    |
| -------------------- | ----------: | ---- | ----------------------------------------------------------- |
| `tracking_lin_vel` |    `+1.0` | 奖励 | XY 平面线速度跟踪，$\exp(-\|v_{cmd}-v\|^2 / 0.25)$。      |
| `tracking_ang_vel` |    `+0.5` | 奖励 | yaw 角速度跟踪，$\exp(-(\omega_{cmd}-\omega)^2 / 0.25)$。 |
| `lin_vel_z`        |    `-2.0` | 惩罚 | 惩罚机体 Z 方向线速度平方，抑制上下弹跳。                   |
| `ang_vel_xy`       |   `-0.05` | 惩罚 | 惩罚机体 X/Y 轴角速度平方。                                 |
| `orientation`      |    `-0.2` | 惩罚 | 惩罚投影重力 XY 分量平方，促使机身保持水平。                |
| `dof_acc`          | `-2.5e-7` | 惩罚 | 惩罚关节加速度平方。                                        |
| `joint_power`      |   `-2e-5` | 惩罚 | 惩罚 $|\dot q| |\tau                                        |
| `base_height`      |    `-1.0` | 惩罚 | 惩罚机身高度与目标高度`0.30 m` 的平方误差。               |
| `foot_clearance`   |   `-0.01` | 惩罚 | 惩罚摆动脚高度偏离`-0.20 m` 目标且存在横向速度的程度。    |
| `action_rate`      |   `-0.01` | 惩罚 | 惩罚相邻动作变化平方。                                      |
| `smoothness`       |   `-0.01` | 惩罚 | 惩罚二阶动作差分，抑制动作抖动。                            |

总 reward 的代码逻辑是：

```text
总 reward = 所有非零项的加权和
```

本任务设置：

```python
only_positive_rewards = False
```

所以总 reward 不会被裁剪为非负数。

### 6.1 实现但当前未启用的奖励/惩罚

以下函数在 `legged_robot.py` 中实现，但 AlienGo 的权重为 0，因此当前不参与优化：

| 名称                           | AlienGo 系数 | 说明                                              |
| ------------------------------ | -----------: | ------------------------------------------------- |
| `termination`                |     `-0.0` | 非超时终止奖励/惩罚；当前关闭。                   |
| `feet_air_time`              |      `0.0` | 鼓励脚离地时间；当前关闭。                        |
| `collision`                  |     `-0.0` | 惩罚指定身体碰撞；当前关闭。                      |
| `feet_stumble` / `stumble` |     `-0.0` | 惩罚脚受到横向碰撞；当前关闭。                    |
| `stand_still`                |     `-0.0` | 零速度命令时惩罚关节偏离默认位置；当前关闭。      |
| `torques`                    |     `-0.0` | 惩罚力矩平方；当前关闭。                          |
| `dof_vel`                    |     `-0.0` | 惩罚关节速度平方；当前关闭。                      |
| `dof_pos_limits`             |      `0.0` | 惩罚接近关节位置限制；当前关闭。                  |
| `dof_vel_limits`             |      `0.0` | 惩罚接近关节速度限制；当前关闭。                  |
| `torque_limits`              |      `0.0` | 惩罚接近力矩限制；当前关闭。                      |
| `feet_contact_forces`        |       未配置 | 函数存在，但没有对应的 reward scale，因此不调用。 |

注意：机器人接触 `base` 会触发 episode reset，但因为 `termination=-0.0`，当前不会额外产生终止惩罚；reset 本身仍然发生。

## 7. 训练方法

当前训练器配置为：

```text
runner: HIMOnPolicyRunner
algorithm: HIMPPO
policy: HIMActorCritic
```

核心方法包括：

### 7.1 PPO

- On-policy rollout。
- 每个环境每次收集 `100` 个策略步。
- 默认每次更新收集 `4096 * 100 = 409,600` 条环境 transition。
- 每次 PPO 更新：`5` 个 learning epochs。
- 每个 epoch：`4` 个 mini-batches。
- clip 参数：`0.2`。
- 折扣因子：`gamma=0.99`。
- GAE 参数：`lambda=0.95`。
- value loss coefficient：`1.0`。
- entropy coefficient：`0.01`。
- Adam 学习率：`1e-3`。
- 最大梯度范数：`1.0`。
- 使用 clipped value loss。
- 使用 adaptive KL 学习率调度，目标 KL：`0.01`，学习率限制为 `[1e-5, 1e-2]`。

### 7.2 HIM 估计和对比学习

HIM 估计器使用 6 帧历史观测：

```text
obs_history: 270 维
```

它输出：

- 3 维当前/下一状态速度估计。
- 16 维归一化隐变量。

估计器训练目标：

```text
estimation_loss = MSE(预测速度, 下一时刻特权观测中的真实速度)
swap_loss = prototype-based contrastive swap loss
总估计器损失 = estimation_loss + swap_loss
```

对比学习细节：

- prototype 数量：`32`。
- prototype 维度：`16`。
- temperature：`3.0`。
- Sinkhorn 正则化温度：`0.05`。
- Sinkhorn 迭代次数：`3`。
- encoder、target 和 prototype 使用估计器自己的 Adam 优化器。

### 7.3 是否使用师生学习

当前代码没有实现传统的 teacher-student learning 或 policy distillation：

- 没有 teacher policy 网络。
- 没有 student policy 网络。
- 没有 teacher action、teacher logits 或蒸馏损失。
- critic 使用 privileged observation 属于 asymmetric actor-critic，不等同于师生学习。
- HIM target network 是对比学习中的目标分支，也不是教师策略。

## 8. 域随机化

当前 AlienGo 继承基类中的以下随机化配置。

### 8.1 已启用的物理和执行器随机化

| 随机化变量        | 范围/设置                        | 作用时机                                                 |
| ----------------- | -------------------------------- | -------------------------------------------------------- |
| 机身 payload mass | 额外质量`[-1, 2]`              | 创建环境和 reset 时设置机身质量。                        |
| 机身质心位移      | X/Y/Z 各`[-0.05, 0.05] m`      | 创建环境和 reset 时设置 base COM。                       |
| 地面摩擦系数      | `[0.2, 1.25]`                  | 每个环境独立随机，reset 时重新随机并写回刚体形状属性。   |
| 电机强度          | `[0.9, 1.1]`                   | reset 时重新随机，作用于力矩。                           |
| Kp 因子           | `[0.9, 1.1]`                   | reset 时重新随机，作用于 P 控制器。                      |
| Kd 因子           | `[0.9, 1.1]`                   | reset 时重新随机，作用于 P 控制器。                      |
| 初始关节位置      | 默认关节位置乘`[0.5, 1.5]`     | 每次 reset 随机，初始关节速度为 0。                      |
| 初始 XY 位置      | 每个方向在`[-1,1] m` 内扰动    | 创建环境和 reset 时使用。                                |
| 初始根部线/角速度 | 6 个分量均在`[-0.5,0.5]`       | 每次 reset 设置。                                        |
| action delay      | `0` 到 `decimation-1` 个子步 | 每个 step 为每个环境随机延迟动作。                       |
| 随机推搡          | XY 速度分量`[-1,1]`            | `push_interval_s=16 s`，通过修改根部 XY 速度模拟推搡。 |
| 外部扰动力        | 3 维局部力各在`[-30,30]`       | `disturbance_interval=8` 个仿真步执行一次。            |

实际控制周期为 `0.02 s`，因此外力扰动间隔约为 `8 * 0.005 = 0.04 s` 的仿真计数口径；推搡间隔为 `16 s / 0.02 s = 800` 个策略 step。代码中 `disturbance_interval` 直接按仿真 step 使用，而 `push_interval` 在 `_parse_cfg()` 中转换为策略控制周期。

### 8.2 当前关闭的随机化

| 随机化变量                | 配置      |
| ------------------------- | --------- |
| link mass randomization   | `False` |
| restitution randomization | `False` |

## 9. 命令和地形课程学习

当前同时启用了两类 curriculum。

### 9.1 地形课程

```text
terrain.mesh_type = trimesh
terrain.curriculum = True
terrain.num_rows = 10
terrain.num_cols = 20
max_init_terrain_level = 5
```

地形类型比例：

```text
smooth slope: 0.1
rough slope: 0.2
stairs up: 0.3
stairs down: 0.3
discrete: 0.1
```

机制：

- 初始时每个环境随机分配 `0` 到 `5` 级地形。
- 如果机器人移动距离超过地形长度的一半，升级地形难度。
- 如果移动距离不足以满足当前命令要求，降级地形难度。
- 达到最高等级后，重新随机到一个等级。

### 9.2 速度命令课程

```text
commands.curriculum = True
max_curriculum = 3.0
```

当低速组和高速组的线速度跟踪奖励都超过阈值时，扩大 `lin_vel_x` 范围：

- 下限每次减少 `0.2`，最低到 `-3.0`。
- 上限每次增加 `0.2`，最高到 `3.0`。

命令重采样时间：`10 s`。

命令范围初始值：

- `lin_vel_x`: `[-1.0, 1.0] m/s`
- `lin_vel_y`: `[-1.0, 1.0] m/s`
- `ang_vel_yaw`: `[-3.14, 3.14] rad/s`
- `heading`: `[-3.14, 3.14] rad`

由于 `heading_command=True`，实际 yaw 角速度命令由 heading 误差计算，并裁剪到 `[-2,2] rad/s`。

## 10. 观测噪声和部分可观测处理

当前 `add_noise=True`，使用均匀噪声模拟传感器误差。噪声应用于：

- 角速度。
- 投影重力。
- 关节位置。
- 关节速度。
- 局部地形高度。

命令和历史动作不加噪声。

HIM 使用 6 帧历史观测处理部分可观测性，并通过历史序列估计速度与环境响应 latent。actor 只接收最小传感器风格的当前观测、估计速度和 latent；critic 则额外使用特权观测进行训练。

## 11. 结论

当前 AlienGo 训练是：

```text
rough-trimesh terrain
+ terrain curriculum
+ command curriculum
+ domain randomization
+ observation noise
+ HIM velocity/latent estimator
+ prototype contrastive learning
+ asymmetric actor-critic
+ PPO
```

策略输出为 12 维随机动作/确定性均值动作，最终通过 PD 控制器驱动 12 个关节。当前没有师生策略学习，也没有 H-infinity 对抗扰动算法；外部扰动是域随机化的一部分，不是独立的可学习 disturber。

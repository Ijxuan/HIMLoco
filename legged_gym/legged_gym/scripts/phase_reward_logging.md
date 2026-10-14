# 按指令阶段记录训练奖励

Mini Cheetah 默认开启 `rewards.log_phase_rewards = True`。HIM 训练器每轮
rollout 结束后写入以下 TensorBoard 分组，保留原 `Episode/*` 曲线便于比较旧训练。
排查原地转动时应看 `PhaseReward/*`，不再用混合的 Episode 曲线代替转动表现。

| 分组 | 分类规则 |
|---|---|
| `PhaseReward/translate` | 平移指令模长 ≥ 0.1 m/s，包括同时有 yaw 指令的运动 |
| `PhaseReward/turn` | 平移指令模长 < 0.1 m/s，且 yaw 指令幅值 > 1e-6 rad/s |
| `PhaseReward/stand` | 其余指令，即平移死区内且 yaw 近零 |
| `PhaseReward/turn_positive` | turn 中 yaw > 0 的样本 |
| `PhaseReward/turn_negative` | turn 中 yaw < 0 的样本 |

分类使用奖励计算时的指令，不使用实际机身速度：站立指令下发生漂移，仍属于
stand。turn 日志也包含低于 0.2 rad/s 的小转动指令；此时原有抬脚/髋部镜像奖励
门控可能不生效。两者定义分别用于日志分类和奖励生效，日志不修改奖励门控。

每组记录全部启用的奖励项，包括 `rew_feet_air_time`、`rew_tracking_ang_vel`、
`rew_tracking_lin_vel`、`rew_raibert_heuristic`、`rew_orientation`、
`rew_base_height` 及启用时的 `rew_hip_mirror`。

**奖励曲线单位为已乘权重的每秒贡献：**本轮该组所有 env-step 的奖励和，除以
`该组 env-step 数 × dt`。各奖励项先由原计算链统一乘权重和 dt；日志只读取其结果。
每组只使用该组的样本作分母，不按回合长短再次平均。跨回合、回合中指令切换以及
尚未结束的回合均可立即记录，episode reset 不清除当前 rollout 统计。

例如在 TensorBoard 查看：

```text
PhaseReward/turn/rew_feet_air_time
PhaseReward/translate/rew_tracking_lin_vel
PhaseReward/stand/rew_orientation
PhaseReward/turn_positive/rew_tracking_ang_vel
PhaseReward/turn_negative/rew_tracking_ang_vel
```

各组还有：

- `samples`：本轮该组 env-step 数，初始化/reset 的采样外步骤不计入。
- `sample_fraction`：占本轮总 env-step 的比例。前三组互斥且完备；正负转向是 turn
  的子集，五组不能一起相加。
- `total_unclipped`：各原始奖励项之和的每秒贡献，包括 termination 项。
- `total_policy`：PPO 实际收到的总奖励的每秒贡献，含总奖励截断及 termination。
- `clipped_fraction`：总奖励非负截断实际生效的样本比例；在 termination 加入前判定。

一轮没有某组样本时只记录样本数/比例为 0，不写该组奖励值，避免把没有测到
误表示成得分 0。每轮导出后清空统计；新的 learn 调用会先排除采样外初始化步骤。

现有奖励函数只调用一次，日志不改变 PPO 使用的奖励、网络输入或训练采样。
日志默认在其他任务关闭；启用此选项的任务需使用支持 `pop_phase_reward_logs()`
的 HIM 训练器。正在运行的旧训练进程需重启/续训后才能产生新曲线，历史事件文件
无法从已有混合均值中恢复阶段曲线。

验证：`test_phase_reward_logging.py` 检查阶段转换、按样本归一化、正负转向、空组、
统计清空和奖励计算透明性；GPU headless 两轮短训练验证 TensorBoard 实际写入。

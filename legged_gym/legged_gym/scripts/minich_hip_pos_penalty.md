# Mini Cheetah 原地转向髋角惩罚

`hip_pos_penalty` 约束 FL、FR、RL、RR 的外展/内收髋关节。
参考角度为 `default_dof_pos`（当前左侧 +0.1 rad、右侧 -0.1 rad），
不是机械零位，也不是策略输出目标角度。使用仿真实际关节角度计算。

仅在 `norm(commands[:, :2]) < 0.1` 且 `abs(commands[:, 2]) >= 0.2`
时启用，正负转向均包含；不根据实际速度判断。

```text
excess_i = max(abs(q_i - default_q_i) - deg2rad(hip_pos_deadband_deg), 0)
raw_cost = sum(excess_i ** 2)  # 四髋相加，rad²
weighted_reward = hip_pos_penalty_scale * raw_cost * control_dt
```

Mini Cheetah 初始配置：`hip_pos_deadband_deg = 3.0`、
`scales.hip_pos_penalty = -1.0`。通用配置权重为 0。
范围内及边界处零惩罚；一个髋偏离默认角度4°时，超出1°，
原始代价约0.0003046；偏离5°时，原始代价约0.0012185（四倍）。
权重 -1 下，已有 Oct15 单次历史更新的 ±0.25 rad/s 回放，
离线估算新项平均贡献约 -0.2043/-0.2290 每秒。
这只是初始量级依据，不证明重新训练后的转向效果。

训练及回放奖励统计自动包含此项：
`Episode/rew_hip_pos_penalty`，以及
`PhaseReward/{turn,turn_positive,turn_negative,translate,stand}/rew_hip_pos_penalty`。
后两组该项应为0。总奖励沿用原有非负截断规则。

该项限制髋角偏离，不直接惩罚机身扭动或奖励抬脚。
训练后需同时验证逐髋超限幅度、yaw跟踪、roll/pitch、XY漂移及逐足抬脚，
避免仅凭惩罚变小判断动作改善。

# 足端竖直力物理子步惩罚

Mini Cheetah 启用 `foot_vertical_force`，初始权重 `-1e-4`。策略周期保持20 ms，物理步5 ms；每次策略动作内刷新四次净接触力，只使用foot刚体的世界坐标Fz。其他任务默认权重0，不增加子步采样。

每个物理步的代价为四足 `max(Fz - T, 0)^2` 之和，奖励函数返回四步平均值。奖励框架再乘控制周期dt一次：

```text
reward = scale * control_dt * mean_substeps(sum_feet(max(Fz - T, 0)^2))
       = scale * physics_dt * sum_substeps(sum_feet(max(Fz - T, 0)^2))
```

不会先平均力再计算超限。负Fz和水平力不参与。该项是超阈值力平方成本，不是冲量、机械功率或力矩惩罚。

配置位于 `envs/minich/minich_config.py`：

- `rewards.scales.foot_vertical_force = -1e-4`：小幅初始权重，尚未验证训练收益。
- `foot_vertical_force_flat_threshold = 130.0` N/足。
- `foot_vertical_force_stair_threshold = 160.0` N/足（用户明确选择，使用5ms采样）。
- `foot_vertical_force_flat_angle_deg = 5.0`。
- `foot_vertical_force_stair_angle_deg = 10.0`。
- `foot_vertical_force_min_forward_speed = 0.1` m/s。

目标上坡角从已有地形拟合目标法向与机身航向计算，正值为机身前方向上坡。≤5°为130 N，≥10°为160 N，中间线性过渡。实际机身俯仰不用于切换。目标法向和指令门控在动作开始时固定，因此采用最近可用的地形目标，有至多一个控制周期的参考更新延迟；不额外推进观测历史。仅前进指令vx≥0.1 m/s启用，站立、后退和纯横移不启用；正向移动同时转弯仍启用。下坡目前使用平地阈值，未单独标定。

累加缓存每个动作开始清零；环境reset只清该环境的数据；奖励查询不更新状态。末子步已经刷新接触力时，post_physics_step复用该读数以避免重复刷新。

开启该项要求 `terrain_relative_orientation=True` 且启用PhysX contact_collection。权重设为0即可关闭采样与奖励。记录使用现有 `Episode/rew_foot_vertical_force` 及按阶段奖励日志机制（实际tag由runner/logger命名）。本项仍受整个奖励总和的only_positive_rewards截断影响。

## 验证

- 11项独立单元测试：阈值插值、航向不变性、实际pitch隔离、前进门控、中间峰值不丢失、仅正Fz、持续时间积分、reset、动作间清零及dt不重复。
- 6项原地形姿态测试通过。
- GPU平地/楼梯验证与开关对照：`logs/policy_eval_20261027_fz_reward_smoke/validation.json`。楼梯27个非零机器人控制帧，物理子步重算与缓存误差0；奖励累计符合scale×dt；平地开关奖励时固定策略原始状态、力和力矩完全相同。
- 历史21条轨迹固定重算：同目录rescore.json；平均惩罚约0至−0.00250/秒，最大单控制步扣分约0.0401。包含失败轨迹，仅检验量级，不代表训练效果或奖励截断后的实际收益。

```bash
/opt/miniconda3/envs/gym4/bin/python -m unittest discover -s legged_gym/legged_gym/tests -p test_foot_vertical_force.py -v
/opt/miniconda3/envs/gym4/bin/python legged_gym/logs/policy_eval_20261027_fz_reward_smoke/validate.py
```

没有启动训练。后续应检查速度跟踪、楼梯完成、正常支撑受罚比例和总奖励截断率；若Fz方案效果不好，按用户要求再评估关节力矩尖峰惩罚。

# Mini Cheetah 持续逐足抬脚奖励

当前 Mini Cheetah 用两个独立项替代原 `feet_air_time`：

- `turn_lift_frequency`：持续的逐足滑动窗口频率得分，初始权重 **+0.25**。
- `turn_air_time_overdue`：超时悬脚代价，初始权重 **-1.0**。

Mini Cheetah 的 `feet_air_time` 权重为0；其他任务默认不启用新项，沿用旧公式。
旧曲线与新项含义不同，不能直接拼接比较。观测、策略网络和原有总奖励截断保持原规则。

## 最低标准与计数

- 足底相对局部地形净空至少1cm（含边界），足球半径2cm。
- 每足最近4秒至少两次有效抬起并落地，即最低0.5Hz。
- 原地转向：`norm(commands[:2]) < 0.1` 且 `abs(commands[2]) >= 0.2`。
- 有效动作必须在当前连续转向阶段中开始离地、达到高度并落地。
  转向开始前已经悬空，或中途经过非转向指令的动作不计数。
- 正负转向使用相同规则；直接从正转换为负转且始终满足门控时，视为连续转向。
- 窗口左开右闭：当前控制步k，记录k-W < event_step <= k。
  使用整数控制步计龄，dt=0.02时W=200，年龄199有效、200过期。
  配置窗口必须对应整数控制步，窗口秒数×最低频率必须为正整数。

每足只需保存最近两次有效落地用于判定达标。每次有效落地都更新记录；
更高频率不会得到额外奖励，也不能由一足的超额次数补偿另一足。

## 频率公式和启动

```text
n_i = min(该足窗口内有效落地次数, quota)
s_i = n_i / quota
ramp = min(连续转向控制步数 / window_steps, 1)
raw_frequency_reward = ramp * (2 * mean_i(s_i) - 1)
```

前4秒线性增大该项权重；始终使用完整窗口配额2，启动期间不把短窗口当成完整达标。
这会同时减弱启动阶段的正、负反馈。完整窗口后，原始得分范围[-1,+1]：

| 状态 | 原始得分 | 权重0.25后的每秒贡献 |
|---|---:|---:|
| 四足没有有效落地 | -1 | -0.25 |
| 每足各一次 | 0 | 0 |
| 每足至少两次 | +1 | +0.25 |
| 三足达标、一足零次 | +0.5 | +0.125 |

取消原来的“缺额时落脚才发奖金”。持续达标会持续得正分，不存在补齐欠额的事件奖金。
平均分仍不能保证每一足达标，必须一起查看最差足和四足同时达标率。

## 独立悬脚超时

保留原超时逻辑：计时超过0.30秒后，每步计算
`(max(T_new-0.30,0)-max(T_old-0.30,0))/(0.30-0.25)`，四足求和。
计时使用整数控制步；延续原接触滤波、落地当步计时和代价累计口径。
这个函数返回非负代价，由负权重施加惩罚；不使用频率项的起转ramp。
一旦在转向期间触发超时，指令改变后仍持续付费到落地。
未在转向中触发的纯站立悬脚不扣此项。

两项状态在 `compute_reward()` 中统一更新一次，两个奖励函数只读取缓存。
奖励调用顺序、是否关闭其中一项，不会重复推进计时。
配置仍启用旧feet_air_time且创建新计数器会报错，避免重复计时/惩罚。
episode reset清理选定环境的窗口、资格、峰值、计时、超时锁存和奖励缓存。

## TensorBoard

已乘权重的每秒奖励贡献：

```text
PhaseReward/turn/rew_turn_lift_frequency
PhaseReward/turn/rew_turn_air_time_overdue
PhaseReward/turn_positive/rew_turn_lift_frequency
PhaseReward/turn_negative/rew_turn_lift_frequency
```

不受奖励权重影响的状态指标（以FL为例）：

```text
TurnLift/turn/FL/lift_rate_hz
TurnLift/turn/FL/window_pass_fraction
TurnLift/turn/FL/overdue_seconds
TurnLift/turn/FL/overdue_fraction
TurnLift/turn/worst_foot_window_pass_fraction
TurnLift/turn/all_feet_window_pass_fraction
```

`TurnLift` 有 turn、turn_positive、turn_negative、startup、steady 五组。
严格使用奖励门控，只有|yaw|>=0.2的原地转向样本进入统计。
`PhaseReward/turn` 仍沿用较宽的转向分类（包含0<|yaw|<0.2），注意两者分母可能不同。

- `lift_rate_hz`：本轮该组所有有效落地数 / 该组环境采样秒数；不会在0.5Hz封顶。
- `window_pass_fraction`：仅对已经连续转向满4秒的样本统计，排除起转窗口。
- `all_feet_window_pass_fraction`：同一采样步四足同时达标比例。
- `worst_foot_window_pass_fraction`：四个逐足达标比例的最小值。
- `overdue_seconds`：该组累计的每足超时秒数增量，汇总所有环境；
  `overdue_fraction`再除以该组环境采样秒数，便于跨批次比较。
- `samples`、`full_window_samples`给出分母。没有完整窗口时不输出达标比例，避免把未测到写成零。

状态指标的超时统计只覆盖转向样本。切换为站立/平移后的悬脚续罚，
可通过对应`PhaseReward`组的`rew_turn_air_time_overdue`查看。
rollout导出后清空累计日志，episode reset不清除本轮已采集日志。

## 初始尺度与验证

对已有Oct14与Oct15_09-24-28、±0.25rad/s的采样离线重算，新频率项约
-0.13至-0.15/秒。选择0.25作为初始权重，完整达标时为+0.25/秒；
超时惩罚保持原权重等效幅度。离线重算不是新策略表现，也不证明训练改进。

验证产物：`legged_gym/logs/turn_lift_v3_validation_20261015/`。
行为测试包含0.5/1/2Hz持续得+1、2.02秒周期不再得更高分、精确窗口过期、
逐足封顶、起转ramp、接触与高度边界、超时续罚、选择性重置、奖励顺序、
单项关闭以及逐足日志分母。GPU PhysX + GPU pipeline + headless执行240步，
写入并重新读取TensorBoard标签。需要重新训练或续训后验证真实策略的转速、
姿态、漂移及逐足达标率；独立评分脚本仍采用原最低抬脚验收标准。

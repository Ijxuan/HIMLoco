# 楼梯零速周期踏步接口 v1

本实现借鉴 Walk These Ways 的相位条件化、摆动脚卸载和支撑脚速度代价；脚高改为局部地形净空。
参考：[奖励源码](https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym/envs/rewards/corl_rewards.py)、[相位实现](https://github.com/Improbable-AI/walk-these-ways/blob/master/go1_gym/envs/base/legged_robot.py)。这里是适配本项目的实现，不是原项目训练结果复现。

## 默认参数

- 目标地形法向相对世界竖直倾角 >5° 开启，<3° 退出。
- 仅平移指令范数 <0.1 m/s 且 |偏航指令|<0.1 rad/s 时运行。
- 周期频率 1.5 Hz，支撑占空比 0.60；每脚一次摆动约 0.267 s。
- 对角偏移：FL/RR=0，FR/RL=0.5；按脚名绑定，不能假定资产数组顺序。
- 摆动净空目标 0.035 m。每脚使用脚底世界高减局部地面高，不能用平均四脚高度或绝对世界 Z。
- `stair_gait_force=-0.4`、`stair_gait_height=-1.0`、`stair_gait_velocity=-0.1`，为初始候选权重，尚未训练验证。
- 原 `stair_step` 和 `stair_step_overdue` 权重为 0；原事件追踪器仍运行，保留有效交替事件等诊断。
- 保留楼梯命令重采样时 30% 的 2–5 s 零速场景，不修改其它训练场景比例。

## 相位与代价

每脚相位 p=(全局相位+脚偏移) mod 1；摆动进度 u=clip((p-duty)/(1-duty),0,1)。
目标脚高 h*=H*sin(pi*u)，平滑摆动权重 s=sin(pi*u)^2。
逐控制步计算四脚平均的三个非负代价，再由负权重和环境 dt 加入总奖励：

- 摆动卸载：mean[s*(1-exp(-||F||²/25²))]。
- 净空跟踪：mean[s*clip(((h-h*)/H)², max=1)]。
- 支撑稳定：mean[(1-s)*(1-exp(-||v_foot_world||²/0.15²))]。

它们不依赖“完成整步落地”才能给反馈；最高单项代价不超过 1，实际值由当时摆动权重决定。保留零速跟踪、姿态和其它稳定性约束。支撑切换有四脚接触重叠时间。

动作由当前观测中的相位产生；物理步后先用该相位评分，再推进 f*dt 生成下一观测。新激活的第一帧是相位 0，本次激活步不评价此前不可见的时钟。停止模式时相位和两个时钟通道清零；下次从相位 0 重新启动。读观测、读奖励及计算终止观测均不能推进时钟。

## 新训练与导出契约

- 每帧原 45 项不变，追加索引 45/46 为 `[sin(2*pi*phase), cos(2*pi*phase)]`，不缩放、不加噪声。
- 非激活状态为 `[0,0]`。激活状态的单位幅值同时表示踏步模式，不能把停用写成 `[0,1]`。
- actor 输入为最新帧在前的 6×47=282；critic 为 47+3 速度+3 扰动+187 地形=240。
- 估计器真实速度标签随帧宽移到 critic 索引 47:50。
- reset 后所有历史帧来自新 episode 首帧，时钟清零，不能继承上一段楼梯的相位。
- 新 checkpoint 保存 `observation_contract`，包含步频、占空比、模式门槛、dt 等。加载时拒绝不同维度和不同契约，避免模型看见错误的周期含义。
- 必须训练新模型；旧 270 输入权重不能通过补零直接当成新模型。原版 45 维模型的 JIT/ONNX 导出仍支持。
- `export_minich_onnx.py` 从 checkpoint 的权重形状推断维度，47 维模型必须带契约。导出图和验证 JSON 同时保存契约。

## 真机边界

当前 `/home/xjtx/rl-deploy/Cheetah-Software-RL` 的 `RapidRLPolicyRunner`/`RapidRLPolicyConfig` 仍固定为 45×6=270，不能加载新 282 输入模型。本次没有替换其模型或二进制。

在部署支持新模型前必须实现：相同的 47 维观测及历史、相同的控制时序/相位重置，以及与训练一致的踏步模式输入。训练用的是地形拟合目标角度；当前真机控制器不能凭空获得该信号，不能直接拿实际机身倾角替代。需要明确可部署的目标角度估计来源，或显式模式命令并在训练中统一其语义，再同步修改部署端。新增时钟也不应破坏平地/楼梯独立历史和交接流程。

## 验证

`test_stair_gait.py` 覆盖模式滞回、对角偏移、周期、当前动作与评分时序、密集代价、局部 reset、观测布局、privileged 速度索引、checkpoint 契约及 45/47 维 JIT/ONNX 数值一致性。

GPU 接线记录在 `legged_gym/logs/stair_clock_20261028_validation/`：3 环境、180 步，平地不激活、楼梯持续激活、所有观测/推理/奖励有限。未训练网络仅用于推理和导出测试，保存在 `untrained_export_fixture/`，不能部署或用于行为结论。

导出验证可选 `--backend reference` 使用 ONNX 官方 ReferenceEvaluator；默认仍用 ONNX Runtime。本次 47 维未训练测试网络通过 279 个 checkpoint/JIT/ONNX 数值对照样本，最大绝对误差约 3.58e-6；不是 ONNX Runtime 或真机执行结果。

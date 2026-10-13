# 实机 LCM 关节指令和反馈解析

日志：`record.lcm.00`；持续 62.117058 秒。

## 字段对应关系

| 含义 | 通道 | 字段 | 单位 |
|---|---|---|---|
| 目标关节位置 | leg_control_command | q_des[12] | rad |
| 目标关节速度 | leg_control_command | qd_des[12] | rad/s |
| 前馈力矩 | leg_control_command | tau_ff[12] | N·m |
| 关节 PD 增益 | leg_control_command | kp_joint[12], kd_joint[12] | N·m/rad, N·m·s/rad |
| 当前关节位置 | leg_control_data | q[12] | rad |
| 当前关节速度 | leg_control_data | qd[12] | rad/s |
| 软件力矩估计 | leg_control_data | tau_est[12] | N·m |

## 数据验证

两个关节通道各有 31058 帧，约 500 Hz；无未配对帧。最大记录接收时间差为 395 微秒；该差值不是执行延迟。

匹配本地 `/home/xjtx/rl/Cheetah-Software-RL/lcm-types` 的消息指纹：command `277f792b531376cf`、data `4fa4eeb480fbd94f`。

全日志 `tau_est` 与 `tau_ff + kp_joint*(q_des-q) + kd_joint*(qd_des-qd)` 的最大绝对差为 1.60303745e-06 N·m。本地 `common/src/Controllers/LegController.cpp` 的 `updateCommand(SpiCommand*)` 也计算此量，并在 `RobotRunner::finalizeStep()` 中计算后发布。它不是独立实测力矩或电流。

日志中 `tau_ff`、`qd_des`、足端前馈和笛卡尔 PD 字段全部为零；目标位置持续变化，关节 PD 增益有多个阶段。因此零前馈力矩并不表示电机没有输出力矩。

## 约 30 秒的一帧

| 关节 | 目标角 rad | 当前角 rad | 当前速度 rad/s | Kp | Kd | 软件力矩估计 N·m |
|---|---:|---:|---:|---:|---:|---:|
| FR_abad | -0.051166 | -0.073434 | -0.520147 | 17 | 0.4 | 0.586615 |
| FR_hip | -0.973462 | -0.986697 | -0.212454 | 17 | 0.4 | 0.309982 |
| FR_knee | 1.398168 | 1.779365 | -0.007326 | 17 | 0.4 | -6.477416 |
| FL_abad | -0.016923 | -0.081063 | -0.036631 | 17 | 0.4 | 1.105039 |
| FL_hip | -0.939349 | -0.888276 | -0.021978 | 17 | 0.4 | -0.859445 |
| FL_knee | 1.719195 | 1.841927 | -0.021978 | 17 | 0.4 | -2.077651 |
| RR_abad | 0.071423 | 0.094416 | -0.695971 | 17 | 0.4 | -0.112494 |
| RR_hip | -0.877787 | -0.905422 | 0.227106 | 17 | 0.4 | 0.378956 |
| RR_knee | 1.733059 | 1.848412 | 0.065933 | 17 | 0.4 | -1.987368 |
| RL_abad | 0.012207 | 0.038339 | 0.710623 | 17 | 0.4 | -0.728488 |
| RL_hip | -0.831781 | -0.839046 | 0.271063 | 17 | 0.4 | 0.015068 |
| RL_knee | 1.387353 | 1.759910 | -0.051283 | 17 | 0.4 | -6.312961 |

关节索引按本地控制器约定解释为 FR(前右)、FL(前左)、RR(后右)、RL(后左)，每条腿依次 abad(髋外展)、hip(大腿)、knee(小腿)。消息本身没有关节名称，因此腿序仍需与实际运行项目确认。

## 输出与限制

- `joint_targets_feedback.csv`：372696 行关节数据，包含时间、关节索引、位置/速度目标和反馈、PD 增益、前馈力矩、软件力矩估计及重建值。
- `summary.json`：全部通道数量、全部解码字段范围、增益分布、类型指纹和配对验证。
- `joint_position_tracking.png`：每 10 帧取一帧绘制 12 个关节的位置目标和反馈；CSV 保留全量数据。

本日志没有原始策略 action、遥控运动命令、独立电流反馈或底层 CAN 命令通道。目标位置是控制器输出的关节空间目标；无法仅凭它确认原始网络输出、后处理过程或最终电机驱动器收到的值。

复现：

```bash
python3 legged_gym/real-robot-lcm/analyze_joint_log.py legged_gym/real-robot-lcm/record.lcm.00
```

后续源码核对：见 [joint_order_coordinate_audit.md](joint_order_coordinate_audit.md)，包含实际加载验证的 HIMLoco 顺序、实机分支证据、零位与方向转换，以及录制版本一致性限制。

录制分支已由用户更正为 origin/local-deploy-nav-integration，已重新核对并更新坐标报告；运动段 17/0.4 增益与该分支一致，原 HIMLoco-deploy 20/0.5 对照不适用于此次录制。

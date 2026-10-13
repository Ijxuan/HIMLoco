# 转向命令字段核查（2026-10-13）

对象：legged_gym/real-robot-lcm/record.lcm.00。

结论：消息结构确有候选遥控速度字段 state_estimator.vRemoter[3]，但整份记录中其值无效，不能恢复目标转向速度。此前“没有运动命令”的表述应精确为“没有找到有效的运动命令记录”。

## 二进制验证

使用 lcm-gen 为本地 state_estimator_lcmt.lcm 生成 Python 解码器，逐帧解码全部31058条 state_estimator 消息；类型指纹全部通过校验。指纹32a7a58b9685d099，消息长度132字节。

vRemoter 在全部31058帧始终为：

- [0] = 0.0
- [1] = 1.4073838995486278e19
- [2] = 7.49218623593249e31

因此，不能将第三项用作目标yaw rate。字段并非缺失，而是未记录有效命令。这不是仅根据字段名称做出的判断，也不是手工偏移解析失败：生成的类型解码器已验证相同结果。

实际角速度 omegaBody[2] 范围为[-1.1530542374, 0.9863867760] rad/s；omegaWorld[2] 为[-1.1538547277, 0.9859675765] rad/s。以上是全日志范围，包含启动和切换阶段，不应解读为稳定转动速度。可用这些反馈划分实际旋转片段，无法由此还原目标命令或计算准确的命令跟踪误差。

## 发布端证据

核对 /home/xjtx/rl/Cheetah-Software-RL 的 origin/local-deploy-nav-integration，提交 f2c356431cabe54791a7db034bcfbd74174125e1。

common/include/Controllers/StateEstimatorContainer.h:46 将 rRemoterVelociy[i] 复制给 vRemoter[i]；134行定义 setRemoterVelocityResult，用于写入该成员。对该分支全仓 git grep 仅找到此接口定义，未找到调用。

已保存的该分支 FSM_State_RLJointPD.cpp:131-133 从 rcCommand->v_des[0:2] 和 omega_des[2] 填入 command_，随后进行死区处理；gamepad分支另有缩放与死区。该函数中没有将最终 command_ 写入 rRemoterVelociy 的代码。

据此推断字段很可能没有被有效赋值（可能发布了未初始化值）；由于缺少录制二进制的构建身份，不能断言录制程序与源码逐字一致。无论原因如何，记录中的数值本身已经不能作为转向命令使用。

## 全部通道

| 通道 | 消息数 | 内容 |
|---|---:|---|
| microstrain | 62118 | IMU姿态、角速度、加速度等 |
| leg_control_command | 31058 | 关节/足端控制目标和增益 |
| leg_control_data | 31058 | 关节与足端反馈、软件PD力矩估计 |
| state_estimator | 31058 | 状态估计和上述无效vRemoter |
| global_to_robot | 31058 | localization_lcmt机体位姿/运动字段 |
| main_cheetah_visualization | 3720 | 关节角、位置、四元数、颜色 |

未发现单独遥控/导航速度命令通道。可视化、定位和IMU类型也没有另一个目标转向速度字段。

后续如需记录策略实际输入的命令，应发布死区、缩放等处理后的 command_ 三项，并核对策略使用及LCM发布的先后时序。此次只做离线核查，未修改部署代码。

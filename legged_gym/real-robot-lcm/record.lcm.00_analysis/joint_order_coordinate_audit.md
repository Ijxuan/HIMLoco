# LCM 腿序、关节顺序和角度坐标核对

结论：依据用户指定实机项目的 local-deploy-nav-integration 分支，LCM 的 q/q_des 是已经处理零位偏置和方向符号的关节坐标，单位 rad；不能再次减电机零位。LCM 顺序 FR、FL、RR、RL，当前 HIMLoco 顺序 FL、FR、RL、RR。

## 核对版本和证据

实机源码：`/home/xjtx/rl/Cheetah-Software-RL`，分支 `origin/local-deploy-nav-integration`，提交 `f2c356431cabe54791a7db034bcfbd74174125e1`。工作树当前是 master，使用 git show 读取分支，没有切换分支或修改部署源码。

当前 HIMLoco 的 mini_cheetah.urdf 已在 gym4 中通过 CPU PhysX 加载，并直接调用 get_asset_dof_names；结果保存到 himloco_dof_names_verified.json。URDF 文件中的 XML 声明顺序不是 Isaac Gym 返回的 DOF 顺序，不能用声明顺序或配置字典的排列代替运行时 DOF 顺序。

## 12 关节映射（零基索引）

| LCM 索引 | 腿 | 关节 | HIMLoco 索引 | HIMLoco 名称 |
|---:|---|---|---:|---|
| 0 | 前右 FR | 髋外展 (abad) | 3 | FR_hip_joint |
| 1 | 前右 FR | 大腿 (hip) | 4 | FR_thigh_joint |
| 2 | 前右 FR | 小腿 (knee) | 5 | FR_calf_joint |
| 3 | 前左 FL | 髋外展 (abad) | 0 | FL_hip_joint |
| 4 | 前左 FL | 大腿 (hip) | 1 | FL_thigh_joint |
| 5 | 前左 FL | 小腿 (knee) | 2 | FL_calf_joint |
| 6 | 后右 RR | 髋外展 (abad) | 9 | RR_hip_joint |
| 7 | 后右 RR | 大腿 (hip) | 10 | RR_thigh_joint |
| 8 | 后右 RR | 小腿 (knee) | 11 | RR_calf_joint |
| 9 | 后左 RL | 髋外展 (abad) | 6 | RL_hip_joint |
| 10 | 后左 RL | 大腿 (hip) | 7 | RL_thigh_joint |
| 11 | 后左 RL | 小腿 (knee) | 8 | RL_calf_joint |

特别注意命名：Cheetah 的 abad 对应 URDF 的 hip_joint；Cheetah 的 hip 对应 thigh_joint；Cheetah 的 knee 对应 calf_joint。

LCM 转当前 HIMLoco 顺序：

```python
index = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
q_him = q_lcm[index]
q_des_him = q_des_lcm[index]
```

q、qd、q_des、qd_des、tau_ff、tau_est 和关节增益均需保持同一换序。此映射用于与当前 HIMLoco 对齐。录制分支的网络输入和输出直接使用实机顺序 FR、FL、RR、RL，没有 HIMLoco-deploy 分支中的换序步骤；这是已导出网络的部署接口顺序，不单凭此推断原训练工程内部顺序。

## 反馈链路

1. robot/src/rt/rt_ethercat.cpp: motor_id_to_joint_idx 将电机 ID 1/2/3 映射为 abad/hip/knee；channel_to_leg_idx 将 CAN1/2/3/4 映射为 FR/FL/RR/RL。
2. CAN 解包后立即执行 motor_pos_to_robot_q，再写入 g_latest_board_data.q。
3. HardwareBridge::runSpi 将转换后的 q 拷贝为 _spiData.q_abad/q_hip/q_knee。
4. LegController::updateData(SpiData*) 拷贝成 datas[leg].q。
5. LegController::setLcm 按 idx=leg*3+axis 发布 q，并按相同顺序发布 q_des。Quadruped::getHipLocation 的前后及左右坐标交叉验证了腿序。

反馈转换（s 为方向符号，o 为偏置）：

```text
q_joint = (q_motor_feedback - o) * s
```

| 实机腿 | abad s/o | 大腿 hip s/o | 小腿 knee s/o |
|---|---|---|---|
| FR | -1 / 0 | -1 / +1.4907963 | -1 / +2.52 |
| FL | -1 / 0 | +1 / -1.4907963 | +1 / -2.52 |
| RR | +1 / 0 | -1 / -1.4907963 | -1 / +2.52 |
| RL | +1 / 0 | +1 / +1.4907963 | +1 / -2.52 |

偏置单位 rad；大腿偏置由 ±(pi/2 - 0.08) 定义。当前 EtherCAT 转换里 s 都是 ±1，没有额外齿比换算步骤；不能自行再除减速比，也不能仅由这里判断驱动器反馈代表电机内部哪一级轴。

## 目标链路

local-deploy-nav-integration 的 FSM_State_RLJointPD 直接把 datas[0..3].q/qd 拼成网络输入，网络输出也直接以 leg*3+joint 写入 commands[leg].qDes。实际网络接口顺序为 FR、FL、RR、RL，各腿 abad/hip/knee，无策略与实机换序步骤。

目标角：q_des = 0.1 * policy.forward(actorObs_) + q_init；q_init 为每腿 [0, -0.9, 1.8]。这里 q_init 是目标角基准姿态，不是电机零位。所有关节 Kp/Kd=17/0.4；control_dt_=0.01 秒，推理后的指令在低层循环间保持。代码加载 actor_model/actor_5000.txt、estimator_5000.txt 及 mean5000.csv/var5000.csv，但日志本身不提供实际模型文件身份。

setLcm 直接发布关节目标。下发 CAN 前，robot_q_to_motor_pos 才转成电机协议位置；当前 ±1 符号下等价于 q_motor_des=q_joint_des*s+o。因此 q_des 也不需要再次减电机零位。

该分支观测由 141 维原始特征拼接后按 mean/var 归一化并裁剪，关节角直接输入 _jointQ；不是当前 HIMLoco 的 45 维单帧/270 维历史观测。不能把 HIMLoco 的减默认姿态规则直接套到该分支网络输入。

## 电机零位和训练默认姿态不是同一个量

分析目标跟踪时，直接比较 q_des 与 q（必要时统一换序）。输入当前 HIMLoco 的关节位置观测时，还需按训练代码计算 q_him-default_dof_pos；这一步减的是默认站姿而非电机零位。默认站姿按训练顺序为：

```text
FL: [+0.1, -0.8, +1.62]
FR: [-0.1, -0.8, +1.62]
RL: [+0.1, -0.8, +1.62]
RR: [-0.1, -0.8, +1.62]
```

## 与录制版本的一致性

用户更正录制运行分支为 origin/local-deploy-nav-integration。已按此分支重新读取全部相关链路。与之前检查的 HIMLoco-deploy 相比：LCM schema、LegController、RobotRunner、Quadruped 未改变；EtherCAT 仅有空行差异，HardwareBridge 的差异是导航速度命令接收处理，关节角转换/发布未改变。因此先前导出的 CSV、腿名和角度坐标解释仍适用。

录制运动阶段 Kp/Kd=17/0.4 与此分支 RLJointPD 配置匹配；此前 20/0.5 不一致的结论源于核对了另一个分支，应撤回。日志仍不包含 commit、实际模型文件标识或原始 CAN 反馈，故无法独立证明录制时二进制/模型恰为此分支当前提交。核对用分支源码快照保存于 source_evidence/local-deploy-nav-integration；上次 HIMLoco-deploy 快照仅为历史核对证据。

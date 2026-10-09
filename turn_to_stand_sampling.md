# Mini Cheetah 原地旋转与转后站立采样

2026-10-09 更新：Mini Cheetah 每次重采样用同一个均匀随机数选择互斥场景。

| 场景 | 采样概率 | 命令与时长 | 到期行为 |
|---|---:|---|---|
| 普通命令 | 80% | 沿用普通命令分布，10 s | 重新采样 |
| 持续原地旋转 | 10% | vx=vy=0，wz 均匀采样于 [-1,1] rad/s，保持 10 s | 重新采样 |
| 短旋转后站立 | 10% | vx=vy=0，wz 均匀采样于 [-1,1] rad/s，旋转 1–3 s | 必定进入零命令站立 2–5 s，随后重新采样 |

这里的概率是每次采样的场景概率，不是训练时间占比。忽略物理重置截断，长期近似时间占比为普通命令 83.77%、持续旋转 10.47%、短旋转 2.09%、附加站立 3.66%。计算使用期望周期 0.8×10+0.1×10+0.1×(2+3.5)=9.55 s。物理 reset/超时可提前终止当前场景。

Mini Cheetah 配置位于 `legged_gym/legged_gym/envs/minich/minich_config.py`：

| 参数 | 默认值 | 含义 |
|---|---|---|
| `in_place_turn_probability` | 0.1 | 持续旋转场景概率 |
| `in_place_turn_duration_s` | 10.0 | 持续旋转时长 |
| `enable_turn_to_stand` | True | 开启独立场景计时 |
| `turn_to_stand_probability` | 0.1 | 额外短转→站立场景概率 |
| `turn_duration_range_s` | [1.0,3.0] | 短旋转时长 |
| `stand_duration_range_s` | [2.0,5.0] | 短旋转后的站立时长 |
| `ranges.ang_vel_yaw` | [-1.0,1.0] | 两种原地旋转的角速度范围 |
| `resampling_time` | 10.0 | 普通命令持续时间 |

`turn_to_stand_probability` 现在表示独立场景的采样概率，取代旧的“旋转到期后进入站立的条件概率”。两种场景概率之和必须≤1。其他机器人默认关闭调度，额外场景概率默认为0，关闭时保持原有固定重采样行为和随机数序列。普通 heading 控制仍按原逻辑计算角速度；它不是本次原地旋转均匀分布的对象。

阶段定义：NORMAL=0、持续 TURN=1、STAND=2、短 TURN_TO_STAND=3。两种 TURN 均设置 `in_place_turn_buf=True`，并保护角速度不被 heading 控制覆盖。进入 STAND 后前三个命令严格为零。

每环境独立倒计时。dt=0.02 s 时，持续旋转为500步，短旋转为50–150步，站立为100–250步。阶段时长在整数步范围内均匀采样；近整数的浮点边界先修正，避免10 s固定区间被ceil/floor判为空。

阶段切换保持身体、动作、接触、腾空计时和观测历史连续；仅物理reset沿用首帧填满六帧历史。场景采样本身不修改奖励、观测/动作维度或执行顺序。后续已单独将feet_air_time超时部分改为持续结算，见[奖励说明](feet_air_time_reward.md)。固定指令回放和负载测量脚本仍显式关闭调度器。

## 验证与复现

使用gym4执行场景概率统计、均匀速度分布、10 s命令保持、短转必站立、heading保护、状态连续性、物理reset、固定时长浮点边界，以及观测历史和Raibert回归。

旧采样行为和其余奖励比较使用固定提交 `83fec985ffe9a30903d41e5891a1b18b74e25e8d`，避免测试基线随着HEAD移动。feet_air_time使用独立测试验证已授权的新持续结算规则。

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate gym4
TORCH_EXTENSIONS_DIR=/tmp/himloco_torch_extensions \
MPLCONFIGDIR=/tmp/himloco_matplotlib OMP_NUM_THREADS=1 \
python -m unittest discover -s legged_gym/legged_gym/tests -p 'test_*.py' -v

TORCH_EXTENSIONS_DIR=/tmp/himloco_torch_extensions \
MPLCONFIGDIR=/tmp/himloco_matplotlib OMP_NUM_THREADS=1 \
python legged_gym/legged_gym/tests/smoke_command_stages.py \
  --headless --sim_device=cpu --pipeline=cpu --rl_device=cpu
```

CPU PhysX集成检查使用16环境、12 s、旧Sep18 JIT策略，测试中将持续旋转概率设为0、短转→站立概率设为1，以验证连续短转→站立→短转和严格零命令。训练默认概率不受该检查影响。

本次修改场景采样，已有checkpoint不会因此获得新行为；后续训练需要单独评价持续旋转和停止能力。

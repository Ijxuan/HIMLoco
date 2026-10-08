# Mini Cheetah 转向后静止采样

Mini Cheetah 默认启用 `commands.enable_turn_to_stand=True`；其他机器人默认关闭。修改位于 `legged_gym/legged_gym/envs/minich/minich_config.py` 及基类 `commands` 配置。

| 参数 | 默认值 | 含义 |
|---|---|---|
| `in_place_turn_probability` | Mini Cheetah 为 0.1 | 原有采样选中转向分支的概率 |
| `turn_to_stand_probability` | 0.30 | 转向阶段到期时进入静止的条件概率 |
| `turn_duration_range_s` | [1.0, 3.0] | 转向持续时间 |
| `stand_duration_range_s` | [2.0, 5.0] | 静止持续时间 |
| `resampling_time` | 10.0 | 普通命令持续时间 |

普通采样仍使用原有速度分布。原地转向以 `in_place_turn_buf` 判定，不以实际角速度或奖励门槛判定。每环境独立计时；随机阶段步数在 `[ceil(min/dt), floor(max/dt)]` 中均匀采样，包含两端。当前 `dt=0.02 s`，转向为 50–150 步，静止为 100–250 步。转向到期只判断一次概率；未进入静止以及静止到期，都回到原有命令采样。

静止期间前三个命令严格为零，并绕过 heading 控制。命令阶段切换不 reset，不清理身体状态、动作、接触或观测历史。物理 reset 重新采样并沿用首帧填满六帧的现有逻辑。

全部奖励及执行顺序保留：物理仿真→命令更新→奖励→观测。因此切换当步仍按新命令评价刚完成的动作。已有手动命令回放和负载测量脚本显式关闭新采样。

## 验证

2026-10-08 已通过 17 项单元测试，涵盖条件概率、阶段计时、静止 heading 保护、状态连续性、reset、旧采样及随机数兼容性、奖励代码一致性和现有 Raibert/观测历史回归。

已在 gym4 / CPU PhysX 中使用旧 JIT 策略，运行 16 环境、固定种子 2026、12 秒仿真：0 次 reset，16/16 环境完成不中断的转向→静止→转向，静止指令始终为零。测试内将两个采样概率设为 1，未修改训练默认值；关闭噪声与域随机化并使用平地，以验证调度器。此结果不代表重训后停止能力已改善。

从项目根目录复现：

```bash
PYTHONPATH="$PWD/legged_gym:$PWD/rsl_rl" \
LD_LIBRARY_PATH=/opt/miniconda3/envs/gym4/lib \
TORCH_EXTENSIONS_DIR=/tmp/himloco_torch_extensions \
MPLCONFIGDIR=/tmp/himloco_matplotlib \
/opt/miniconda3/envs/gym4/bin/python -m unittest discover \
  -s legged_gym/legged_gym/tests -p 'test_*.py' -v

PYTHONPATH="$PWD/legged_gym:$PWD/rsl_rl" \
LD_LIBRARY_PATH=/opt/miniconda3/envs/gym4/lib \
TORCH_EXTENSIONS_DIR=/tmp/himloco_torch_extensions \
MPLCONFIGDIR=/tmp/himloco_matplotlib \
/opt/miniconda3/envs/gym4/bin/python \
  legged_gym/legged_gym/tests/smoke_command_stages.py \
  --headless --sim_device=cpu --pipeline=cpu --rl_device=cpu
```

单元测试中的旧实现比较以执行时 Git HEAD 为参照。仿真脚本使用本地 `Sep18_18-36-28_smoke_minich/exported/policies/model_1000.jit`，需要该模型存在。详细验证输出保存在 `/tmp/himloco_command_tests.log` 和 `/tmp/himloco_command_smoke.log`。

本次未启动完整训练。旧 checkpoint 不包含此次新场景的训练经历；需要之后重新训练并单独评估停止后的悬脚情况。

# Mini Cheetah 任务

本项目已接入参考项目 `/home/xjtx/rl/legged_gym` 中的标准 12 自由度 Mini Cheetah 模型。

## 资产

- URDF：`legged_gym/resources/robots/mini_cheetah/urdf/mini_cheetah.urdf`
- 网格：`legged_gym/resources/robots/mini_cheetah/urdf/meshes/`
- link 命名包含 `base`、`trunk`、四条腿的 `hip/thigh/calf/foot`。
- 任务名称：`minich`

## 碰撞与资产设置

配置位于 `legged_gym/legged_gym/envs/minich/minich_config.py`：

- `foot_name = "foot"`，使用四个独立固定足部刚体索引。
- `terminate_after_contacts_on = ["trunk"]`，躯干接触时终止 episode。
- `penalize_contacts_on = []`，不启用基类组合碰撞惩罚。
- `collapse_fixed_joints = False`，保留固定足部 link。
- `self_collisions = 1`，关闭 actor 内部自碰撞。
- `flip_visual_attachments = False`，沿用参考 Mini Cheetah URDF 的视觉朝向。

## 控制配置

- 12 个动作输出。
- P 控制器。
- 普通关节：`Kp=17.0, Kd=0.4`。
- 小腿关节：`Kp=34.0, Kd=0.8`。
- `action_scale=0.25`。
- `decimation=4`。
- 初始机身高度：`0.30 m`。
- 默认关节角来自参考 Mini Cheetah 配置。

## 训练配置

Mini Cheetah 复用当前 HIMLoco 的训练栈：

- `HIMActorCritic`
- `HIMPPO`
- 6 帧历史观测
- HIM 速度估计和 16 维 latent
- prototype swap contrastive loss
- privileged critic
- rough trimesh terrain
- domain randomization

运行：

```bash
cd /home/xjtx/rl/HIMLoco/legged_gym/legged_gym/scripts
conda run -n gym4 python -u train.py --task=minich --headless
```

建议先做冒烟测试：

```bash
conda run -n gym4 python -u train.py \
  --task=minich \
  --headless \
  --num_envs=16 \
  --max_iterations=1 \
  --run_name=smoke_minich
```

## 使用 `play.py` 回放训练结果

`play.py` 不会继续训练，而是加载已保存的 Mini Cheetah 策略并在 Isaac Gym 中进行推理回放。其主要流程如下：

1. 通过 `task_registry.get_cfgs(name=args.task)` 获取任务和训练配置。
2. 默认将回放环境限制为最多 50 个，并固定测试设置：地形为 10 行 × 8 列、启用地形 curriculum、最高初始地形等级为 9；关闭观测噪声、摩擦随机化、机器人推扰、disturbance、负载质量随机化和 heading command。若显式传入 `--num_envs`，该参数会在随后创建环境时覆盖默认的 50 个环境设置。
3. 创建环境后，把所有机器人的速度指令设为 `x_vel=1.0`、`y_vel=0.0`、`yaw_vel=0.0`，即期望沿机身 $x$ 方向以 `1.0 m/s` 前进。
4. 设置 `train_cfg.runner.resume=True`，由 `make_alg_runner()` 自动查找并加载模型。
5. 每个仿真步调用策略得到动作，再调用 `env.step()`；默认运行 `10 × max_episode_length` 个仿真步。
6. 默认将本次选中的 `model_*.pt` checkpoint 通过 TorchScript 序列化为 `.jit`，输出到该 checkpoint 所在 run 的 `exported/policies/` 目录；导出的 HIM 策略包含速度估计器和 actor，不是简单复制或重命名 `.pt` 文件。导出后脚本会重新加载 `.jit`，用同一批观测和原策略输出进行数值校验。
7. 同时记录第 0 个机器人第 1 个关节的目标位置、位置、速度、力矩、指令、机身速度和足端接触力。
8. 回放结束后绘制前 100 步状态曲线，并输出一个 episode 周期内的平均奖励。

### 模型查找规则

`play.py` 没有在脚本函数体中手动调用 `get_load_path()`，而是把 `resume=True` 传给任务 runner；runner 再根据训练配置和命令行参数加载模型。模型路径由 `get_load_path()` 决定：

- 根目录为 `legged_gym/logs/rough_minich/`（来自 `MiniChRoughCfgPPO.runner.experiment_name`）。
- 未指定 `--load_run` 时，默认选择按目录名排序后的最后一个训练 run。
- 未指定 `--checkpoint` 或其值为 `-1` 时，默认选择该 run 中按文件名排序后的最后一个 `model_*.pt`。
- 因此，运行前应确认 `legged_gym/logs/rough_minich/` 下存在训练目录和至少一个模型文件；否则会报 `No runs in this directory` 或加载失败。
- 可以用 `--load_run=<目录名>` 指定训练 run，用 `--checkpoint=N` 指定 `model_N.pt`，例如 `--load_run=Aug06_16-16-21_smoke_minich --checkpoint=0`。目录名必须是 `rough_minich` 下的直接子目录。

### 运行时参数

脚本通过 `get_args()` 解析 Isaac Gym 通用参数和项目自定义参数。至少需要设置：

| 参数                       | 是否必须 | 作用                                                                                                   |
| -------------------------- | -------- | ------------------------------------------------------------------------------------------------------ |
| `--task=minich`          | 必须     | 选择 Mini Cheetah 任务；不写时默认任务是`aliengo`。                                                  |
| `--headless`             | 推荐     | 无图形显示运行；服务器或远程环境必须使用。                                                             |
| `--rl_device=cuda:0`     | 通常默认 | 指定 RL 推理设备；默认是`cuda:0`，无 GPU 时可改为 `cpu`，但 Isaac Gym 物理仿真仍需满足其设备配置。 |
| `--num_envs=N`           | 可选     | 覆盖回放环境数量；显式设置后可以超过脚本默认的 50。通常回放只需设置较小值，如`1` 或 `16`。         |
| `--load_run=目录名`      | 可选     | 指定`logs/rough_minich/` 下要加载的训练 run；不指定时加载排序最后的 run。                            |
| `--checkpoint=N`         | 可选     | 指定该 run 下的`model_N.pt`；不指定时加载排序最后的模型。                                            |
| `--experiment_name=名称` | 可选     | 覆盖模型根目录名；Mini Cheetah 正常回放应保持默认`rough_minich`。                                    |
| `--seed=N`               | 可选     | 设置随机种子。                                                                                         |

还可以使用 `--physics_engine`、`--sim_device` 等 Isaac Gym 自带参数；具体可通过 `python play.py --help` 查看。`--resume` 和 `--run_name` 虽然由通用参数解析器提供，但回放时通常不需要设置；脚本会强制令 `train_cfg.runner.resume=True`。

### 启动命令

推荐从脚本目录启动，并使用与训练相同的 `gym4` 环境：

```bash
cd /home/xjtx/rl/HIMLoco/legged_gym/legged_gym/scripts
conda run -n gym4 python -u play.py --task=minich --headless
```

如果当前 shell 已经激活 `gym4` 环境，也可以运行：

```bash
cd /home/xjtx/rl/HIMLoco/legged_gym/legged_gym/scripts
python -u play.py --task=minich --headless
```

常用变体：

```bash
# 指定 RL 设备和并行环境数量
conda run -n gym4 python -u play.py --task=minich --headless --rl_device=cuda:0 --num_envs=16

# 指定训练 run 和 checkpoint
conda run -n gym4 python -u play.py \
  --task=minich \
  --headless \
  --load_run=Aug06_16-16-21_smoke_minich \
  --checkpoint=0

# 显示 Isaac Gym viewer；不要加 --headless
conda run -n gym4 python -u play.py --task=minich
```

### 修改回放速度与可选功能

当前脚本入口固定调用：

```python
play(args, x_vel=1.0, y_vel=0.0, yaw_vel=0.0)
```

因此速度指令不能通过现有命令行参数直接传入。需要修改文件末尾的三个数值，例如将 `x_vel` 改为 `0.5`；横向速度和偏航角速度分别修改 `y_vel`、`yaw_vel`。文件末尾的开关含义如下：

- `EXPORT_POLICY = True`：把本次 `--load_run` / `--checkpoint` 实际选中的 checkpoint 导出为真正的 TorchScript 文件。若加载 `model_4000.pt`，输出为 `<run>/exported/policies/model_4000.jit`；文件包含 HIM estimator 与 actor，输入为 6 帧历史观测，输出为 12 维动作。设为 `False` 可跳过导出和一致性校验。
- `RECORD_FRAMES = False`：设为 `True` 后，每隔 2 步保存 viewer 图像到 `logs/rough_minich/exported/frames/`；应先创建该目录，并使用非 headless viewer，否则可能无法写入图像。
- `MOVE_CAMERA = False`：设为 `True` 后，使相机沿 `[1, 1, 0]` 方向移动。

注意：`play.py` 中 `EXPORT_POLICY`、`RECORD_FRAMES`、`MOVE_CAMERA` 只在 `__main__` 中定义，因此应直接从命令行运行脚本，不要在其他模块中直接调用 `play()` 而忽略这些全局变量。

## `model_1000.jit` 部署适配

部署仓库为 `/home/xjtx/rl-deploy/Cheetah-Software-RL`，当前适配分支为
`HIMLoco-deploy`。`FSM_State_RLJointPD` 及相关 LibTorch runner 已从旧的 48 维
flat-policy 接口切换为 HIMLoco 的 270 维历史观测接口。

部署模型安装在：

```text
/home/xjtx/rl-deploy/Cheetah-Software-RL/rl-checkpoints/model_1000.jit
```

源文件与部署副本 SHA256 均为：

```text
f54bcdc97d274a26adaa341bd2b9073807d64261f9f85a139c71852228e39b0c
```

每个 45 维单帧观测严格按训练端顺序拼接：

```text
0:3    command * [2.0, 2.0, 0.25]
3:6    base angular velocity * 0.25
6:9    projected gravity
9:21   q - q_default
21:33  qd * 0.05
33:45  last action
```

Runner 维护 6 帧历史，顺序为 `[当前帧, t-1, t-2, t-3, t-4, t-5]`；进入
`RL_JOINT_PD` 时清零。新模型不直接输入机身线速度，TorchScript 内部的 HIM
estimator 使用历史观测估计速度和 latent，再由 actor 输出 12 维动作。部署速度
命令范围与训练一致：`vx/vy∈[-1,1] m/s`、`yaw∈[-3.14,3.14] rad/s`。

部署模型的关节顺序必须保持训练环境实际返回的顺序：
`左前腿、右前腿、左后腿、右后腿`，每条腿内部为
`外展关节、大腿关节、小腿关节`。Cheetah 控制器内部顺序为
`右前腿、左前腿、右后腿、左后腿`，因此部署控制器必须在策略输入、策略输出边界
交换左右腿顺序，不能使用恒等映射。该顺序已通过实际启动 Isaac Gym 环境并打印
`env.dof_names` 确认，而不是根据 URDF 文本声明顺序推断。

2026-08-07 已完成验证：

- `model_1000.jit` 实际模型契约为 `[1,270] -> [1,12]`。
- `rapid_rl_policy_config_test` 通过。
- CTest `1/1` 通过。
- `rapid_rl_policy_benchmark 100` 输出均为有限值。
- 单线程 CPU 推理约为 `p50=0.041 ms`、`p95=0.043 ms`、`max=0.061 ms`。
- `rapid_rl_policy_benchmark` 与 `mit_ctrl` 均构建成功。

## 平地基座高度检查

Mini Cheetah 当前训练包含基座高度奖励：`base_height=-1.0`，目标离地高度为
`0.26 m`。奖励函数计算基座相对地面高度与目标值之差的平方，因此对应奖励项为：

$$
r_{height}=-1.0\,(h_{base}-0.26)^2
$$

另外还包含足端抬脚高度惩罚：`foot_clearance=-0.01`，目标值为 `0.04 m`。

已新增 `legged_gym/legged_gym/scripts/play_minich_flat_height.py`。该脚本使用一个
Mini Cheetah、plane 平地、零速度命令和指定的 `model_1000.pt`，关闭噪声及域随机化，
并实时打印世界坐标基座高度、相对地面高度和相对 `0.26 m` 目标的偏差。

启动带界面的平地检查：

```bash
cd /home/xjtx/rl/HIMLoco/legged_gym/legged_gym/scripts
conda run -n gym4 python -u play_minich_flat_height.py --task=minich
```

无界面打印高度：

```bash
conda run -n gym4 python -u play_minich_flat_height.py --task=minich --headless
```

2026-08-07 已完成 0.5 秒冒烟运行，基座相对地面高度从 `0.2924 m` 逐步下降到
`0.2534 m`，经过并接近训练目标 `0.26 m`。

## 已验证结果

2026-08-06 已使用 `gym4` 完成 16 个环境、1 次更新的 GPU 冒烟训练：

- Isaac Gym Preview 4 成功加载。
- GPU PhysX 和 GPU pipeline 成功启用。
- Actor、critic、HIM estimator 成功创建。
- 成功完成 `Learning iteration 0/1`。
- 速度约 `1466 steps/s`。
- 训练日志和模型位于 `legged_gym/logs/rough_minich/`。

## 说明

本次接入使用当前 HIMLoco 基类兼容的 rough-terrain 配置。参考项目中更复杂的 Mini Cheetah 专用奖励扩展、腿长形态随机化和最新训练入口没有直接复制，因为当前 HIMLoco 基类尚未包含对应接口；这样可以先保证 `minich` 任务在当前 HIM-PPO 框架中稳定运行。

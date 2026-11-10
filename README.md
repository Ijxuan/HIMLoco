## Learning-based Locomotion Control from OpenRobotLab
This repository contains learning-based locomotion control research from OpenRobotLab, currently including [Hybrid Internal Model](/projects/himloco/README.md) & [H-Infinity Locomotion Control](/projects/h_infinity/README.md).
## 🔥 News
- [2024-04] Code of HIMLoco is released.
- [2024-04] We release the [paper](https://arxiv.org/abs/2404.14405) of H-Infinity Locomotion Control. Please check the :point_right: [webpage](https://junfeng-long.github.io/HINF/) :point_left: and view our demos! :sparkler:
- [2024-01] HIMLoco is accepted by ICLR 2024.
- [2023-12] We release the [paper](https://arxiv.org/abs/2312.11460) of HIMLoco. Please check the :point_right: [webpage](https://junfeng-long.github.io/HIMLoco/) :point_left: and view our demos! :sparkler:

## 📝 TODO List
- \[x\] Release the training code of HIMLoco, please see `rsl_rl/rsl_rl/algorithms/him_ppo.py`.
- \[ \] Release deployment guidance of HIMLoco.
- \[ \] Release the training code of H-Infinity Locomotion Control.
- \[ \] Release deployment guidance of H-Infinity Locomotion Control.

## 📚 Getting Started

### Installation

We test our codes under the following environment:

- Ubuntu 20.04
- NVIDIA Driver: 525.147.05
- CUDA 12.0
- Python 3.7.16
- PyTorch 1.10.0+cu113
- Isaac Gym: Preview 4

1. Create an environment and install PyTorch:

  - `conda create -n himloco python=3.7.16`
  - `conda activate himloco`
  - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`

2. Install Isaac Gym:
  - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
  - `cd isaacgym/python && pip install -e .`

3. Clone this repository.

  - `git clone https://github.com/OpenRobotLab/HIMLoco.git`
  - `cd HIMLoco`


4. Install HIMLoco.
  - `cd rsl_rl && pip install -e .`
  - `cd ../legged_gym && pip install -e .`

**Note:** Please use legged_gym and rsl_rl provided in this repo, we have modefications on these repos.

### Tutorial

1. Train a policy:

  - `cd legged_gym/legged_gym/scripts`
  - `python train.py`

   Mini Cheetah training (`python train.py --task=minich`) automatically exports
   ONNX every 1000 iterations and after the final partial block to
   `legged_gym/logs/<experiment>/<run>/exported/policies/model_<iteration>.onnx`.
   On normal completion or interruption it also exports `latest.onnx` from the
   current in-memory weights. Training checkpoints remain `model_<iteration>.pt`.
   Other tasks keep their existing TorchScript training exports.

   Install the export dependency in the training environment before starting:

   ```bash
   conda activate gym4
   python -m pip install onnx==1.17.0
   ```

   This export uses PyTorch's ONNX exporter (opset 11) and runs the ONNX checker.
   The complete HIM estimator and Actor accept float32 `obs` of shape `[1,270]`
   and return raw `actions` of shape `[1,12]`. The caller maintains six 45-D
   observation frames, newest first, and applies observation preprocessing,
   action clipping and `q_target = q_default + 0.25 * actions`.
   For Python inference or numerical validation, optionally install
   `python -m pip install onnxruntime==1.19.2 flatbuffers==24.3.25`.
   ONNX Runtime is not required for training export.

2. Play and export the latest policy:
  - `cd legged_gym/legged_gym/scripts`
  - `python play.py`


## Mini Cheetah 历史最佳设置重训

当前 `minich` 默认设置重建自 `Oct23_19-43-17_smoke_minich/model_500.pt`
对应时期的源码。基准模型 SHA256：
`ddbdc2d51dde6f766b5eae1e4895a8ffe09545ae1794c2a685be40429bda3a1e`。

重建依据为提交 `73cd811`（10 月 23 日之前最近的相关提交）和
`legged_gym/logs/policy_eval_20261023_torque_compare/source_snapshot/`。
两处 Mini Cheetah 配置一致，但它们不是完整的训练启动快照：历史源码的
`torques=0` 与实际训练日志不符，因此采用当时记录确认的 **-2e-4**，
依据见该评估目录的 `comparison_context.json`。其余历史值按源码恢复；
不能据此承诺逐位复现或再次得到相同效果，原训练是否接续旧模型也尚未确认。

| 设置 | 当前重训基准 |
|---|---|
| 观测 | 单帧 45、6 帧历史 270、特权观测 238、动作 12 |
| 后摆奖励 | 系数 0.10；下界 5 mm、最佳 10 mm、上界 15 mm；所有环境满足命令门槛时生效 |
| 抬脚峰值奖励 | 系数 0.10；下界 2.5 cm、最佳 5 cm、上界 8 cm；仅前 20% 步态组，且满足命令门槛 |
| 专用长楼梯 | 15% 地形列；0–1 级平地、2–9 级固定 8 cm 高；每侧两级、踏面 1.2 m、首阶距中心 1.2 m |
| 初始地形等级 | 各类均沿用基础配置的 0–5 级随机采样；固定高度专用列不再额外限制到 2 级 |
| 命令 | 关闭专用列仅前进；恢复普通前后/侧向/转向采样，保留历史 20% 低速直行步态组 |
| 功率 / 力矩 | `joint_power=-2e-5`、`torques=-2e-4` |
| 后加功能 | 楼梯时钟关闭，六项踏步奖励、足端竖直力、质心居中系数全为 0 |
| 训练参数 | 4096 环境、每轮 100 步、学习率 1e-3、自适应调节、5 次学习、4 个小批次、最大 200000 轮 |

其他历史奖励系数（含姿态 -1.2、身高 -30、转向抬脚 0.25、腾空超时 -1、
Raibert 落点 -10、髋部镜像 -1、髋部姿态 -1）及模型、控制增益、随机化参数保留历史值。
日志、导出、观测检查和已有性能修复保留；这次恢复的是训练设置与奖励范围，
不是把整个仓库重置到旧提交。后加功能的参数仍保留供显式启用，当前不参与训练。

在项目根目录从头训练（默认 `resume=False`，不加载历史模型参数）：

```bash
/opt/miniconda3/envs/gym4/bin/python legged_gym/legged_gym/scripts/train.py --task=minich --run_name=oct23_retrain
```

如需从历史模型接续训练，另行显式指定：

```bash
/opt/miniconda3/envs/gym4/bin/python legged_gym/legged_gym/scripts/train.py --task=minich --resume --load_run=Oct23_19-43-17_smoke_minich --checkpoint=500 --run_name=oct23_resume
```

接续模式同时加载优化器状态及历史迭代号，与从头训练是不同实验。
新运行目录会保存 `reward_weights.json`，用于核对实际生效系数；
应同时保存本次提交及未提交差异，训练期间不要改源码。本次配置恢复没有启动训练。

## Mini Cheetah 训练楼梯等级与台阶高度

以下对照适用于当前 `minich` 训练配置，等级从 **0** 开始，共 10 个等级。
表中普通楼梯高度取实际生成高度场的相邻台阶高度差的绝对值，单位均为 **cm**；
专用长楼梯列分别给出单级高度和两级顶层相对出生平台的总高度。

| 地形等级 | 上楼梯 `stairs_down`：单级高度 | 下楼梯 `stairs_up`：单级高度 | 专用长楼梯 `long_stairs`：单级高度 | 专用长楼梯：两级总高度 |
|---|---:|---:|---:|---:|
| 0 | 5.0 | 5.0 | 0（平地预备级） | 0 |
| 1 | 6.5 | 6.5 | 0（平地预备级） | 0 |
| 2 | 8.5 | 8.5 | 8 | 16 |
| 3 | 10.0 | 10.0 | 8 | 16 |
| 4 | 12.0 | 12.0 | 8 | 16 |
| 5 | 14.0 | 14.0 | 8 | 16 |
| 6 | 15.5 | 15.5 | 8 | 16 |
| 7 | 17.5 | 17.5 | 8 | 16 |
| 8 | 19.0 | 19.0 | 8 | 16 |
| 9 | 21.0 | 21.0 | 8 | 16 |

### 日常上下楼方向与日志字段

普通楼梯采用金字塔形地形，机器人从中央平台附近出生。按从出生点向外运动的方向：

- **日常上楼梯对应 `stairs_down`**：中央低、四周高，向外运动需要上阶。
- **日常下楼梯对应 `stairs_up`**：中央高、四周低，向外运动需要下阶。
- **专用长楼梯对应 `long_stairs`**：中央平地，沿正、负 X 方向向外均为上阶；
  当前恢复为两级长踏面台阶。

在 TensorBoard（训练指标面板）中查看：

| 日常含义 | 完整指标名 |
|---|---|
| 普通上楼梯平均等级 | `others/terrain_level_stairs_down` |
| 普通下楼梯平均等级 | `others/terrain_level_stairs_up` |
| 专用长楼梯平均等级（包含平地预备级） | `others/terrain_level_long_stairs` |

这些指标是各类环境的**平均等级**，可能为小数，不代表某个固定高度楼梯的通过率，
也不应直接把小数等级插值换算成已掌握的台阶高度。

### 生成规则与尺寸

- 普通楼梯的名义单级高度为 `0.05 + 0.18 * (level / 10)` m。
  高度场使用 0.005 m（5 mm）竖直网格，生成器通过
  `int(step_height / vertical_scale)` 向零截断，因此表中实际高度与名义值可能不同。
  正、负高度对应的单级高度绝对值相同。
- 普通楼梯名义踏面深度为 `0.30` m；当前 0.10 m 水平网格下，
  `int(0.30 / 0.10)` 因浮点数表示得到 2，实际生成踏面深度为 **0.20 m**。
- 专用长楼梯等级 0–1 为平地，等级 2–9 固定为 8 cm。
  每侧两级，踏面深度 **1.20 m**，出生中心到第一阶距离 **1.20 m**；
  高度与深度使用 `round` 转换为网格单元。
- 本表对应当前配置；历史训练与手工回放脚本的楼梯可能不同，应按各自配置核对。

代码来源：[普通与专用楼梯生成逻辑](legged_gym/legged_gym/utils/terrain.py)、
[Mini Cheetah 配置](legged_gym/legged_gym/envs/minich/minich_config.py)、
[基础地形网格与等级配置](legged_gym/legged_gym/envs/base/legged_robot_config.py)。
专用台阶的命令、登顶判据与升级规则见[专用台阶说明](legged_gym/legged_gym/scripts/minich_long_stairs.md)。

## 🔗 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{long2023him,
  title={Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated Robot Response},
  author={Long, Junfeng and Wang, ZiRui and Li, Quanyi and Cao, Liu and Gao, Jiawei and Pang, Jiangmiao},
  booktitle={The Twelfth International Conference on Learning Representations},
  year={2024}
}

@misc{long2024hinf,
  title={Learning H-Infinity Locomotion Control}, 
  author={Junfeng Long and Wenye Yu and Quanyi Li and Zirui Wang and Dahua Lin and Jiangmiao Pang},
  year={2024},
  eprint={2404.14405},
  archivePrefix={arXiv},
}
```

## 📄 License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## 👏 Acknowledgements
- [legged_gym](https://github.com/leggedrobotics/legged_gym): Our codebase is built upon legged_gym.

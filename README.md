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


## Mini Cheetah 训练楼梯等级与台阶高度

以下对照适用于当前 `minich` 训练配置，等级从 **0** 开始，共 10 个等级。
表中普通楼梯高度取实际生成高度场的相邻台阶高度差的绝对值，单位均为 **cm**；
专用长楼梯列分别给出单级高度和两级顶层相对出生平台的总高度。

| 地形等级 | 上楼梯 `stairs_down`：单级高度 | 下楼梯 `stairs_up`：单级高度 | 专用长楼梯 `long_stairs`：单级高度 | 专用长楼梯：两级总高度 |
|---|---:|---:|---:|---:|
| 0 | 5.0 | 5.0 | 0（平地预备级） | 0 |
| 1 | 6.5 | 6.5 | 0（平地预备级） | 0 |
| 2 | 8.5 | 8.5 | 8 | 16 |
| 3 | 10.0 | 10.0 | 10 | 20 |
| 4 | 12.0 | 12.0 | 12 | 24 |
| 5 | 14.0 | 14.0 | 15 | 30 |
| 6 | 15.5 | 15.5 | 18 | 36 |
| 7 | 17.5 | 17.5 | 18 | 36 |
| 8 | 19.0 | 19.0 | 18 | 36 |
| 9 | 21.0 | 21.0 | 18 | 36 |

### 日常上下楼方向与日志字段

普通楼梯采用金字塔形地形，机器人从中央平台附近出生。按从出生点向外运动的方向：

- **日常上楼梯对应 `stairs_down`**：中央低、四周高，向外运动需要上阶。
- **日常下楼梯对应 `stairs_up`**：中央高、四周低，向外运动需要下阶。
- **专用长楼梯对应 `long_stairs`**：中央平地，沿正、负 X 方向向外均为上阶；
  仍沿用长楼梯字段名，但当前实际是两级短踏面台阶。

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
- 专用长楼梯等级 0–1 为平地，等级 2–6 按高度表递增，等级 7–9 保持 18 cm。
  每侧两级，踏面深度 **0.20 m**，出生中心到第一阶距离 **1.20 m**；
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

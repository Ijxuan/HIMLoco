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

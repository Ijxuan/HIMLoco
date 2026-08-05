# HIMLoco 本机训练记录

本文记录在本机 Ubuntu 20.04 + NVIDIA RTX 4070 Ti SUPER 上，根据项目根目录 `README.md` 配置并启动 HIMLoco 训练的实际过程。

## 1. 原 README 的目标环境

项目原 README 给出的测试环境为：

- Ubuntu 20.04
- NVIDIA Driver 525.147.05
- CUDA 12.0
- Python 3.7.16
- PyTorch 1.10.0+cu113
- Isaac Gym Preview 4

本机实际环境为：

- Ubuntu 20.04，内核 `5.15.0-139-generic`
- NVIDIA Driver `535.230.02`
- 驱动报告的 CUDA 版本 `12.2`
- GPU：`NVIDIA GeForce RTX 4070 Ti SUPER`
- GPU compute capability：`8.9`
- 系统 `nvcc` 为 CUDA `10.1`

## 2. Conda 环境选择

本机已有以下环境：

- `isaac`：按要求未修改
- `gym4`：用于本项目训练，允许修改
- `from-unitree-rl`
- `unitree-rl`
- `base`

最终使用 `gym4`，原因是它已经包含 Isaac Gym Preview 4、Python 3.8 和旧版 PyTorch CUDA 栈，且与项目要求接近。

激活环境：

```bash
conda activate gym4
```

## 3. Isaac Gym

本机已有 Isaac Gym Preview 4：

```text
/home/xjtx/rl/isaac_gym/IsaacGym_Preview_4_Package
```

其 Python 包已安装到 `gym4`，版本为 `isaacgym 1.0rc4`。

验证时必须先导入 `isaacgym`，再导入 `torch`，否则 Isaac Gym 会报：

```text
PyTorch was imported before isaacgym modules
```

## 4. 实际配置步骤

在项目根目录执行：

```bash
cd /home/xjtx/rl/HIMLoco
conda activate gym4

cd rsl_rl
pip install -e .

cd ../legged_gym
pip install -e .
```

另外安装训练日志依赖：

```bash
pip install tensorboard==2.11.2 \
  tensorboard-data-server==0.6.1 \
  tensorboard-plugin-wit==1.8.1
```

根目录 `requirements.txt` 中的 CUDA wheel 命令不能直接在本机执行成功，因为它指定的 `torch==1.10.0+cu113` 未被当前 pip 索引解析；同时该版本的 NVRTC 不支持本机 RTX 4070 Ti SUPER 的 compute capability 8.9。

为适配本机 GPU，在允许修改的 `gym4` 环境中使用了：

```text
Python 3.8.20
PyTorch 1.13.1+cu117
torchvision 0.14.1+cu117
torchaudio 0.13.1+cu117
NumPy 1.21.6
setuptools 59.5.0
TensorBoard 2.11.2
Isaac Gym 1.0rc4
```

其中：

- PyTorch 1.13.1+cu117 用于解决 PyTorch 1.10/cu113 在 Ada GPU 上的 `nvrtc: invalid value for --gpu-architecture (-arch)`。
- NumPy 固定为 1.21.6，用于兼容 Isaac Gym Preview 4 中仍使用的 `np.float`。
- setuptools 固定为 59.5.0，用于兼容旧版 `torch.utils.tensorboard` 对 `distutils.version` 的使用。

## 5. 训练验证

项目默认训练任务为 `aliengo`。使用以下命令进行了无渲染短训练：

```bash
cd /home/xjtx/rl/HIMLoco/legged_gym/legged_gym/scripts
conda run -n gym4 python train.py \
  --task=aliengo \
  --headless \
  --num_envs=64 \
  --max_iterations=2 \
  --run_name=smoke_20260805
```

验证结果：

- Isaac Gym Preview 4 成功加载
- GPU PhysX 成功启用
- GPU pipeline 成功启用
- HIM actor、critic 和 estimator 成功创建
- 成功完成第 0、1 次 PPO 更新
- 总步数：`12800`
- 运行时间：约 `3.43s`
- 训练日志：`legged_gym/logs/rough_aliengo/Aug05_14-49-17_smoke_20260805/`
- 已生成 checkpoint：`model_0.pt`、`model_2.pt`
- 已生成 TensorBoard 日志：`events.out.tfevents...`

## 6. 正式训练

确认冒烟测试正常后，可以提高环境数量并移除迭代次数限制：

```bash
cd /home/xjtx/rl/HIMLoco/legged_gym/legged_gym/scripts
conda run -n gym4 python train.py --task=aliengo --headless
```

也可以显式指定环境数量、随机种子和训练轮数：

```bash
conda run -n gym4 python train.py \
  --task=aliengo \
  --headless \
  --num_envs=4096 \
  --seed=1 \
  --max_iterations=200000
```

训练结果默认保存在：

```text
legged_gym/logs/rough_aliengo/<时间>_<run_name>/
```

## 7. 查看和导出策略

项目 README 中的播放命令为：

```bash
cd /home/xjtx/rl/HIMLoco/legged_gym/legged_gym/scripts
conda run -n gym4 python play.py --task=aliengo
```

`play.py` 默认从对应实验目录加载最近一次 checkpoint，并导出 JIT 策略到实验目录下的 `exported/policies`。

## 8. 当前注意事项

- 不要修改 `isaac` 环境；本次配置和依赖调整均在 `gym4` 完成。
- 本项目依赖较老的 Isaac Gym/PyTorch 代码，NumPy、setuptools 和 PyTorch 版本不宜随意升级。
- 本机显卡为 Ada 架构，原 README 的 PyTorch 1.10+cu113 无法直接用于 GPU pipeline；当前使用 PyTorch 1.13.1+cu117 解决该兼容性问题。
- 正式训练会占用较多显存和 GPU 计算资源。首次运行建议保留 `--headless`，并先用较小的 `--num_envs` 和 `--max_iterations` 验证。
- 配置过程中 VS Code/Python 工具创建了工作区下的 `.venv/` 未跟踪目录；它不是训练环境，也未用于本次训练。

# 当前URDF + 9月14日奖励系数

已直接修改标准`envs/minich/minich_config.py`，使用正常`train.py`入口手动启动，不需要独立启动器。

系数依据当前分支9月14日前提交`9a5ad47`重建；当时日志没有保存完整配置，TensorBoard奖励项名称与feet_air_time禁用一致。只恢复系数，保留当前URDF、12维动作、PD增益、命令采样和奖励函数实现。因此不是完整复现9月14日训练。

修改：orientation=-0.2；feet_air_time=0；后来新增的turn_lift_frequency、turn_air_time_overdue、hip_mirror、hip_pos_penalty均为0。tracking_lin_vel=1、tracking_ang_vel=1、base_height=-30、raibert_heuristic=-10及其余历史系数保留。

手动从头训练1000轮示例：

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate gym4
cd /home/xjtx/rl/HIMLoco/legged_gym/legged_gym/scripts
python -u train.py --task=minich --headless --max_iterations=1000 \
  --run_name=sep14_scales_current_urdf
```

不要加`--resume`，除非明确希望接续已有策略。当前默认4096环境；此前GPU PhysX初始化4096环境时遇到`CUBLAS_STATUS_NOT_INITIALIZED`，未开始策略更新。若再次遇到相同错误，可尝试添加`--num_envs=1024`降低资源需求；该CUDA错误的根因尚未确认。

2026-10-16曾准备独立入口并尝试启动，但在初始化阶段退出；根据后续要求不再自动启动。失败日志位于`legged_gym/logs/sep14_reward_retrain_20261016/train.log`。gym4已补装训练导出所需的onnx==1.17.0。

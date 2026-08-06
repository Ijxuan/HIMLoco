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

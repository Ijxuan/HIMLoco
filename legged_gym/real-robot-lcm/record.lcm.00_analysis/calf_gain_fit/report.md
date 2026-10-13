# 小腿 Kp/Kd 拟合测试：未得到可靠的新训练基准

结论：保留当前小腿 Kp=34、Kd=0.8 和已有增益随机化。部署端未修改。曾试写8/1.2作为候选，后续响应与条件敏感性验证不支持，已完整恢复训练配置；最终 minich_config.py 无差异。没有启动新训练。

## 测试规模与方法

使用 gym4 / CPU PhysX / 当前 Mini Cheetah URDF，总计 236 组增益与条件组合，每组回放六个6秒窗口。物理步长5ms，目标按日志零阶保持；反馈插值到物理步时间。日志窗口20–26、26–32、32–38秒用于拟合，38–44、44–50、50–56秒用于验证。每段剔除前0.5秒。髋外展和大腿 PD=17/0.4；关闭随机化、延迟、自动复位。

粗扫 Kp=6,8,10,12,17,24,34 与 Kd=0.15,0.25,0.4,0.6,0.8,1.2；扩展/细扫包含Kp=3,4和Kd=2,3,4。扫描覆盖并对照实机17/0.4和原训练34/0.8。各子目录summary.json和replay.npz记录实际网格与输入输出。

初始关节角/速度、机体高度/四元数/速度来自日志。state_estimator的XY位置字段异常巨大，未使用；自由模式在各环境原点初始化，约束模式按记录世界速度积分XY。估计器高度和姿态并非独立真值，因此结果存在几何/接触条件误差。

## 分阶段结果

| 回放模式 | 拟合段优选 Kp/Kd | 验证角度匹配 RMSE ° | 34/0.8 验证匹配 RMSE ° | 判断 |
|---|---|---:|---:|---|
| 自由机体 | 无有效候选 | — | — | 所有候选出现低机身/明显倾倒 |
| 按实机机体运动 | 8/1.2 | 6.420 | 8.324 | 条件拟合，非自由运动验证 |
| 实机机体运动 + 高度1cm | 6/1.2 | 5.154 | 6.520 | 条件拟合，非自由运动验证 |
| 实机机体运动 - 高度1cm | 8/1.2 | 8.288 | 10.363 | 条件拟合，非自由运动验证 |
| 实机机体运动 + 摩擦0.6 | 8/1.2 | 6.281 | 8.206 | 条件拟合，非自由运动验证 |
| 实机机体与其余关节运动 | 34/1.2 | 7.298 | 7.351 | 条件拟合，非自由运动验证 |

角度匹配RMSE比较 q_sim 与 q_real，是目标回放下的实际响应匹配指标；不是 q_sim 与 q_des 的跟踪RMSE。

自由机体回放时42组候选均出现低机身/倾倒，预录指令无法像实机闭环策略一样适应不断变化的状态。因此这一轮不能用于选增益。机体约束会人为施加外部支撑，valid=true仅代表未触发低高度/姿态阈值，不代表独立稳定行走。

仅约束机体时，8/1.2 的验证匹配RMSE约6.42°，原34/0.8约8.32°，改善约23%；但其前腿响应与实机相关性仅约0.20–0.23，后腿约0.51–0.52，不能说时序响应已匹配。

进一步约束非小腿关节为实机反馈后，优选变为34/1.2，验证匹配RMSE7.298°；原34/0.8为7.351°，只有约0.054°（0.73%）改善。四条腿仍有约5.7–6.3°系统偏差，动态相关性约0.42–0.58。此前低Kp优选依赖其他关节的模拟轨迹，不能视为可靠的小腿动力学辨识。该模式优选位于Kp扫描上边界，因此34/1.2也不应被解释为全局最优。

## 旧策略闭环对照

另使用既有 Sep18_18-36-28_smoke_minich/model_1000.jit，按既有七工况诊断：3秒站立预热、1秒指令渐变、2秒稳定、5秒测量；关闭随机化。该检查评价旧checkpoint兼容性，不评价尚未训练的新策略。

| 工况 | 34/0.8 是否触发低机身/倾倒标志 | 8/1.2 是否触发标志 |
|---|---|---|
| stand | False | True |
| forward_0.5 | False | True |
| forward_1.0 | False | True |
| backward_0.5 | False | True |
| lateral_0.5 | False | True |
| turn_left_1.0 | False | True |
| turn_right_1.0 | False | True |

原组7/7未触发，候选组7/7触发。标志定义为基座高度<0.13m或 projected_gravity_z>-0.7；不能把所有标志一概描述为机器人完全倒地。旧策略失败本身不证明新训练一定失败，但不能将该候选宣称为已验证的稳定训练基准。

## 最终决定

没有采用新增益，也未改变随机化范围；minich_config.py保持小腿34/0.8、髋外展/大腿17/0.4，现有Kp/Kd因子[0.9,1.1]不变。没有修改部署代码、奖励、观测维度或历史。

下一步应先核对实机估计器高度和URDF接触几何（1cm高度变化显著影响拟合误差），再用悬空小幅、多频率的关节目标/反馈辨识执行器动态。当前单次地面闭环数据不足以通过只改Kp/Kd唯一匹配角度幅度、偏差和响应相位；不应将接触模型偏差硬塞进增益。

## 输出和复现

- 各模式scores.csv：完整排名。summary.json：初始化、模型配置、评分和限制。replay.npz：全量仿真/实机轨迹。
- response_diagnostics.json：逐腿响应相关性、偏差和匹配误差。
- validation_trajectories.png：验证段目标、实机反馈和仿真响应对照。

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate gym4
export TORCH_EXTENSIONS_DIR=/tmp/himloco_calf_fit_extensions
export XDG_CACHE_HOME=/tmp/himloco_calf_fit_cache
export OPENBLAS_NUM_THREADS=1
python -u legged_gym/legged_gym/scripts/fit_minich_calf_gains.py --headless --sim_device=cpu --pipeline=cpu --rl_device=cpu --root_mode=logged --noncalf_mode=logged --kp_grid=4,6,8,10,12,17,24,34 --kd_grid=0.15,0.25,0.4,0.6,0.8,1.2,2 --output=/tmp/calf_fit_reproduce
python -u legged_gym/real-robot-lcm/check_gain_policy.py --calf_kp=34 --calf_kd=0.8 --headless --sim_device=cpu --pipeline=cpu --rl_device=cpu --sample_s=5 --output=/tmp/calf_policy_check
```

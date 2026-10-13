# Mini Cheetah 质量参数检查（2026-10-13）

右前/右后腿以对应左腿为镜像基准：hip质心Y从+0.036改为−0.036m，thigh从+0.016改为−0.016m。反射矩阵S=diag(1,−1,1)，质心c_R=S c_L，惯性张量I_R=S I_L S^T。因此右腿hip/thigh的ixy、iyz反号，ixx、iyy、izz、ixz保持原值。对应惯性坐标系rpy均为0。此修正保证左右镜像一致，不代表已用CAD重新标定绝对质量参数。

已验证：全部8对左右对应腿部link（前后×hip/thigh/calf/foot）的质量、质心和惯性张量镜像一致；腿部张量正定；默认姿态可视化中的质心镜像误差由hip 72mm / thigh 32mm降至浮点数零。Isaac Gym CPU PhysX导入并调用训练使用的recomputeInertia=True后，读回8个hip/thigh质心与修正值一致。

## 尚未解决的原始惯量问题

URDF中18个具有inertial的link的矩阵都正定，但其中5个不满足主惯量三角不等式。正定并不足以保证物理可实现。

| link | 排序后的主惯量（kg·m²） | 最大值超过其余两项之和 |
|---|---|---:|
| base | 0.012958, 0.049139, 0.416883 | 0.354786 |
| 四个thigh，每个 | 0.00137463, 0.00603384, 0.00773252 | 0.0003240453 |

机身Iyy=0.416883明显可疑，但不能据此认定应改为0.0416883。大腿的问题在考虑非对角项后体现，仅检查ixx/iyy/izz会漏报。这些惯量大小保留原值，等待CAD或可信原始质量属性；不通过任意缩放、清零耦合项或投影来假定正确物理模型。

上述为URDF输入参数审查；仿真器导入或recomputeInertia的处理不构成物理参数正确性的证明。现有checkpoint是在旧模型上训练的，本次没有重训或重新评测，不代表已证明转动性能改善。已运行的训练进程也不会因磁盘URDF改动自动更新其模型。

验证命令（gym4）：

```bash
python3 legged_gym/legged_gym/tests/test_minich_inertial_symmetry.py
python3 legged_gym/legged_gym/scripts/visualize_minich_com.py --show
```

完整数值审计、PhysX读回及修正后图片位于 `legged_gym/logs/minich_com_corrected_20261013/`。回归测试只覆盖镜像与正定性；完整三角不等式审计明确报告上述5个未解决项。

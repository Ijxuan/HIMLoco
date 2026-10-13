# Mini Cheetah原地旋转跑分

`score_minich_turn.py`既能离线重算已有诊断数据，也能直接加载一个或多个`.pt`策略测试。评价独立于训练奖励，不修改环境奖励、训练配置或原播放脚本。每个模型在相同机器人索引2测试；物理reset建立初始状态，命令切换保留历史，每控制步只滚动一次观测。

用户已选定：必须交替抬脚，贴地旋转要扣分；零命令后1秒内稳定停止，之后角速度≤0.02 rad/s、额外累计转角≤3°。其他数值为初稿，可编辑`minich_turn_score_standards.json`讨论调整。默认100分=旋转40+足端30+停止30，摔倒场景总分归零；保留各项分数、原始指标、扣分原因和checkpoint哈希。

## 满分初稿

| 类别 | 满分条件 | 类内权重 |
|---|---|---|
| 旋转 | 稳态角速度RMSE/max(命令绝对值,0.25)≤8%；最大水平漂移≤3 cm | 跟踪70%，漂移30% |
| 足端：超时与滑动 | 每足连续无接触≤0.30 s；支撑足水平速度均值≤0.005 m/s、P95≤0.02 m/s | 占足端分50%；其中超时40%，均值30%，P95 30% |
| 足端：正常交替抬脚 | ≥80%有效落脚的腾空时间在0.20–0.30 s且抬高≥1 cm；每个4秒滑动窗口内每足有效落脚≥0.5次/秒；相邻落脚组无共同足的比例≥80%；旋转中至少两足支撑的时间≥98% | 占足端分50%；四项分别35%、30%、20%、15% |
| 停止 | 1 s内进入并保持四足接触、角速度绝对值≤0.02 rad/s、平移速度≤0.01 m/s；切停1 s后的角速度RMSE≤0.02、累计绝对转角≤3°、平均平移速度≤0.01 m/s、四足接触时间≥98% | 稳定时间30%，角速度30%，转角20%，平移10%，接触10% |

每个指标在JSON中的`[满分阈值,零分阈值]`之间线性扣分，越界截到0–100%。所有子项都达标才能满分；总分不是合格判定，仍需检查子项和最差场景。

- 旋转稳态舍去前2 s；短旋转最多舍去其一半时长。漂移、足端指标统计整个旋转阶段。
- 正常落脚仅计旋转期间开始的腾空；各足腾空时间包含原环境的落脚计时步。0.04 s内相邻落脚合并成一组，交替检查不强制某一对角步态。
- 接触阈值1 N；滑动统计仅在垂直支撑力>5 N时进行。滑动是足刚体中心的世界水平速度代理，未测接触点瞬时速度。
- 抬脚高度以平地足中心2 cm为基准，最低足底离地高度1 cm（含等号），对应足中心世界高度≥3 cm。不支持直接给楼梯/坡地打分。
- 稳定停止要求至少0.5 s观测，并保持到测量结束。始终不稳定则记为“未稳定”，稳定时间项0分。
- 额外转角是切停1 s以后`∫|wz|dt`，左右摆动不会抵消。另保存从切停瞬间开始的累计转角，供评估制动过程。
- 无抬脚不会满分；少抬脚但无超时、无滑动仍可得部分足端分。`require_steps=false`可切换到允许低速少抬脚的对照标准。
- 未消除滑动与停转问题时，不应把“长期悬脚消失”视为整体满分。统计跨阶段时，最长无接触可能被测量窗口截断；JSON保留窗口末悬空标记。

## 执行

从仓库根目录运行，默认CPU、headless：

```bash
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate gym4
export TORCH_EXTENSIONS_DIR=/tmp/himloco_torch_extensions
export MPLCONFIGDIR=/tmp/himloco_matplotlib
export OMP_NUM_THREADS=1
python legged_gym/legged_gym/scripts/score_minich_turn.py \
  --policy legged_gym/logs/rough_minich/Oct09_17-33-13_smoke_minich/model_1000.pt \
  --policy legged_gym/logs/rough_minich/Oct10_20-01-28_smoke_minich/model_220.pt \
  --yaw-rates 0.25 -0.25 1 -1 --seeds 1 2 3 \
  --standards legged_gym/legged_gym/scripts/minich_turn_score_standards.json \
  --output legged_gym/logs/turn_scores_comparison
```

每次试验默认站立3 s→旋转10 s→站立6 s。可用`--turn-seconds 2`检查短旋转转停；`--mode original`保留原回放启用的随机因素，`controlled`关闭域随机化。默认禁用观测噪声。不重复写已有实时采样目录，应选择新的输出目录。

离线重算只需NumPy，不导入Isaac Gym：

```bash
python legged_gym/legged_gym/scripts/score_minich_turn.py \
  --capture legged_gym/logs/turn_hang_check_20261010/controlled_previous_positive \
  --capture legged_gym/logs/turn_hang_check_20261010/controlled_positive \
  --standards legged_gym/legged_gym/scripts/minich_turn_score_standards.json \
  --output legged_gym/logs/turn_scores_offline
```

输出`scores.md`和`scores.json`；实时测试还保存每场景接触力、足端速度/高度、姿态、奖励原始数据及运行日志。比较必须使用相同目标、阶段时长、随机种子、环境配置、随机因子和观测时序；JSON记录测试矩阵是否一致，均分和最差场景分都保留。场景种子不代表独立训练种子。

## 验证

独立评分测试覆盖正常交替步态满分、贴地旋转扣分、长期悬脚、支撑足滑动、停止振荡不能抵消转角、短旋转、摔倒/非有限数据、稳定停止定义及异常标准；另有一次真实CPU PhysX checkpoint→采样→跑分的端到端验证。

已有正反向±0.25受控轨迹的初稿分数：上一策略51.4/55.9；最新策略56.4/57.1。两者都远未达到满分。结果见`legged_gym/logs/turn_scores_20261010_draft/`。

## 最低抬脚验收（lift-minimum-v2）

每次完整离地—落地过程中，足底最高离地间隙≥0.01m才计为有效抬脚。地面为z=0、足球半径0.02m，因此使用足中心高度减0.02m；仅适用平地。频率和相邻落脚组只统计达到这个高度的事件，微小离地不计数。

以4秒滑动窗口检查所有足：每只脚至少2次有效落地，即≥0.5Hz。短于4秒的转动按整个转动时长计算频率；旋转前开始或结束时尚未落地的动作不计完整事件。频率评分取所有窗口、所有脚中的最低频率，避免其他脚或前半段密集抬脚掩盖某只脚长期不抬。

新增`minimum_lift_pass`独立标记最低抬脚标准是否通过（摔倒或无效数据不能通过），在scores.md显示；该标记不等于整体转向合格。正常腾空时间、交替、支撑及其他评分项继续单独评价。诊断保存每足有效次数、全段平均频率及最差窗口频率。1cm/0.5Hz是验收标准，不是对训练奖励的修改。

验证覆盖1cm和0.5Hz恰好达标、低于高度不计频率、单足不抬及先密集抬脚后停止不能蒙混通过。

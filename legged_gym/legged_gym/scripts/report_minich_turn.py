"""Generate reproducible metrics and plots from compare_minich_turn.py captures."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def metrics(meta, data, robot):
    dt = meta["dt"]
    s = data["states"][:, robot]
    c = {name: i for i, name in enumerate(meta["columns"])}
    assert len(s) == round(sum(meta["schedule_s"]) / dt), "Unexpected samples"
    assert len(data["torques"]) == len(s) * 4, "Unexpected physics samples"
    assert np.isfinite(s).all() and np.isfinite(data["torques"]).all()
    yaw = np.unwrap(np.deg2rad(s[:, c["yaw_deg"]]))
    start = round(meta["schedule_s"][0] / dt)
    end = start + round(meta["schedule_s"][2] / dt)
    out = {}
    for phase, lo, hi, command in (("turn", start, end, .2),
                                    ("turn_steady", start + round(2 / dt), end, .2),
                                    ("stand_after", end, len(s), 0),
                                    ("stand_after_steady", end + round(2 / dt), len(s), 0)):
        a = s[lo:hi]
        xy = a[:, [c["x"], c["y"]]]
        xy_start = s[lo-1, [c["x"], c["y"]]]
        rp = a[:, [c["roll_deg"], c["pitch_deg"]]]
        wz = a[:, c["body_wz"]]
        torque = data["torques"][lo*4:hi*4, robot]
        velocity = data["joint_velocities"][lo*4:hi*4, robot]
        contacts = a[:, [c["foot_fz_" + str(i)] for i in range(4)]] > 5
        feet_speed = a[:, [c["foot_xy_speed_" + str(i)] for i in range(4)]]
        power = np.abs(torque * velocity)
        out[phase] = {
            "mean_yaw_rad_s": float(wz.mean()),
            "yaw_rmse_rad_s": float(np.sqrt(np.mean((wz-command)**2))),
            "yaw_std_rad_s": float(wz.std()),
            "yaw_change_deg": float(np.rad2deg(yaw[hi-1]-yaw[lo-1])),
            "mean_xy_speed_m_s": float(np.linalg.norm(a[:, [c["body_vx"], c["body_vy"]]], axis=1).mean()),
            "endpoint_drift_m": float(np.linalg.norm(xy[-1]-xy_start)),
            "max_drift_m": float(np.linalg.norm(xy-xy_start, axis=1).max()),
            "path_length_m": float(np.linalg.norm(np.diff(np.vstack([xy_start, xy]), axis=0), axis=1).sum()),
            "height_mean_m": float(a[:, c["z"]].mean()),
            "height_std_m": float(a[:, c["z"]].std()),
            "height_min_m": float(a[:, c["z"]].min()),
            "roll_rms_deg": float(np.sqrt(np.mean(rp[:, 0]**2))),
            "pitch_rms_deg": float(np.sqrt(np.mean(rp[:, 1]**2))),
            "tilt_rms_deg": float(np.sqrt(np.mean(np.sum(rp**2, axis=1)))),
            "tilt_peak_deg": float(np.sqrt(np.sum(rp**2, axis=1)).max()),
            "trunk_contact_peak_N": float(a[:, c["trunk_contact_N"]].max()),
            "mean_absolute_joint_power_sum_W": float(power.sum(axis=1).mean()),
            "absolute_mechanical_energy_J": float(power.sum() * meta["physics_dt"]),
            "absolute_mechanical_energy_per_yaw_rad_J": float(power.sum() * meta["physics_dt"] / max(abs(yaw[hi-1]-yaw[lo-1]), 1e-9)),
            "worst_joint_torque_rms_Nm": float(np.sqrt(np.mean(torque**2, axis=0)).max()),
            "peak_joint_torque_Nm": float(np.abs(torque).max()),
            "mean_support_feet_count": float(contacts.sum(axis=1).mean()),
            "support_foot_horizontal_speed_m_s": float(feet_speed[contacts].mean()),
            "contact_changes_per_s": float(np.abs(np.diff(contacts.astype(int), axis=0)).sum() / ((hi-lo)*dt)),
            "reward_mean_per_step": float(a[:, c["reward"]].mean()),
            "reward_terms_mean_per_step": dict(zip(meta["reward_names"], data["reward_terms"][lo:hi, robot].mean(axis=0).tolist())),
            "joint_torque_rms_Nm": np.sqrt(np.mean(torque**2, axis=0)).tolist(),
            "joint_torque_peak_Nm": np.abs(torque).max(axis=0).tolist(),
            "joint_power_mean_abs_W": power.mean(axis=0).tolist(),
            "torque_saturation_fraction": float((np.abs(torque) >= .99*np.asarray(meta["torque_limits"])).mean()),
        }
    out["fell"] = bool((s[:, c["z"]] < .13).any() or (s[:, c["gravity_z"]] > -.7).any())
    return out


def main(root):
    captures, all_metrics = {}, {}
    for path in sorted(root.glob("*/metadata.json")):
        meta = json.loads(path.read_text())
        data = dict(np.load(path.parent / "samples.npz"))
        captures[path.parent.name] = (meta, data)
        all_metrics[path.parent.name] = {meta["policies"][i]["label"]: metrics(meta, data, i) for i in (1, 2)}
    assert len(captures) >= 2
    aggregate = {}
    for mode in ("original", "controlled"):
        selected = [key for key in all_metrics if key.startswith(mode + "_")]
        aggregate[mode] = {}
        for label in ("manual_2", "latest"):
            aggregate[mode][label] = {}
            for phase in ("turn", "turn_steady", "stand_after", "stand_after_steady"):
                aggregate[mode][label][phase] = {}
                example = all_metrics[selected[0]][label][phase]
                for key, value in example.items():
                    if not isinstance(value, (int, float)):
                        continue
                    v = np.array([all_metrics[name][label][phase][key] for name in selected])
                    aggregate[mode][label][phase][key] = {"mean": float(v.mean()), "std_across_seeds": float(v.std()),
                                                        "min": float(v.min()), "max": float(v.max()), "n": len(v)}
    (root / "summary.json").write_text(json.dumps({"per_run": all_metrics, "aggregate": aggregate}, indent=2))
    colors = {"manual_2": "#2474b7", "latest": "#d64a3b"}
    labels = {"manual_2": "Previous (Sep18)", "latest": "Latest (Oct09)"}
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for mode in ("original", "controlled"):
        meta, data = captures[mode + "_seed1"]
        c = {v: i for i, v in enumerate(meta["columns"])}
        dt = meta["dt"]
        t = (np.arange(len(data["states"]))+1)*dt
        fig, ax = plt.subplots(3, 2, figsize=(13, 11))
        for i, label in ((1, "manual_2"), (2, "latest")):
            s = data["states"][:, i]
            col = colors[label]
            for axis, values in ((ax[0, 0], s[:, c["body_wz"]]),
                                 (ax[1, 0], s[:, c["z"]]),
                                 (ax[1, 1], np.sqrt(s[:, c["roll_deg"]]**2+s[:, c["pitch_deg"]]**2))):
                axis.plot(t, values, color=col, lw=1.2, label=labels[label])
            yaw = np.rad2deg(np.unwrap(np.deg2rad(s[:, c["yaw_deg"]])))
            ax[0, 1].plot(t, yaw-yaw[149], color=col, label=labels[label])
            xy = s[149:650, [c["x"], c["y"]]] - s[149, [c["x"], c["y"]]]
            ax[2, 0].plot(xy[:, 0], xy[:, 1], color=col, label=labels[label])
            ax[2, 0].scatter(*xy[-1], color=col, marker="x", s=50)
            power = np.abs(data["torques"][:, i]*data["joint_velocities"][:, i]).sum(axis=1).reshape(-1, 4).mean(axis=1)
            # 0.2 s trailing mean for legible power traces.
            smooth = np.convolve(power, np.ones(10)/10, mode="valid")
            ax[2, 1].plot(t[9:], smooth, color=col, label=labels[label])
        command = np.where((t > 3)&(t <= 13), .2, 0)
        ax[0, 0].plot(t, command, "k--", label="Command", lw=1.3)
        ax[0, 1].plot(t, np.clip(t-3, 0, 10)*.2*180/np.pi, "k--", label="Ideal command integral")
        ax[1, 0].axhline(.26, color="black", ls="--", label="Height target")
        titles = (("Yaw-rate tracking", "Yaw angle from start of turn"),
                  ("Base height", "Roll/pitch combined magnitude"),
                  ("XY trajectory during 10 s turn", "Absolute mechanical joint power (12 joints)"))
        units = (("rad/s", "deg"), ("m", "deg"), ("Y displacement (m)", "W"))
        for row in range(3):
            for column in range(2):
                axis = ax[row, column]
                axis.set_title(titles[row][column])
                axis.set_ylabel(units[row][column])
                axis.grid(alpha=.2)
                axis.legend(fontsize=8)
                if (row, column) == (2, 0):
                    axis.set_xlabel("X displacement (m)")
                    axis.set_aspect("equal", adjustable="datalim")
                else:
                    axis.set_xlabel("Simulation time (s)")
                    axis.axvspan(3, 13, alpha=.08, color="gray")
                    axis.set_xlim(0, 19)
        fig.suptitle(f"Mini Cheetah: 0.2 rad/s in-place turn | {mode} | seed 1 | CPU PhysX", fontsize=14)
        fig.tight_layout()
        fig.savefig(root / (mode + "_comparison.png"), dpi=160)
        plt.close(fig)
    lines = ["# Mini Cheetah 0.2 rad/s 原地旋转策略对比", "",
             "运行环境：conda gym4，CPU PhysX，50 Hz 策略，200 Hz 物理步。3 s 站立 → 10 s 旋转 → 6 s 站立。",
             "原脚本、训练配置、奖励与检查点均未修改。独立包装脚本将检查点映射到 CPU 并记录数据。",
             "上一策略按原脚本 -2 选择上一训练目录，而非最新目录中的上一迭代。", ""]
    for p in captures["controlled_seed1"][0]["policies"][1:]:
        lines += [f"- {p['label']}: `{p['path']}`；SHA256 `{p['sha256']}`"]
    lines += ["", "## 结论", "",
              "在当前脚本的 0.2 rad/s 原地旋转任务中，上一策略更接近目标转速、旋转姿态更稳、停止后残余旋转更小。最新策略的优势是旋转时平移更少，但伴随明显欠转，不能视为整体改善。",
              "受控对比中，上一策略稳态达到目标的约 92.9%，最新策略约 61.9%；角速度 RMSE 最新约为上一策略的 3.08 倍。10 s 内分别旋转约 101.7° 与 73.5°，目标为 114.6°。",
              "最新策略 10 s 终点漂移约 3.8 cm，上一策略约 17.4 cm；最新策略最大偏移约 5.9 cm，上一策略约 18.2 cm。应结合轨迹长度和实际转角解释，不能仅用终点位移判断完全原地。",
              "最新策略旋转稳态平均绝对机械功率更低，但转得更慢。稳态每实际旋转 1 rad 的绝对机械能约 43.4 J，上一策略约 34.8 J，因此本次结果不支持其转动能效更好。其最大单关节 RMS 力矩也略高。",
              "停止后的 6 s 中，最新策略反向回转约 10.0°，上一策略约 3.7°；最新策略停止稳态的残余角速度 RMSE 和关节机械功率均更大。",
              "原设置 5 个种子下，最新策略的角速度 RMSE 每次均更大；终点漂移优势仅在 3/5 次出现，因此漂移结论比转速结论更依赖随机条件。",
              "全部 7 次、共 14 条被比较轨迹均未触发跌倒判据，也未出现机身接触力。受控两次曲线数值一致，仅用于确认复现，不代表广泛鲁棒性验证。",
              "若优先满足目标转速和停止稳定性，本次选择上一策略；若优先减小旋转时平移，最新策略值得继续检查，但须解决欠转与停止后的反向漂转。"]
    for mode, title in (("controlled", "关闭域随机化的相同条件对比"), ("original", "原脚本随机设置复测")):
        lines += ["", "## " + title, "", "数值为各种子均值 ± 种子间标准差；不是置信区间。",
                  "旋转稳态取旋转后 2–10 s；停止稳态取停止后 2–6 s。", "",
                  "| 指标 | 上一策略 Sep18 | 最新策略 Oct09 |", "|---|---:|---:|"]
        specs = [("旋转稳态角速度 rad/s", "turn_steady", "mean_yaw_rad_s"),
                 ("旋转稳态角速度 RMSE rad/s", "turn_steady", "yaw_rmse_rad_s"),
                 ("10 s 实际转角 deg（目标 114.59）", "turn", "yaw_change_deg"),
                 ("10 s 旋转终点平移 m", "turn", "endpoint_drift_m"),
                 ("10 s 旋转最大偏移 m", "turn", "max_drift_m"),
                 ("10 s 旋转轨迹长度 m", "turn", "path_length_m"),
                 ("旋转稳态平移速度 m/s", "turn_steady", "mean_xy_speed_m_s"),
                 ("旋转稳态姿态 RMS deg", "turn_steady", "tilt_rms_deg"),
                 ("旋转稳态平均高度 m（目标 0.260）", "turn_steady", "height_mean_m"),
                 ("旋转稳态高度标准差 m", "turn_steady", "height_std_m"),
                 ("旋转稳态关节绝对机械功率总和 W", "turn_steady", "mean_absolute_joint_power_sum_W"),
                 ("旋转稳态每实际 rad 绝对机械能 J", "turn_steady", "absolute_mechanical_energy_per_yaw_rad_J"),
                 ("旋转稳态最重关节 RMS 力矩 Nm", "turn_steady", "worst_joint_torque_rms_Nm"),
                 ("旋转稳态最大关节峰值力矩 Nm", "turn_steady", "peak_joint_torque_Nm"),
                 ("停止稳态角速度 RMSE rad/s", "stand_after_steady", "yaw_rmse_rad_s"),
                 ("停止后 6 s 净转角 deg", "stand_after", "yaw_change_deg"),
                 ("停止后 6 s 平移 m", "stand_after", "endpoint_drift_m"),
                 ("停止稳态姿态 RMS deg", "stand_after_steady", "tilt_rms_deg"),
                 ("停止稳态关节绝对机械功率总和 W", "stand_after_steady", "mean_absolute_joint_power_sum_W")]
        for title, phase, key in specs:
            cells = []
            for label in ("manual_2", "latest"):
                v = aggregate[mode][label][phase][key]
                cells.append(f"{v['mean']:.4f} ± {v['std_across_seeds']:.4f}")
            lines.append(f"| {title} | " + " | ".join(cells) + " |")
    lines += ["", "## 复现", "", "```bash", "source /opt/miniconda3/etc/profile.d/conda.sh", "conda activate gym4",
              "COMPARE_MODE=controlled \\", "COMPARE_OUTPUT=legged_gym/logs/turn_comparison_20261009/controlled_seed1 \\",
              "TORCH_EXTENSIONS_DIR=/tmp/himloco_torch_extensions \\", "MPLCONFIGDIR=/tmp/himloco_matplotlib \\",
              "python -u legged_gym/legged_gym/scripts/compare_minich_turn.py \\",
              "  --headless --sim_device=cpu --pipeline=cpu --rl_device=cpu --seed=1", "```", "",
              "原设置使用 COMPARE_MODE=original 和种子 1–5；受控对比使用种子 1、2。",
              "`samples.npz` 保存 950 个策略步及 3800 个物理步；元数据保存模型哈希、配置、随机因子及列定义。",
              "`summary.json` 包含逐次与聚合指标；PNG 使用 seed 1 完整时间序列。", "",
              "## 测量边界", "",
              "当前 GPU 驱动不可用，DISPLAY=:1 连接失败，未取得 Isaac Gym 界面截图；PNG 是仿真真值曲线。",
              "原设置中的不同机器人带有不同增益、强度、质心和动作延迟样本，不能把单次差异完全归因于策略。受控对比关闭全部 domain_rand 布尔开关。",
              "保留原脚本的观测调用时序：循环前与 env.step 后各更新一次历史。这与训练每步更新一次存在差异，结论针对该回放脚本；不可直接推断全部训练/实机表现。",
              "机械功率为 Σ|τ·qdot|，不是电池功耗；力矩使用每个物理子步的截断 PD 输出和积分前关节速度。",
              "跌倒判据为高度 <0.13 m 或 projected_gravity.z >-0.7；同时独立记录机身接触力。",
              "站立/旋转各阶段使用现有奖励计算；奖励分值仅为辅助诊断，优劣按角速度、漂移、姿态及停止表现评价。"]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))

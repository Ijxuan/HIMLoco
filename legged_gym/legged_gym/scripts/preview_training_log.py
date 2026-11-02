"""Preview console formatting with synthetic data; no training dependencies.

Run from the repository root:
    python3 legged_gym/legged_gym/scripts/preview_training_log.py --width 140
"""

import argparse
import importlib.util
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, help="Preview a fixed terminal width.")
    args = parser.parse_args()
    if args.width is not None and args.width < 1:
        parser.error("--width must be positive")

    # Load only the formatter, avoiding torch / Isaac Gym package imports.
    path = Path(__file__).resolve().parents[3] / "rsl_rl/rsl_rl/utils/console_log.py"
    spec = importlib.util.spec_from_file_location("console_log_preview", path)
    console = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(console)

    rewards = {
        "tracking_lin_vel": 0.7213, "tracking_ang_vel": 0.8124,
        "step_retraction": 0.0362, "swing_peak_height": 0.0241,
        "foot_vertical_force": -0.0098,
        "stair_gait_force": -0.0125, "stair_gait_height": -0.0381,
        "stair_gait_velocity": -0.0047, "stair_gait_pair_deficit": -0.0216,
        "lin_vel_z": -0.0421, "ang_vel_xy": -0.0182,
        "orientation": -0.0314, "dof_acc": -0.0156,
        "joint_power": -0.0082, "base_height": -0.0104,
        "action_rate": -0.0523, "smoothness": -0.0412,
        "torques": -0.0173, "raibert_heuristic": -0.0251,
        "hip_mirror": -0.0132, "hip_pos_penalty": -0.0164,
    }
    lines = [
        "#" * 80, "\033[1mLearning iteration 1234/2000\033[0m", "",
        "Computation: 98304 steps/s (collection: 0.910s, learning 0.090s)",
        "Value function loss: 0.0123", "Surrogate loss: -0.0045",
        "Estimation loss: 0.0081", "Swap loss: 0.6932",
        "Mean action noise std: 0.65", "Mean reward: 18.42",
        "Mean episode length: 856.30",
    ]
    lines.extend(f"Mean episode rew_{key}: {value:.4f}" for key, value in rewards.items())
    lines.extend([
        "Mean episode terrain_level: 3.2100",
        "Mean episode terrain_level_stairs: 2.8400",
        "-" * 80, "Total timesteps: 121405440", "Iteration time: 1.00s",
        "Total time: 1235.00s", "ETA: 766.0s",
    ])
    print("[模拟预览] 全部数值为虚构示例，不创建环境、不加载模型、不启动训练。")
    print(console.compact_training_log("\n".join(lines), columns=args.width))


if __name__ == "__main__":
    main()

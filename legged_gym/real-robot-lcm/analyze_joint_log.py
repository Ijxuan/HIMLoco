#!/usr/bin/env python3
"""Offline LCM joint export; never publishes messages or connects to a robot."""
import argparse
import csv
import json
import math
import struct
from collections import Counter
from pathlib import Path

SCHEMAS = {
    "leg_control_command": ("277f792b531376cf", (
        "tau_ff", "f_ff", "q_des", "qd_des", "p_des", "v_des",
        "kp_cartesian", "kd_cartesian", "kp_joint", "kd_joint")),
    "leg_control_data": ("4fa4eeb480fbd94f", ("q", "qd", "p", "v", "tau_est")),
}
JOINTS = [f"{leg}_{joint}" for leg in ("FR", "FL", "RR", "RL")
          for joint in ("abad", "hip", "knee")]


def events(path):
    with path.open("rb") as stream:
        while True:
            header = stream.read(28)
            if not header:
                return
            if len(header) != 28:
                raise ValueError("Truncated event header")
            magic, number, timestamp, nc, nd = struct.unpack(">Iqqii", header)
            if magic != 0xEDA1DA01 or nc <= 0 or nd < 0:
                raise ValueError(f"Invalid event header at event {number}")
            channel, payload = stream.read(nc), stream.read(nd)
            if len(channel) != nc or len(payload) != nd:
                raise ValueError(f"Truncated event {number}")
            yield number, timestamp, channel.decode(), payload


def decode(channel, payload):
    fingerprint, fields = SCHEMAS[channel]
    if payload[:8].hex() != fingerprint or len(payload) != 8 + 48 * len(fields):
        raise ValueError(f"Schema mismatch for {channel}")
    values = struct.unpack(f">{12 * len(fields)}f", payload[8:])
    if not all(math.isfinite(v) for v in values):
        raise ValueError(f"Non-finite value in {channel}")
    return {field: values[i * 12:(i + 1) * 12] for i, field in enumerate(fields)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.log.parent / (args.log.name + "_analysis")
    out.mkdir(parents=True, exist_ok=True)
    channels, ranges = {}, {}
    gain_counts = Counter()
    pending = None
    paired = 0
    unpaired = 0
    max_dt = 0
    max_residual = 0.0
    square_residual = 0.0
    sample = None
    plot_rows = []
    start = end = None
    with (out / "joint_targets_feedback.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time_s", "command_timestamp_us", "feedback_timestamp_us",
                         "log_arrival_delta_us", "joint_index", "joint",
                         "q_des_rad", "q_rad", "qd_des_rad_s", "qd_rad_s",
                         "kp_joint", "kd_joint", "tau_ff_Nm", "tau_est_Nm",
                         "tau_pd_reconstructed_Nm"])
        for number, timestamp, channel, payload in events(args.log):
            if start is None:
                start = timestamp
            end = timestamp
            info = channels.setdefault(channel, {"count": 0, "first_us": timestamp,
                                                  "last_us": timestamp})
            info["count"] += 1
            info["last_us"] = timestamp
            if channel not in SCHEMAS:
                continue
            decoded = decode(channel, payload)
            for field, values in decoded.items():
                key = f"{channel}.{field}"
                stat = ranges.setdefault(key, {"min": math.inf, "max": -math.inf,
                                              "nonzero": 0, "count": 0})
                stat["min"] = min(stat["min"], min(values))
                stat["max"] = max(stat["max"], max(values))
                stat["nonzero"] += sum(v != 0 for v in values)
                stat["count"] += 12
            if channel == "leg_control_command":
                if pending is not None:
                    unpaired += 1
                pending = (number, timestamp, decoded)
                gain_counts.update(zip(decoded["kp_joint"], decoded["kd_joint"]))
                continue
            # Pair consecutive command/data messages, allowing other channels
            # (e.g. asynchronous IMU traffic) to interleave in the log.
            # These are logger receive times, not motor hardware timestamps.
            if pending is None:
                unpaired += 1
                continue
            command_number, command_time, command = pending
            pending = None
            if number <= command_number or not 0 <= timestamp - command_time <= 2000:
                raise ValueError("Command/data ordering or arrival interval mismatch")
            dt = timestamp - command_time
            max_dt = max(max_dt, dt)
            t = (timestamp - start) / 1e6
            snapshot = []
            for i, joint in enumerate(JOINTS):
                torque = (command["tau_ff"][i]
                          + command["kp_joint"][i] * (command["q_des"][i] - decoded["q"][i])
                          + command["kd_joint"][i] * (command["qd_des"][i] - decoded["qd"][i]))
                residual = torque - decoded["tau_est"][i]
                max_residual = max(max_residual, abs(residual))
                square_residual += residual ** 2
                row = [t, command_time, timestamp, dt, i, joint,
                       command["q_des"][i], decoded["q"][i], command["qd_des"][i],
                       decoded["qd"][i], command["kp_joint"][i], command["kd_joint"][i],
                       command["tau_ff"][i], decoded["tau_est"][i], torque]
                writer.writerow(row)
                snapshot.append(row)
            if sample is None and t >= 30:
                sample = snapshot
            if paired % 10 == 0:
                plot_rows.append((t, command["q_des"], decoded["q"]))
            paired += 1
    if pending is not None:
        unpaired += 1
    for info in channels.values():
        duration = (info["last_us"] - info["first_us"]) / 1e6
        info["duration_s"] = duration
        info["rate_hz"] = (info["count"] - 1) / duration if duration else None
    summary = {"input": str(args.log.resolve()), "duration_s": (end - start) / 1e6,
               "channels": channels, "joint_order_assumption": JOINTS,
               "schema_fingerprints": {k: v[0] for k, v in SCHEMAS.items()},
               "field_ranges": ranges, "paired_cycles": paired,
               "unpaired_messages": unpaired, "max_log_arrival_delta_us": max_dt,
               "tau_est_vs_pd_max_abs_error_Nm": max_residual,
               "tau_est_vs_pd_rmse_Nm": math.sqrt(square_residual / (paired * 12)),
               "gain_counts_joint_samples": [{"kp": kp, "kd": kd, "count": count}
                    for (kp, kd), count in sorted(gain_counts.items())],
               "sample_30s": sample,
               "notes": ["Joint order follows local Cheetah-Software-RL; schema alone does not encode joint names.",
                         "tau_est matches the software PD estimate; it is not independent measured torque/current.",
                         "All command fields are summarized; CSV exports joint-space fields.",
                         "No raw policy action, operator command, current or CAN command channel is present."]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 3, figsize=(15, 10), sharex=True)
        times = [r[0] for r in plot_rows]
        for i, ax in enumerate(axes.flat):
            ax.plot(times, [r[1][i] for r in plot_rows], label="Target", linewidth=0.8)
            ax.plot(times, [r[2][i] for r in plot_rows], label="Feedback", linewidth=0.8)
            ax.set_title(JOINTS[i])
            ax.set_ylabel("Joint angle (rad)")
            ax.grid(alpha=0.25)
        axes[0, 0].legend()
        for ax in axes[-1]:
            ax.set_xlabel("Log time (s)")
        fig.tight_layout()
        fig.savefig(str(out / "joint_position_tracking.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        print("matplotlib unavailable; CSV and summary still exported")
    print(json.dumps({"output": str(out), "paired_cycles": paired,
                      "max_pd_residual_Nm": max_residual,
                      "unpaired_messages": unpaired}, indent=2))


if __name__ == "__main__":
    main()

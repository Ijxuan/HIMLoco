"""Compact presentation of runner logs without changing recorded metrics."""

import shutil


# Keep training health and recently tuned Mini Cheetah rewards easy to scan.
STANDALONE_METRICS = frozenset({
    "Mean reward", "Mean episode length", "Mean action noise std",
    "rew_tracking_lin_vel", "rew_tracking_ang_vel",
    "rew_step_retraction", "rew_swing_peak_height", "rew_foot_vertical_force",
    "rew_stair_gait_force", "rew_stair_gait_height",
    "rew_stair_gait_velocity", "rew_stair_gait_pair_deficit",
})


def compact_training_log(log_string, width=80, *, columns=None):
    """Pack metric lines to terminal width and repeat progress at the bottom."""
    if columns is None:
        columns = shutil.get_terminal_size(fallback=(width, 24)).columns
    lines = []
    row = ""
    progress = ""
    for line in log_string.splitlines():
        item = line.strip()
        if not item:
            continue
        if set(item) in ({"#"}, {"-"}):
            if row:
                lines.append(row)
                row = ""
            lines.append(item[0] * columns)
            continue
        if "Learning iteration " in item:
            progress = item
            lines.append(item)
            continue
        if item.startswith("Mean episode ") and not item.startswith("Mean episode length:"):
            item = item[len("Mean episode "):]
        if item.partition(":")[0] in STANDALONE_METRICS:
            if row:
                lines.append(row)
                row = ""
            lines.append(item)
            continue
        candidate = f"{row} | {item}" if row else item
        if row and len(candidate) > columns:
            lines.append(row)
            row = item
        else:
            row = candidate
    if row:
        lines.append(row)
    if progress:
        lines.append(progress)
    return "\n".join(lines)

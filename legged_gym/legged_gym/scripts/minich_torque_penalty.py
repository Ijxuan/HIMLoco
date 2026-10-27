"""Offline torque reward calibration matching LeggedRobot reward timing."""
import numpy as np


def torque_square_statistics(torques):
    torque = np.asarray(torques, dtype=np.float64)
    if torque.ndim != 2 or not len(torque) or not np.isfinite(torque).all():
        raise ValueError("Require nonempty finite [time, joint] torques")
    raw = np.square(torque).sum(axis=1)
    return {
        "torque_square_sum_mean_Nm2": float(raw.mean()),
        "torque_square_sum_p95_Nm2": float(np.percentile(raw, 95)),
        "torque_square_sum_peak_Nm2": float(raw.max()),
    }


def policy_torque_squares(physics_torques, decimation):
    """Reward uses the final clipped torque of each policy cycle, not its mean."""
    torque = np.asarray(physics_torques, dtype=np.float64)
    if (torque.ndim != 2 or not len(torque) or decimation < 1
            or int(decimation) != decimation or len(torque) % decimation
            or not np.isfinite(torque).all()):
        raise ValueError("Require complete finite physics-step torque cycles")
    return np.square(torque.reshape(-1, int(decimation), torque.shape[-1])[:, -1]).sum(axis=1)


def rescore_torque_scale(raw_torque_square, existing_unclipped_rate,
                        candidate, existing_scale=0., dt=.02):
    raw = np.asarray(raw_torque_square, dtype=np.float64)
    original = np.asarray(existing_unclipped_rate, dtype=np.float64)
    if (raw.ndim != 1 or not len(raw) or raw.shape != original.shape
            or not np.isfinite(raw).all() or not np.isfinite(original).all()
            or (raw < 0).any() or not np.isfinite([candidate, existing_scale, dt]).all()
            or candidate > 0 or dt <= 0):
        raise ValueError("Require matching finite rates, nonnegative squares, negative scale and positive dt")
    baseline = original - existing_scale * raw
    updated = baseline + candidate * raw
    return {
        "scale": float(candidate),
        "mean_penalty_rate": float((-candidate * raw).mean()),
        "p95_penalty_rate": float(np.percentile(-candidate * raw, 95)),
        "mean_penalty_per_policy_step": float((-candidate * raw).mean() * dt),
        "baseline_unclipped_rate_mean": float(baseline.mean()),
        "new_unclipped_rate_mean": float(updated.mean()),
        "baseline_zero_clipping_fraction": float((baseline < 0).mean()),
        "new_zero_clipping_fraction": float((updated < 0).mean()),
    }

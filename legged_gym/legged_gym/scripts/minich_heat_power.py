"""Nominal, uncalibrated equivalent copper loss; no temperature prediction.

Parameters copied from Cheetah-Software-RL/common/include/Dynamics/
MiniCheetah.h and ActuatorModel.h: R=0.173, effective Kt=1.5*0.05,
gear ratios=6/6/9.33. Treat R as an equivalent resistance in this simplified
I^2 R model. Phase/RMS conventions and actual hardware must be calibrated
before interpreting these estimates as measured winding losses.
"""
import numpy as np


def heat_model(joint_names):
    ratios = []
    for name in joint_names:
        if name.endswith(("_hip_joint", "_thigh_joint")):
            ratios.append(6.0)
        elif name.endswith("_calf_joint"):
            ratios.append(9.33)
        else:
            raise ValueError(f"Unknown joint type: {name}")
    coefficients = 0.173 / (0.075 * np.asarray(ratios)) ** 2
    return {
        "name": "nominal_equivalent_copper_loss_v1",
        "calibrated": False,
        "formula": "P_heat_joint = coefficient * joint_torque**2; total = sum(joints)",
        "equivalent_resistance_ohm": 0.173,
        "effective_motor_torque_constant_Nm_per_A": 0.075,
        "gear_ratios": ratios,
        "joint_names": list(joint_names),
        "coefficients_W_per_Nm2": coefficients.tolist(),
        "limitations": "Nominal equivalent-current convention; no phase/RMS calibration, temperature-dependent resistance, iron/driver/gear losses, cooling or temperature prediction. Not battery power.",
    }


def heat_power(torques, coefficients):
    """Last axis is joints. Coefficients can be replaced by calibrated values."""
    torque = np.asarray(torques, dtype=float)
    coefficient = np.asarray(coefficients, dtype=float)
    if (coefficient.ndim != 1 or torque.ndim < 1
            or torque.shape[-1] != len(coefficient)
            or not np.isfinite(torque).all()
            or not np.isfinite(coefficient).all() or (coefficient <= 0).any()):
        raise ValueError("Require finite torques and positive per-joint coefficients")
    return torque ** 2 * coefficient


def whole_body_statistics(torques, velocities, coefficients):
    torque = np.asarray(torques)
    velocity = np.asarray(velocities)
    if torque.ndim != 2 or not len(torque) or velocity.shape != torque.shape or not np.isfinite(velocity).all():
        raise ValueError("Require nonempty finite matching [time, joint] arrays")
    heat = heat_power(torque, coefficients).sum(axis=-1)
    mechanical = np.abs(torque * velocity).sum(axis=-1)
    return {
        "mechanical_abs_mean_W": float(mechanical.mean()),
        "mechanical_abs_peak_W": float(mechanical.max()),
        "estimated_copper_loss_mean_W": float(heat.mean()),
        "estimated_copper_loss_p95_W": float(np.percentile(heat, 95)),
        "estimated_copper_loss_peak_W": float(heat.max()),
    }

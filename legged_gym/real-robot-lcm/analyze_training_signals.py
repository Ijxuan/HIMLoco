#!/usr/bin/env python3
"""Quantify observed closed-loop tracking, without identifying motor latency."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_joint_log import JOINTS, decode, events


def quantiles(x):
    return {name: float(v) for name, v in zip(
        ("min", "p05", "p50", "p95", "p99", "max"),
        np.percentile(x, [0, 5, 50, 95, 99, 100]))}


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def correlation_lags(target, actual, dt, max_ms=150):
    # Positive lag means target at t resembles actual at t + lag.
    # Search negative lags as a diagnostic for feedback confounding.
    m = int(round(max_ms / (dt * 1000)))
    values = []
    for k in range(-m, m + 1):
        if k > 0:
            x, y = target[:-k], actual[k:]
        elif k < 0:
            x, y = target[-k:], actual[:k]
        else:
            x, y = target, actual
        x, y = x - np.mean(x), y - np.mean(y)
        den = np.sqrt(np.sum(x*x) * np.sum(y*y))
        values.append(float(np.sum(x*y) / den) if den > 1e-12 else 0.)
    lags = np.arange(-m, m + 1) * dt * 1000
    best = int(np.argmax(values))
    return {"best_lag_ms": float(lags[best]), "best_correlation": values[best],
            "zero_lag_correlation": values[m],
            "search_boundary_peak": bool(best in (0, 2*m))}, lags, values


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path)
    args = ap.parse_args()
    out = args.log.parent / (args.log.name + "_analysis") / "training_signals"
    out.mkdir(parents=True, exist_ok=True)
    timestamps, commands, feedbacks = [], [], []
    pending = None
    for _, ts, ch, data in events(args.log):
        if ch == "leg_control_command":
            if pending is not None:
                raise ValueError("Missing feedback")
            pending = decode(ch, data)
        elif ch == "leg_control_data":
            if pending is None:
                raise ValueError("Missing command")
            timestamps.append(ts)
            commands.append(pending)
            feedbacks.append(decode(ch, data))
            pending = None
    t = (np.array(timestamps, dtype=np.int64) - timestamps[0]) / 1e6
    c = {f: np.array([r[f] for r in commands]) for f in commands[0]}
    d = {f: np.array([r[f] for r in feedbacks]) for f in feedbacks[0]}
    dt = float(np.median(np.diff(t)))
    active = np.all(np.isclose(c["kp_joint"], 17) & np.isclose(c["kd_joint"], .4), axis=1)
    active_runs = runs(active)
    if not active_runs:
        raise ValueError("No RL gain segment")
    gain_segments = []
    boundaries = np.r_[0, np.flatnonzero(np.any(np.diff(c["kp_joint"], axis=0) != 0, axis=1))+1, len(t)]
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        gain_segments.append({"start_s": float(t[a]), "end_s": float(t[b-1]),
                              "samples": int(b-a), "kp": c["kp_joint"][a].tolist(),
                              "kd": c["kd_joint"][a].tolist()})
    # Exclude 1 s after entry and 1 s before exit for primary tracking metrics.
    trimmed = active.copy()
    margin = int(round(1 / dt))
    for a, b in active_runs:
        trimmed[a:min(b, a+margin)] = False
        trimmed[max(a, b-margin):b] = False
    errors = d["q"] - c["q_des"]
    rows, lag_rows, curves = [], [], []
    for i, joint in enumerate(JOINTS):
        e = errors[trimmed, i]
        q, target, vel = d["q"][trimmed, i], c["q_des"][trimmed, i], d["qd"][trimmed, i]
        row = {"joint": joint, "bias_rad": float(e.mean()),
               "rmse_rad": float(np.sqrt(np.mean(e*e))),
               "mae_rad": float(np.mean(np.abs(e))),
               "p95_abs_error_rad": float(np.percentile(np.abs(e), 95)),
               "max_abs_error_rad": float(np.max(np.abs(e))),
               "centered_rmse_rad": float(np.std(e)),
               "q_min_rad": float(q.min()), "q_max_rad": float(q.max()),
               "target_min_rad": float(target.min()), "target_max_rad": float(target.max()),
               "qd_p99_abs_rad_s": float(np.percentile(np.abs(vel),99)),
               "qd_max_abs_rad_s": float(np.max(np.abs(vel))),
               "software_tau_p99_abs_Nm": float(np.percentile(np.abs(d['tau_est'][trimmed,i]),99))}
        increments = np.abs(np.diff(d['q'][:,i]))
        increments = increments[increments > 1e-7]
        row['q_nonzero_increment_p01_rad'] = float(np.percentile(increments,1))
        row['actual_target_std_ratio'] = float(q.std()/target.std())
        for key in ("bias", "rmse", "p95_abs_error"):
            row[key+"_deg"] = float(np.degrees(row[key+"_rad"]))
        rows.append(row)
        # Six-second non-overlapping windows wholly inside trimmed gain segment.
        # Subsample to ~250 Hz for inexpensive correlation; no interpolation.
        per_joint = []
        for a, b in runs(trimmed):
            width = int(round(6/dt))
            for start in range(a, b-width+1, width):
                sl = slice(start, start+width, 2)
                x,y = c["q_des"][sl,i], d["q"][sl,i]
                metric,lags,vals = correlation_lags(x, y, dt*2)
                metric.update(joint=joint,start_s=float(t[start]),end_s=float(t[start+width-1]),
                              target_std_rad=float(x.std()), actual_std_rad=float(y.std()))
                lag_rows.append(metric)
                per_joint.append(metric)
                curves.append((i, lags, vals))
        usable = [r for r in per_joint if r['target_std_rad'] >= .03 and r['best_correlation'] >= .5 and not r['search_boundary_peak']]
        row["lag_usable_windows"] = len(usable)
        row["apparent_lag_median_ms"] = float(np.median([r['best_lag_ms'] for r in usable])) if usable else None
        row["apparent_lag_min_ms"] = min((r['best_lag_ms'] for r in usable),default=None)
        row["apparent_lag_max_ms"] = max((r['best_lag_ms'] for r in usable),default=None)
    changed = np.any(np.abs(np.diff(c["q_des"],axis=0)) > 1e-7,axis=1)
    updates = np.flatnonzero(changed & active[1:] & active[:-1])+1
    intervals = np.diff(t[updates])
    intervals = intervals[intervals < .1]
    delta_target = np.diff(c["q_des"],axis=0)[updates-1]
    held = np.all(np.diff(d["q"],axis=0)==0,axis=1) & np.all(np.diff(d["qd"],axis=0)==0,axis=1)
    joint_held = ((np.diff(d["q"],axis=0)==0) & (np.diff(d["qd"],axis=0)==0))[active[1:] & active[:-1]]
    summary = {"primary_mask": "Kp=17,Kd=0.4; trim first/last 1s of each segment; not a contact/gait label",
               "primary_samples": int(trimmed.sum()), "dt_median_s":dt,
               "log_interval_ms":quantiles(np.diff(t)*1000),"gain_segments":gain_segments,
               "command_update_interval_ms":quantiles(intervals*1000),
               "command_updates":int(len(updates)),
               "command_step_p99_abs_rad":np.percentile(np.abs(delta_target),99,axis=0).tolist(),
               "feedback_identical_q_qd_fraction":joint_held.mean(axis=0).tolist(),
               "all_feedback_held_fraction":float(held[active[1:] & active[:-1]].mean()),
               "joints":rows,
               "lag_method":"6s windows, position cross-correlation after per-window mean removal, +/-150ms; positive=target leads feedback; diagnostics only",
               "lag_usable_filter":"target std>=0.03rad, peak correlation>=0.5, peak not at search boundary",
               "not_identified":["hardware transport delay","motor time constant","encoder noise variance","true torque/power","friction coefficients","backlash","torque saturation","contact-conditioned tracking"]}
    (out/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    for name, data in [('joint_tracking.csv',rows),('apparent_lag_windows.csv',lag_rows)]:
        with (out/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(4,3,figsize=(15,10),sharex=True)
    for i,ax in enumerate(axes.flat):
        mask=trimmed[::5];tt=t[::5][mask]
        ax.plot(tt,np.degrees(errors[::5,i][mask]),linewidth=.6)
        ax.axhline(0,color='k',linewidth=.5);ax.set_title(JOINTS[i]);ax.set_ylabel('Feedback - target (deg)');ax.grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('Log time (s)')
    fig.tight_layout();fig.savefig(str(out/'tracking_errors.png'),dpi=150);plt.close(fig)
    fig,axes=plt.subplots(4,3,figsize=(15,10),sharex=True,sharey=True)
    for i,lags,vals in curves:axes.flat[i].plot(lags,vals,linewidth=.8,alpha=.7)
    for i,ax in enumerate(axes.flat):
        ax.axvline(0,color='k',linewidth=.5);ax.set_title(JOINTS[i]);ax.set_ylabel('Correlation');ax.grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('Apparent lag (ms); positive = target leads')
    fig.tight_layout();fig.savefig(str(out/'lag_diagnostics.png'),dpi=150);plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()

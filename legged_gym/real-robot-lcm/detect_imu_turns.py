#!/usr/bin/env python3
"""Offline turn segmentation from microstrain omega[2]; no robot connection."""
import argparse
import csv
import json
import struct
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter
from analyze_joint_log import events


def spans(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def detect(t, w, low, high):
    dt = float(t[1] - t[0]); result = []
    for sign in (1, -1):
        mask = sign * w >= low
        for a, b in spans(~mask):
            if a > 0 and b < len(mask) and (b-a)*dt <= .15:
                mask[a:b] = True
        for a,b in spans(mask):
            duration = (b-a)*dt
            high_runs = spans(sign*w[a:b] >= high)
            if duration < .35 or not high_runs or max(y-x for x,y in high_runs)*dt < .15:
                continue
            end = min(t[-1],t[b-1]+dt)
            result.append(dict(start_s=float(t[a]), end_s=float(end), duration_s=float(end-t[a]),
                direction='positive' if sign == 1 else 'negative',
                mean_yaw_rate_rad_s=float(w[a:b].mean()),
                peak_abs_yaw_rate_rad_s=float(np.abs(w[a:b]).max()),
                integrated_body_z_deg=float(np.rad2deg(w[a:b].sum()*dt))))
    return sorted(result,key=lambda r:r['start_s'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('log',type=Path); p.add_argument('--output',type=Path)
    args=p.parse_args(); out=args.output or args.log.with_name(args.log.name+'_analysis')/'imu_turns'; out.mkdir(parents=True,exist_ok=True)
    rows=[]; origin=None
    for n,ts,ch,payload in events(args.log):
        if origin is None:origin=ts
        if ch != 'microstrain':continue
        if payload[:8].hex() != 'e21531ea1392faaa' or len(payload)!=76:
            raise ValueError('microstrain schema fingerprint/length mismatch')
        v=struct.unpack('>13f2q',payload[8:])
        rows.append([(ts-origin)*1e-6,v[9],v[6]]) # omega z, rpy yaw
    raw=np.asarray(rows)
    if not np.isfinite(raw).all() or not np.all(np.diff(raw[:,0])>0):raise ValueError('Invalid IMU samples')
    t=np.arange(raw[0,0],raw[-1,0],.005)
    w=median_filter(np.interp(t,raw[:,0],raw[:,1]),size=31,mode='nearest')
    cases={label:detect(t,w,low,high) for label,low,high in [('loose',.04,.08),('default',.06,.12),('strict',.08,.16)]}
    result={'log':str(args.log.resolve()),'time_origin_lcm_us':origin,'imu_samples':len(raw),'duration_s':float(raw[-1,0]),
        'method':{'field':'microstrain.omega[2]','grid_dt_s':.005,'median_window_s':.155,'low_rad_s':.06,'high_rad_s':.12,'bridge_gap_s':.15,'min_duration_s':.35,'min_continuous_high_s':.15},'segments':cases['default'],'sensitivity':cases}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'segments.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=list(cases['default'][0]));wr.writeheader();wr.writerows(cases['default'])
    np.savez_compressed(out/'samples.npz',raw=raw,filtered_time_s=t,filtered_yaw_rate_rad_s=w)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(13,7),sharex=True)
    axes[0].plot(raw[::10,0],raw[::10,1],color='0.7',lw=.6,label='raw IMU omega z')
    axes[0].plot(t,w,lw=1,label='155 ms median')
    axes[1].plot(raw[::10,0],np.rad2deg(np.unwrap(raw[:,2]))[::10],label='unwrapped IMU yaw')
    for i,s in enumerate(cases['default'],1):
        for ax in axes:ax.axvspan(s['start_s'],s['end_s'],alpha=.15,color='tab:orange' if s['direction']=='positive' else 'tab:blue')
        axes[0].text((s['start_s']+s['end_s'])/2,.85,str(i),ha='center')
    for ax in axes:ax.grid(alpha=.25);ax.legend(loc='upper left')
    axes[0].set_ylabel('Body yaw-axis rate (rad/s)'); axes[1].set_ylabel('IMU yaw angle (deg)');axes[1].set_xlabel('Time since first LCM event (s)')
    fig.tight_layout();fig.savefig(out/'turn_segments.png',dpi=160)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

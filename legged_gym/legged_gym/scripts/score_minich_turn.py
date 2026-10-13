"""Independent 100-point turn/foot/stop benchmark; no training reward dependency.

Offline: --capture DIR [--capture DIR ...].
Live: --policy /path/model_N.pt [--policy ...], CPU/headless by default.
Use --write-standards FILE to export provisional full/zero-score thresholds.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

DEFAULT = {
    'version': 'lift-minimum-v2', 'weights': {'turn': 40., 'feet': 30., 'stop': 30.},
    'contact_N': 1., 'stance_contact_N': 5., 'stand_grace_s': 1.,
    'stable_hold_s': .5, 'stable_yaw_rad_s': .02, 'stable_xy_m_s': .01,
    'turn_settle_s': 2., 'require_steps': True, 'lift_frequency_window_s': 4.,
    # Pairs mean [full-score threshold, zero-score threshold].
    'limits': {
        'yaw_relative_rmse': [.08, .50], 'turn_max_drift_m': [.03, .30],
        'longest_flight_s': [.30, 2.], 'stance_slip_mean_m_s': [.005, .05],
        'stance_slip_p95_m_s': [.02, .15],
        'settling_time_s': [1., 3.], 'stop_yaw_rmse_rad_s': [.02, .10],
        'stop_xy_speed_m_s': [.01, .08], 'stop_abs_angle_after_grace_deg': [3., 30.],
        'stop_all_contact_fraction': [.98, .50],
        'normal_touchdown_fraction': [.8, .2], 'touchdowns_per_foot_per_s': [.5, 0.],
        'alternating_touchdown_fraction': [.8, .2], 'turn_two_support_fraction': [.98, .50],
    },
    'normal_flight_s': [.20, .30], 'ground_foot_center_m': .02, 'minimum_lift_clearance_m': .01,
}


def validate(cfg):
    if set(cfg) != set(DEFAULT) or set(cfg['limits']) != set(DEFAULT['limits']):
        raise ValueError('Standards keys must match the exported schema')
    if set(cfg['weights']) != {'turn','feet','stop'} or not np.isclose(sum(cfg['weights'].values()),100):
        raise ValueError('Weights must sum to 100')
    if any(not np.isfinite(x) or x < 0 for x in cfg['weights'].values()):
        raise ValueError('Weights must be nonnegative finite numbers')
    for key,pair in cfg['limits'].items():
        if len(pair)!=2 or not np.isfinite(pair).all() or pair[0]==pair[1]:
            raise ValueError('Invalid threshold pair: '+key)
        higher = key in ('stop_all_contact_fraction','normal_touchdown_fraction','touchdowns_per_foot_per_s','alternating_touchdown_fraction','turn_two_support_fraction')
        if (pair[0]>pair[1]) != higher or min(pair)<0:
            raise ValueError('Threshold direction is invalid: '+key)
    for key in ('contact_N','stance_contact_N','stand_grace_s','stable_hold_s','stable_yaw_rad_s','stable_xy_m_s','turn_settle_s','lift_frequency_window_s'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0: raise ValueError('Invalid '+key)
    for key in ('ground_foot_center_m','minimum_lift_clearance_m'):
        if not np.isfinite(cfg[key]) or cfg[key] <= 0: raise ValueError('Invalid '+key)
    if not isinstance(cfg['require_steps'], bool): raise ValueError('require_steps must be boolean')
    if not 0 < cfg['normal_flight_s'][0] < cfg['normal_flight_s'][1]: raise ValueError('Invalid normal flight interval')
    return cfg


def grade(value, pair):
    return float(np.clip((value-pair[1])/(pair[0]-pair[1]),0.,1.))


def longest(mask, dt):
    edges=np.diff(np.r_[False, mask, False].astype(int))
    lengths=np.flatnonzero(edges==-1)-np.flatnonzero(edges==1)
    return float(lengths.max()*dt) if len(lengths) else 0.


def evaluate(meta, data, robot, cfg):
    validate(cfg)
    dt=float(meta['dt']); before,forward,turn,stop=meta['schedule_s']
    if forward != 0 or turn<=0 or stop<=cfg['stand_grace_s']+cfg['stable_hold_s']:
        raise ValueError('Require stand -> nonzero turn -> sufficiently long stand, without forward phase')
    command=float(meta['yaw_command_rad_s'])
    if not np.isfinite(command) or command==0: raise ValueError('Nonzero finite yaw command required')
    if not np.isfinite(dt) or dt<=0: raise ValueError('Invalid control dt')
    states=data['states'][:,robot]; c={k:i for i,k in enumerate(meta['columns'])}
    if len(states)!=round(sum(meta['schedule_s'])/dt): raise ValueError('Incomplete capture')
    if not np.isfinite(states).all():
        return {'total':0.,'invalid':True,'reason':'Nonfinite telemetry','policy':meta['policies'][robot]}
    lo=round(before/dt); end=lo+round(turn/dt); grace=round(cfg['stand_grace_s']/dt)
    warm=min(round(cfg['turn_settle_s']/dt),round((end-lo)/2))
    turn_s=states[lo:end]; steady=states[lo+warm:end]; stand=states[end:]; late=stand[grace:]
    def cols(s,prefix): return s[:,[c[prefix+str(i)] for i in range(4)]]
    force=cols(states,'foot_fz_'); contact=force>cfg['contact_N']
    flights=[longest(~contact[lo:end,f],dt) for f in range(4)]
    speed=cols(turn_s,'foot_xy_speed_'); stance=cols(turn_s,'foot_fz_')>cfg['stance_contact_N']
    support_speed=speed[stance]
    # Flights starting before the turn are counted, but initialisation artefacts
    # are excluded from optional normal-touchdown quality/counts.
    foot_height=cols(states,'foot_world_z_')
    air=np.zeros(4); prev=np.zeros(4,dtype=bool); touchdowns=[]; counts=np.zeros(4); events=[]; peaks=[]; peak=np.full(4,cfg['ground_foot_center_m'])
    qualified = np.zeros((end-lo, 4), dtype=int)
    for step in range(end):
        peak=np.maximum(peak,foot_height[step])
        filtered=contact[step]|prev; first=(air>0)&filtered; air+=dt
        if step>=lo:
            event=set()
            for f in np.flatnonzero(first):
                if (step+1)*dt-air[f]>=before-dt/2:
                    clearance = float(peak[f]-cfg['ground_foot_center_m'])
                    touchdowns.append(float(air[f]));peaks.append(clearance);counts[f]+=1
                    if clearance >= cfg['minimum_lift_clearance_m'] - 1e-6:
                        qualified[step-lo, f] = 1
                        event.add(int(f))
            if event:
                if events and (step-events[-1][0])*dt<=.04+1e-6:
                    events[-1][1].update(event)
                else: events.append((step,event))
        air*=~filtered;peak[filtered]=cfg['ground_foot_center_m'];prev=contact[step]
    # Count completed lifts in every rolling window; the least active foot governs.
    window_steps = min(end-lo, max(1, round(cfg['lift_frequency_window_s']/dt)))
    window_s = window_steps * dt
    cumulative = np.vstack((np.zeros((1,4), dtype=int), np.cumsum(qualified, axis=0)))
    window_counts = cumulative[window_steps:] - cumulative[:-window_steps]
    minimum_rates = window_counts.min(axis=0) / window_s
    xy=lambda s: np.linalg.norm(s[:,[c['body_vx'],c['body_vy']]],axis=1)
    wz=stand[:,c['body_wz']]
    stable=(np.abs(wz)<=cfg['stable_yaw_rad_s']) & (xy(stand)<=cfg['stable_xy_m_s']) & np.all(contact[end:],axis=1)
    hold=max(1,round(cfg['stable_hold_s']/dt))
    # The first hold-length window which remains stable for the rest of capture.
    # End-of-capture snippets shorter than hold are not accepted as settling.
    suffix=np.logical_and.accumulate(stable[::-1])[::-1]
    eligible=np.flatnonzero(suffix[:len(suffix)-hold+1])
    settling=float((eligible[0]+1)*dt) if len(eligible) else float(stop+dt)
    metrics={
        'yaw_relative_rmse':float(np.sqrt(np.mean((steady[:,c['body_wz']]-command)**2))/max(abs(command),.25)),
        'turn_max_drift_m':float(np.linalg.norm(turn_s[:,[c['x'],c['y']]]-states[max(lo-1,0),[c['x'],c['y']]],axis=1).max()),
        'longest_flight_s':max(flights),
        'stance_slip_mean_m_s':float(support_speed.mean()) if len(support_speed) else cfg['limits']['stance_slip_mean_m_s'][1],
        'stance_slip_p95_m_s':float(np.percentile(support_speed,95)) if len(support_speed) else cfg['limits']['stance_slip_p95_m_s'][1],
        'settling_time_s':settling, 'stop_yaw_rmse_rad_s':float(np.sqrt(np.mean(late[:,c['body_wz']]**2))),
        'stop_xy_speed_m_s':float(xy(late).mean()),
        'stop_abs_angle_after_grace_deg':float(np.rad2deg(np.abs(late[:,c['body_wz']]).sum()*dt)),
        'stop_all_contact_fraction':float(np.all(contact[end+grace:],axis=1).mean()),
        'normal_touchdown_fraction':float(np.mean((np.array(touchdowns)>=cfg['normal_flight_s'][0]-1e-6)&(np.array(touchdowns)<=cfg['normal_flight_s'][1]+1e-6)&(np.array(peaks)>=cfg['minimum_lift_clearance_m']-1e-6))) if touchdowns else 0.,
        'touchdowns_per_foot_per_s':float(minimum_rates.min()),
        'alternating_touchdown_fraction':float(np.mean([not (a[1]&b[1]) for a,b in zip(events,events[1:])])) if len(events)>1 else 0.,
        'turn_two_support_fraction':float((contact[lo:end].sum(axis=1)>=2).mean()),
    }
    g={key:grade(value,cfg['limits'][key]) for key,value in metrics.items()}
    lift=.35*g['normal_touchdown_fraction']+.30*g['touchdowns_per_foot_per_s']+.20*g['alternating_touchdown_fraction']+.15*g['turn_two_support_fraction']
    # Require-steps mode reserves half of feet score for normal alternating steps.
    feet=.4*g['longest_flight_s']+.3*g['stance_slip_mean_m_s']+.3*g['stance_slip_p95_m_s']
    if cfg['require_steps']: feet=.5*feet+.5*lift
    fractions={'turn':.7*g['yaw_relative_rmse']+.3*g['turn_max_drift_m'], 'feet':feet,
               'stop':.3*g['settling_time_s']+.3*g['stop_yaw_rmse_rad_s']+.2*g['stop_abs_angle_after_grace_deg']+.1*g['stop_xy_speed_m_s']+.1*g['stop_all_contact_fraction']}
    fell=bool((states[:,c['z']]<.13).any() or (states[:,c['gravity_z']]>-.7).any())
    points={k:fractions[k]*cfg['weights'][k] for k in fractions}
    return {'policy':meta['policies'][robot],'target_yaw_rad_s':command,'turn_duration_s':turn,
            'total':0. if fell else sum(points.values()), 'fell':fell,
            'minimum_lift_pass':bool(not fell and np.all(minimum_rates >= cfg['limits']['touchdowns_per_foot_per_s'][0]-1e-6)), 'components':points,'metrics':metrics,
            'metric_scores':{k:v*100 for k,v in g.items()},'deductions':[k for k,v in g.items() if v<1 and (cfg['require_steps'] or k not in ('normal_touchdown_fraction','touchdowns_per_foot_per_s','alternating_touchdown_fraction','turn_two_support_fraction'))],
            'diagnostics':{'longest_flight_per_foot_s':flights,'touchdowns_per_foot':counts.tolist(),
                           'qualified_lifts_per_foot':qualified.sum(axis=0).tolist(),
                           'qualified_lift_rates_per_foot_hz':(qualified.sum(axis=0)/turn).tolist(),
                           'minimum_window_lift_rates_per_foot_hz':minimum_rates.tolist(),
                           'lift_frequency_window_s':window_s,
                           'touchdown_event_groups':[sorted(e[1]) for e in events],
                           'touchdown_clearance_m':peaks,
                           'turn_mean_yaw_rad_s':float(turn_s[:,c['body_wz']].mean()),
                           'turn_all_contact_fraction':float(np.all(contact[lo:end],axis=1).mean()),
                           'stop_abs_angle_total_deg':float(np.rad2deg(np.abs(wz).sum()*dt)),
                           'foot_center_mean_height_m':cols(turn_s,'foot_world_z_').mean(axis=0).tolist(),
                           'flight_censored_at_turn_end':(~contact[end-1]).tolist(),'never_settled':not len(eligible)}}


def worker(args):
    # Isaac Gym must load before torch (imported by playback).
    import compare_minich_turn as diagnostic
    p=Path(args.policy[0]).resolve()
    number=int(p.stem.split('_')[-1])
    diagnostic.playback.POLICY_SPECS=tuple((label,str(p.parent),number) for label in ('manual_1','manual_2','latest'))
    diagnostic.playback.TURN_S=args.turn_seconds
    diagnostic.playback.STAND_AFTER_S=args.stop_seconds
    os.environ.update(COMPARE_MODE=args.mode,COMPARE_HISTORY='single',COMPARE_YAW_RAD_S=str(args.yaw_rates[0]),COMPARE_OUTPUT=args.output)
    sys.argv=[sys.argv[0],'--task=minich','--headless','--sim_device='+args.sim_device,'--pipeline=cpu','--rl_device=cpu','--seed='+str(args.seeds[0])]
    diagnostic.main()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture',action='append',default=[],help='Directory with metadata.json and samples.npz')
    p.add_argument('--policy',action='append',default=[],help='Explicit model_N.pt path; no implicit latest selection')
    p.add_argument('--output',default='legged_gym/logs/turn_scores')
    p.add_argument('--standards');p.add_argument('--write-standards')
    p.add_argument('--yaw-rates',nargs='+',type=float,default=[.25,-.25,1.,-1.])
    p.add_argument('--seeds',nargs='+',type=int,default=[1])
    p.add_argument('--turn-seconds',type=float,default=10.);p.add_argument('--stop-seconds',type=float,default=6.)
    p.add_argument('--mode',choices=['controlled','original'],default='controlled')
    p.add_argument('--sim-device',default='cpu');p.add_argument('--robot-index',type=int,default=2)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    cfg=validate(json.loads(Path(args.standards).read_text()) if args.standards else json.loads(json.dumps(DEFAULT)))
    if args.write_standards:
        Path(args.write_standards).write_text(json.dumps(cfg,indent=2)+'\n')
        if not args.capture and not args.policy:return
    if args.worker:return worker(args)
    if not args.capture and not args.policy:p.error('Specify --capture or --policy')
    if not np.isfinite([args.turn_seconds,args.stop_seconds]).all() or args.turn_seconds<=0 or args.stop_seconds<=cfg['stand_grace_s']+cfg['stable_hold_s']:p.error('Invalid test durations')
    if any(not np.isfinite(w) or w==0 for w in args.yaw_rates):p.error('Yaw commands must be finite and nonzero')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    captures=list(args.capture)
    for policy in args.policy:
        path=Path(policy).resolve()
        if not path.is_file() or not path.stem.startswith('model_'):p.error('Checkpoint must be an existing model_N.pt: '+str(path))
        digest=hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        for yaw in args.yaw_rates:
            for seed in args.seeds:
                dest=output/f'{path.parent.name}_{path.stem}_{digest}_yaw{yaw:g}_seed{seed}_turn{args.turn_seconds:g}_{args.mode}'
                if dest.exists():raise FileExistsError('Capture exists; choose a new --output: '+str(dest))
                cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--policy',str(path),'--output',str(dest),'--yaw-rates',str(yaw),'--seeds',str(seed),'--turn-seconds',str(args.turn_seconds),'--stop-seconds',str(args.stop_seconds),'--mode',args.mode,'--sim-device',args.sim_device]
                print('Running',path.name,'yaw',yaw,'seed',seed,flush=True)
                with (output/(dest.name+'.log')).open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
                if hashlib.sha256(path.read_bytes()).hexdigest()[:12]!=digest:raise RuntimeError('Checkpoint changed during capture')
                captures.append(str(dest))
    rows=[]
    for path in captures:
        root=Path(path).resolve();meta=json.loads((root/'metadata.json').read_text())
        with np.load(root/'samples.npz') as data:r=evaluate(meta,data,args.robot_index,cfg)
        r.update(capture=str(root),mode=meta.get('mode'),seed=meta.get('seed'),history=meta.get('history_timing'),schedule_s=meta['schedule_s'],environment_sha256=hashlib.sha256(json.dumps({'cfg':meta.get('env_config'),'factors':meta.get('realized_random_factors')},sort_keys=True).encode()).hexdigest())
        rows.append(r)
    groups={}
    for r in rows:
        key=r['policy']['sha256'];groups.setdefault(key,[]).append(r)
    ranking=[]
    for key,group in groups.items():
        values=np.array([r['total'] for r in group]);ranking.append({'policy':group[0]['policy'],'mean':float(values.mean()),'worst':float(values.min()),'std':float(values.std()),'n':len(group)})
    matrices=[sorted((r.get('target_yaw_rad_s',0),tuple(r['schedule_s']),r['mode'],r['seed'],r['history'],r['environment_sha256']) for r in group) for group in groups.values()]
    comparable=all(matrix==matrices[0] for matrix in matrices)
    ranking.sort(key=lambda r:r['mean'],reverse=True)
    result={'matching_case_matrices':comparable,'standards':cfg,'ranking':ranking,'cases':rows,'note':'Scores are provisional. Compare identical case matrices; slip is foot-body velocity proxy, not contact-point slip.'}
    (output/'scores.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    lines=['# 原地旋转跑分（满分100，标准初稿）','','不同测试矩阵不应直接比较均分；摔倒场景总分归零。支撑足滑动采用足刚体水平速度代理。','',f"| 策略 | 目标 rad/s | 旋转秒数 | 旋转 /{cfg['weights']['turn']:g} | 足端 /{cfg['weights']['feet']:g} | 停止 /{cfg['weights']['stop']:g} | 总分 | 最低抬脚标准 |",'|---|---:|---:|---:|---:|---:|---:|---|']
    if not comparable: lines[2] += ' 本次策略测试矩阵不一致，均分排名不能直接比较。'
    for r in rows:
        a=r.get('components',{'turn':0,'feet':0,'stop':0})
        lines.append(f"| {Path(r['policy']['path']).parent.name}/{Path(r['policy']['path']).name} | {r.get('target_yaw_rad_s',0):g} | {r.get('turn_duration_s',0):g} | {a['turn']:.1f} | {a['feet']:.1f} | {a['stop']:.1f} | {r['total']:.1f} | {'通过' if r.get('minimum_lift_pass',False) else '未通过'} |")
    lines+=['','各项原始指标、扣分原因及满分/零分阈值见 scores.json。未要求抬脚时，正常落脚率只作诊断，不影响总分。']
    (output/'scores.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines));print('Saved to',output)


if __name__=='__main__':main()

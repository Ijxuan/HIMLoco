"""Compare calf PD gains using recorded real-robot targets in CPU Isaac Gym.

Six independent six-second windows initialized from logged joint/base state.
Optional logged-base/non-calf constraints diagnose contact confounding. These
tests compare effective simulation gains, not hardware gains. A valid flag only
means no low-base/tilt threshold violation; constraints preclude locomotion validation.
"""
import csv
import hashlib
import json
import sys
from pathlib import Path

import isaacgym
from isaacgym import gymtorch, gymutil
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *
from legged_gym.utils import class_to_dict, task_registry

DATA_DIR = Path(LEGGED_GYM_ROOT_DIR) / "real-robot-lcm"
sys.path.insert(0, str(DATA_DIR))
from analyze_joint_log import decode, events


def load_log(path):
    import struct
    times, commands, feedback, state = [], [], [], []
    pending = data = None
    for _, ts, ch, payload in events(path):
        if ch == 'leg_control_command':
            pending = decode(ch, payload)
        elif ch == 'leg_control_data':
            data = decode(ch, payload)
            data_time = ts
        elif ch == 'state_estimator':
            if pending is None or data is None:
                raise ValueError('Unpaired channels')
            if payload[:8].hex() != '32a7a58b9685d099':
                raise ValueError('State estimator schema mismatch')
            times.append(data_time)
            commands.append(pending['q_des'])
            feedback.append((data['q'], data['qd']))
            state.append(struct.unpack('>31f', payload[8:]))
            pending = data = None
    times = (np.asarray(times, dtype=np.int64)-times[0])/1e6
    reorder = [3,4,5,0,1,2,9,10,11,6,7,8]
    return times, np.asarray(commands)[:,reorder], np.asarray(feedback)[:,:,reorder], np.asarray(state)


def main():
    args = gymutil.parse_arguments(description=__doc__, custom_parameters=[
        {'name':'--headless','action':'store_true'},
        {'name':'--rl_device','type':str,'default':'cpu'},
        {'name':'--log','type':str,'default':str(DATA_DIR/'record.lcm.00')},
        {'name':'--output','type':str,'default':str(DATA_DIR/'record.lcm.00_analysis/calf_gain_fit/coarse')},
        {'name':'--kp_grid','type':str,'default':'6,8,10,12,17,24,34'},
        {'name':'--kd_grid','type':str,'default':'0.15,0.25,0.4,0.6,0.8,1.2'},
        {'name':'--friction','type':float,'default':1.0},
        {'name':'--root_height_offset','type':float,'default':0.0},
        {'name':'--root_mode','type':str,'default':'free'},
        {'name':'--noncalf_mode','type':str,'default':'free'},
    ])
    torch.set_num_threads(1)
    if args.root_mode not in ('free','logged'):
        raise ValueError('root_mode must be free or logged')
    if args.noncalf_mode not in ('free','logged'):
        raise ValueError('noncalf_mode must be free or logged')
    if args.rl_device != 'cpu' or args.sim_device_type != 'cpu':
        raise ValueError('This diagnostic currently requires CPU simulation and RL device')
    grid = [(kp,kd) for kp in map(float,args.kp_grid.split(',')) for kd in map(float,args.kd_grid.split(','))]
    out = Path(args.output);out.mkdir(parents=True,exist_ok=True)
    times, targets, real, states = load_log(Path(args.log))
    cfg,_ = task_registry.get_cfgs('minich')
    cfg.env.num_envs=len(grid);cfg.env.env_spacing=4.;cfg.env.episode_length_s=60
    cfg.terrain.mesh_type='plane';cfg.terrain.curriculum=False
    cfg.terrain.static_friction=args.friction;cfg.terrain.dynamic_friction=args.friction
    cfg.noise.add_noise=False
    for key in dir(cfg.domain_rand):
        if isinstance(getattr(cfg.domain_rand,key),bool):setattr(cfg.domain_rand,key,False)
    cfg.commands.curriculum=False;cfg.commands.enable_turn_to_stand=False
    cfg.commands.in_place_turn_probability=0.;cfg.commands.heading_command=False
    args.task='minich';args.num_envs=len(grid);args.seed=1
    args.max_iterations=args.resume=args.experiment_name=args.run_name=None
    args.load_run=args.checkpoint=None
    env=task_registry.make_env(name='minich',args=args,env_cfg=cfg)[0]
    expected=['FL_hip_joint','FL_thigh_joint','FL_calf_joint','FR_hip_joint','FR_thigh_joint','FR_calf_joint',
              'RL_hip_joint','RL_thigh_joint','RL_calf_joint','RR_hip_joint','RR_thigh_joint','RR_calf_joint']
    if env.dof_names!=expected:raise ValueError('Unexpected DOF order')
    calf=[2,5,8,11]
    kp=torch.full((len(grid),12),17.);kd=torch.full((len(grid),12),.4)
    for i,(p,d) in enumerate(grid):kp[i,calf]=p;kd[i,calf]=d
    dt=env.sim_params.dt
    window_starts=[20.,26.,32.,38.,44.,50.]
    duration=6.;steps=int(round(duration/dt))
    q_all=[];qd_all=[];height_all=[];up_all=[];reference_all=[];velocity_all=[];target_all=[]
    try:
        ids=torch.arange(len(grid),dtype=torch.int32)
        for window,start in enumerate(window_starts):
            ix=np.searchsorted(times,start,side='right')-1
            st=states[ix]
            # Ignore corrupted/unbounded estimator X/Y position fields. Plane
            # is translation invariant; use environment origins instead.
            env.root_states.zero_();env.root_states[:,:3]=env.env_origins
            env.root_states[:,2]=float(st[2])+args.root_height_offset
            env.root_states[:,3:7]=torch.tensor(st[[22,23,24,21]],dtype=torch.float)
            env.root_states[:,3:7]/=torch.linalg.norm(env.root_states[:,3:7],dim=1,keepdim=True)
            env.root_states[:,7:10]=torch.tensor(st[3:6],dtype=torch.float)
            env.root_states[:,10:13]=torch.tensor(st[18:21],dtype=torch.float)
            env.dof_pos[:]=torch.tensor(real[ix,0],dtype=torch.float)
            env.dof_vel[:]=torch.tensor(real[ix,1],dtype=torch.float)
            env.gym.set_actor_root_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.root_states),gymtorch.unwrap_tensor(ids),len(grid))
            env.gym.set_dof_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.dof_state),gymtorch.unwrap_tensor(ids),len(grid))
            q_hist=[];qd_hist=[];height_hist=[];up_hist=[];refs=[];vrefs=[];tgts=[]
            xy=torch.zeros(2)
            for step in range(steps):
                before=start+step*dt;after=before+dt
                ix=np.searchsorted(times,before,side='right')-1
                if args.root_mode=='logged':
                    st=states[ix]
                    env.root_states[:,:2]=env.env_origins[:,:2]+xy
                    env.root_states[:,2]=float(st[2])+args.root_height_offset
                    env.root_states[:,3:7]=torch.tensor(st[[22,23,24,21]],dtype=torch.float)
                    env.root_states[:,3:7]/=torch.linalg.norm(env.root_states[:,3:7],dim=1,keepdim=True)
                    env.root_states[:,7:10]=torch.tensor(st[3:6],dtype=torch.float)
                    env.root_states[:,10:13]=torch.tensor(st[18:21],dtype=torch.float)
                    env.gym.set_actor_root_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.root_states),gymtorch.unwrap_tensor(ids),len(grid))
                    xy+=torch.tensor(st[3:5],dtype=torch.float)*dt
                desired=torch.tensor(targets[ix],dtype=torch.float)
                if args.noncalf_mode=='logged':
                    other=[j for j in range(12) if j not in calf]
                    env.dof_pos[:,other]=torch.tensor([np.interp(before,times,real[:,0,j]) for j in other],dtype=torch.float)
                    env.dof_vel[:,other]=torch.tensor([np.interp(before,times,real[:,1,j]) for j in other],dtype=torch.float)
                    env.gym.set_dof_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.dof_state),gymtorch.unwrap_tensor(ids),len(grid))
                tau=torch.clamp(kp*(desired-env.dof_pos)-kd*env.dof_vel,-env.torque_limits,env.torque_limits)
                env.gym.set_dof_actuation_force_tensor(env.sim,gymtorch.unwrap_tensor(tau))
                env.gym.simulate(env.sim);env.gym.fetch_results(env.sim,True)
                env.gym.refresh_dof_state_tensor(env.sim);env.gym.refresh_actor_root_state_tensor(env.sim)
                if not torch.isfinite(env.dof_pos).all():raise ValueError('Non-finite simulation')
                ri=min(np.searchsorted(times,after,side='right')-1,len(times)-1)
                q_hist.append(env.dof_pos.numpy().copy());qd_hist.append(env.dof_vel.numpy().copy())
                height_hist.append(env.root_states[:,2].numpy().copy())
                quat=env.root_states[:,3:7]
                up_hist.append((1-2*(quat[:,0]**2+quat[:,1]**2)).numpy().copy())
                # Interpolate feedback ONLY; commands retain zero-order hold.
                refs.append([np.interp(after,times,real[:,0,j]) for j in range(12)])
                vrefs.append([np.interp(after,times,real[:,1,j]) for j in range(12)])
                tgts.append(targets[ri])
            q_all.append(q_hist);qd_all.append(qd_hist);height_all.append(height_hist);up_all.append(up_hist)
            reference_all.append(refs);velocity_all.append(vrefs);target_all.append(tgts)
            print(f'Window {window+1}/6 complete: log {start:g}--{start+duration:g}s',flush=True)
    finally:env.gym.destroy_sim(env.sim)
    q=np.asarray(q_all);qd=np.asarray(qd_all);height=np.asarray(height_all);up=np.asarray(up_all)
    ref=np.asarray(reference_all);vref=np.asarray(velocity_all);target=np.asarray(target_all)
    # Chronological split, no overlapping calibration and validation windows.
    burn=int(round(.5/dt));rows=[]
    for i,(p,d) in enumerate(grid):
        row={'kp':p,'kd':d}
        for label,sl in [('calibration',slice(0,3)),('validation',slice(3,6))]:
            sim=q[sl,burn:,i][:,:,calf];actual=ref[sl,burn:][:,:,calf];des=target[sl,burn:][:,:,calf]
            mismatch=sim-actual;err=sim-des;realerr=actual-des
            row[label+'_angle_match_rmse_deg']=float(np.degrees(np.sqrt(np.mean(mismatch**2))))
            row[label+'_tracking_rmse_deg']=float(np.degrees(np.sqrt(np.mean(err**2))))
            row[label+'_real_tracking_rmse_deg']=float(np.degrees(np.sqrt(np.mean(realerr**2))))
            row[label+'_bias_deg']=float(np.degrees(err.mean()))
            row[label+'_real_bias_deg']=float(np.degrees(realerr.mean()))
            row[label+'_velocity_match_rmse_rad_s']=float(np.sqrt(np.mean((qd[sl,burn:,i][:,:,calf]-vref[sl,burn:][:,:,calf])**2)))
            row[label+'_fall_fraction']=float(np.mean((height[sl,burn:,i]<.13)|(up[sl,burn:,i]<.7)))
        row['min_base_height_m']=float(height[:,:,i].min())
        row['valid']=row['calibration_fall_fraction']==0 and row['validation_fall_fraction']==0
        rows.append(row)
    ranked=sorted(rows,key=lambda r:(not r['valid'],r['calibration_angle_match_rmse_deg']))
    meta={'log':str(Path(args.log).resolve()),'log_sha256':hashlib.sha256(Path(args.log).read_bytes()).hexdigest(),
          'dt_s':dt,'cfg':class_to_dict(cfg),'dof_names':expected,'windows_log_start_s':window_starts,
          'duration_s':duration,'excluded_start_s':.5,'calibration_windows':[0,1,2],'validation_windows':[3,4,5],
          'friction':args.friction,'root_height_offset':args.root_height_offset,
          'root_mode':args.root_mode,
          'noncalf_mode':args.noncalf_mode,
          'method':'Free-base target replay initialized per window from logged joint states, base quaternion, height, velocities. No policy inference, randomization, delay, automatic reset or root forcing.',
          'limitations':['Unmeasured contact forces; root motion evolves independently after initialization.',
                         'The base XY estimator positions are corrupted and unused.',
                         'The log is one closed-loop trajectory; effective gains are conditional on model/contact assumptions.'],
          'ranked_results':ranked,'best_valid':next((r for r in ranked if r['valid']),None)}
    if args.root_mode=='logged':
        meta['method']='Conditional target replay with logged base height/quaternion/velocities imposed each physics step. Base XY integrates logged velocity. Joint dynamics remain free. External base constraint prevents free locomotion validation.'
    if args.noncalf_mode=='logged':
        meta['method']+=' Non-calf positions/velocities are additionally imposed from logged feedback; only calf dynamics evolve freely.'
    (out/'summary.json').write_text(json.dumps(meta,indent=2,default=str)+'\n')
    with (out/'scores.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(ranked)
    np.savez_compressed(out/'replay.npz',q_sim=q,qd_sim=qd,q_real=ref,qd_real=vref,q_target=target,
                        base_height=height,up_z=up,grid=grid,time_in_window_s=(np.arange(steps)+1)*dt)
    print('BEST_VALID='+json.dumps(meta['best_valid']),flush=True)


if __name__=='__main__':main()

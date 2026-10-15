"""Compare checkpoint joint tracking across calf gains and a fixed command.

Example: --gains=34:0.8,24:0.8,22:0.8 --command=0.5,0,0
CPU diagnostic only; no training or deployment changes.
"""
import csv
import hashlib
import io
import json
from pathlib import Path
from types import MethodType

import isaacgym
from isaacgym import gymutil
import numpy as np
import torch

import measure_minich_joint_loads as diagnostic
from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.utils import class_to_dict
from legged_gym.utils.helpers import get_load_path
from rsl_rl.modules import HIMActorCritic


def main():
    args=gymutil.parse_arguments(description=__doc__,custom_parameters=[
        {'name':'--headless','action':'store_true'},
        {'name':'--rl_device','type':str,'default':'cpu'},
        {'name':'--checkpoint','type':str,'default':None},
        {'name':'--output','type':str,'default':str(Path(LEGGED_GYM_ROOT_DIR)/'real-robot-lcm/record.lcm.00_analysis/latest_stand_gain_comparison')},
        {'name':'--command','type':str,'default':'0,0,0'},
        {'name':'--gains','type':str,'default':'34:0.8,8:1.2'},
        {'name':'--settle_s','type':float,'default':5.},
        {'name':'--duration_s','type':float,'default':10.},
    ])
    if args.rl_device != 'cpu' or args.sim_device_type != 'cpu':
        raise ValueError('This diagnostic requires CPU simulation and RL device')
    torch.set_num_threads(1)
    checkpoint=Path(args.checkpoint or get_load_path(str(Path(LEGGED_GYM_ROOT_DIR)/'logs/rough_minich'),load_run=-1,checkpoint=-1))
    payload=checkpoint.read_bytes()
    saved=torch.load(io.BytesIO(payload),map_location='cpu')
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    gains=[tuple(float(v) for v in pair.split(':')) for pair in args.gains.split(',')]
    if not gains or any(len(g)!=2 or not all(np.isfinite(v) and v>0 for v in g) for g in gains):
        raise ValueError('--gains must contain positive finite kp:kd pairs')
    command=tuple(float(v) for v in args.command.split(','))
    if len(command)!=3 or not all(np.isfinite(v) for v in command):
        raise ValueError('--command must contain finite vx,vy,yaw')
    if args.settle_s < 0 or args.duration_s <= 0:
        raise ValueError('Require settle_s >= 0 and duration_s > 0')
    diagnostic.CASES=tuple((f'calf_{kp:g}_{kd:g}',command) for kp,kd in gains)
    args.warmup_s=0.;args.ramp_s=1.;args.sample_s=args.duration_s
    env=diagnostic.configure(args)
    try:
        diagnostic.install_fixed_reset(env)
        diagnostic.reset_to_fixed_state(env)
        env.p_gains=env.p_gains.unsqueeze(0).repeat(len(gains),1)
        env.d_gains=env.d_gains.unsqueeze(0).repeat(len(gains),1)
        calf=[i for i,n in enumerate(env.dof_names) if n.endswith('calf_joint')]
        for i,(kp,kd) in enumerate(gains):
            env.p_gains[i,calf]=kp;env.d_gains[i,calf]=kd
        _,train_cfg=diagnostic.task_registry.get_cfgs('minich')
        model=HIMActorCritic(env.num_obs,env.num_privileged_obs,env.num_one_step_obs,env.num_actions,
                             **class_to_dict(train_cfg.policy)).to('cpu')
        model.load_state_dict(saved['model_state_dict'],strict=True);model.eval()
        def no_reset(self):
            self.reset_buf.zero_();self.time_out_buf.zero_()
        env.check_termination=MethodType(no_reset,env)
        original=env._compute_torques
        errors=[];positions=[];targets=[];torques=[];heights=[];up=[];base_velocity=[];body_velocity=[]
        def capture(self,actions):
            tau=original(actions)
            errors.append((self.dof_pos-self.joint_pos_target).numpy().copy())
            positions.append(self.dof_pos.numpy().copy());targets.append(self.joint_pos_target.numpy().copy())
            torques.append(tau.numpy().copy())
            return tau
        env._compute_torques=MethodType(capture,env)
        n=int(round((args.settle_s+args.duration_s)/env.dt))
        with torch.inference_mode():
            for step in range(n):
                env.commands.zero_()
                # Ramp over the first second, refreshing only current command channels.
                env.commands[:,:3]=torch.tensor(command)*min(1.,step*env.dt)
                env.obs_buf[:,:3]=env.commands[:,:3]*env.commands_scale
                actions=model.act_inference(env.get_observations())
                env.step(actions)
                if not torch.isfinite(env.dof_pos).all():
                    raise RuntimeError('Non-finite simulation state')
                for _ in range(env.cfg.control.decimation):
                    heights.append(env.root_states[:,2].numpy().copy())
                    up.append((-env.projected_gravity[:,2]).numpy().copy())
                    body_velocity.append(torch.stack((env.base_lin_vel[:,0],env.base_lin_vel[:,1],env.base_ang_vel[:,2]),dim=1).numpy().copy())
                    base_velocity.append(env.root_states[:,7:13].numpy().copy())
        errors=np.asarray(errors);positions=np.asarray(positions);targets=np.asarray(targets)
        torques=np.asarray(torques);heights=np.asarray(heights);up=np.asarray(up)
        base_velocity=np.asarray(base_velocity);body_velocity=np.asarray(body_velocity)
        dt=env.sim_params.dt;t=np.arange(len(errors))*dt
        # Use integer steps to avoid excluding the boundary through float dt rounding.
        mask=np.arange(len(errors))>=int(round(args.settle_s/dt))
        rows=[];groups=[]
        for i,(name,_) in enumerate(diagnostic.CASES):
            for j,joint in enumerate(env.dof_names):
                e=np.degrees(errors[mask,i,j]);q=np.degrees(positions[mask,i,j]);target=np.degrees(targets[mask,i,j])
                rows.append({'gains':name,'joint':joint,'mean_error_deg':float(e.mean()),
                             'mae_deg':float(np.abs(e).mean()),'rmse_deg':float(np.sqrt(np.mean(e**2))),
                             'p95_abs_deg':float(np.percentile(np.abs(e),95)),
                             'max_abs_deg':float(np.abs(e).max()),'mean_q_deg':float(q.mean()),
                             'mean_target_deg':float(target.mean()),'kp':float(env.p_gains[i,j]),'kd':float(env.d_gains[i,j])})
            e=np.degrees(errors[mask,i][:,calf])
            groups.append({'gains':name,'calf_rmse_deg':float(np.sqrt(np.mean(e**2))),
                           'actual_mean_vx_vy_yaw':body_velocity[mask,i].mean(axis=0).tolist(),
                           'calf_mae_deg':float(np.abs(e).mean()),'calf_p95_abs_deg':float(np.percentile(np.abs(e),95)),
                           'calf_mean_error_deg':float(e.mean()),'min_height_m':float(heights[mask,i].min()),
                           'mean_height_m':float(heights[mask,i].mean()),
                           'min_up_z':float(up[mask,i].min()),
                           'max_linear_speed_m_s':float(np.linalg.norm(base_velocity[mask,i,:3],axis=1).max()),
                           'max_angular_speed_rad_s':float(np.linalg.norm(base_velocity[mask,i,3:],axis=1).max()),
                           'max_joint_range_deg':float(np.degrees(np.ptp(positions[mask,i],axis=0)).max()),
                           'low_or_tilt_fraction':float(np.mean((heights[mask,i]<.13)|(up[mask,i]<.7))),
                           'low_or_tilt_during_settle':bool(np.any((heights[~mask,i]<.13)|(up[~mask,i]<.7)))})
        summary={'checkpoint':str(checkpoint.resolve()),'checkpoint_sha256':hashlib.sha256(payload).hexdigest(),
                 'checkpoint_iteration':saved.get('iter'), 'settle_s':args.settle_s,'sample_s':args.duration_s,
                 'physics_dt_s':dt,'policy_dt_s':env.dt,'commands':list(command),'command_ramp_s':1.,'environment_config':class_to_dict(env.cfg),
                 'sample_physics_steps':int(mask.sum()),
                 'state_sampling':'base state sampled at policy-step end and repeated across its physics substeps',
                 'error_definition':'feedback minus policy target, BEFORE each physics integration',
                 'randomization':False,'noise':False,'delay':False,'automatic_reset':False,
                 'initial_height_m':.36,'dof_names':env.dof_names,'groups':groups,'joints':rows}
        (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        with (out/'joint_errors.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        np.savez_compressed(out/'samples.npz',time_s=t,error_rad=errors,q_rad=positions,q_target_rad=targets,
                            torque_Nm=torques,height_m=heights,up_z=up,base_velocity=base_velocity,body_velocity=body_velocity,dof_names=env.dof_names)
        lines=['# Checkpoint 闭环增益对照','',f'Checkpoint: `{checkpoint.resolve()}`',f'SHA256: `{summary["checkpoint_sha256"]}`','',
               f'指令 vx/vy/yaw={command}，首秒渐变；{args.settle_s:g}秒稳定，{args.duration_s:g}秒统计（{mask.sum()}个物理步）；CPU PhysX。关闭噪声、随机化、动作延迟和自动复位。误差为每个物理步积分前的反馈角减策略目标角。髋/大腿均维持17/0.4。',
               '', '| 小腿增益 | 四小腿合并RMSE ° | MAE ° | P95绝对误差 ° | 平均高度 m | 低高度/倾倒比例 |',
               '|---|---:|---:|---:|---:|---:|']
        for g in groups:lines.append(f"| {g['gains']} | {g['calf_rmse_deg']:.3f} | {g['calf_mae_deg']:.3f} | {g['calf_p95_abs_deg']:.3f} | {g['mean_height_m']:.3f} | {g['low_or_tilt_fraction']:.3f} |")
        lines+=['','| 小腿关节 | '+' | '.join(g['gains']+' RMSE °' for g in groups)+' |',
                '|---|'+'---:|'*len(groups)]
        for j in calf:
            values=[next(r['rmse_deg'] for r in rows if r['gains']==g['gains'] and r['joint']==env.dof_names[j]) for g in groups]
            lines.append('| '+env.dof_names[j]+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
        lines+=['','统计段稳定性：']
        for g in groups:
            lines.append(f"- {g['gains']}: 平均 vx/vy/yaw={g['actual_mean_vx_vy_yaw']}；最大机身线速度 {g['max_linear_speed_m_s']:.6f} m/s，最大角速度 {g['max_angular_speed_rad_s']:.6f} rad/s；最大单关节角度范围 {g['max_joint_range_deg']:.4f}°；最小机身竖直分量 {g['min_up_z']:.6f}。")
        lines+=['','不能只根据未触发13cm低高度阈值就判为正常站立，须同时比较机身高度。站立误差不能与实机运动段误差视作同工况指标。结果只验证此checkpoint直接换增益，不代表重新训练后的表现。','']
        (out/'report.md').write_text('\n'.join(lines))
        print(json.dumps({'checkpoint':str(checkpoint),'groups':groups},indent=2),flush=True)
    finally:env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()

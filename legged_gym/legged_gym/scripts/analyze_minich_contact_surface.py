"""Replay recorded stair samples; does not change training or run a policy."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from minich_contact_surface import (CLASS_NAMES,RISER,MIXED,SUPPORT,
    UrdfFootKinematics,classify_stair_contact,contact_joint_torque,mixed_wall_normal_lower_bound)


def sustained(mask,dt,seconds):
    elapsed=np.zeros(mask.shape[1:],float);out=np.zeros_like(mask)
    for k in range(len(mask)):
        elapsed=np.where(mask[k],elapsed+dt,0.)
        out[k]=elapsed>=seconds-1e-6
    return out


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rows=[];mixed_rows=[];inputs=[];max_fk=0.;expected=0
    for c in json.loads((args.input/'cases.json').read_text()):
        d=args.input/c['label'];m=json.loads((d/'metadata.json').read_text());z=np.load(d/'samples.npz')
        assert all(np.isfinite(z[k]).all() for k in z.files)
        assert hashlib.sha256(Path(m['urdf']).read_bytes()).hexdigest()==m['urdf_sha256']
        dt=m['dt'];dec=round(dt/m['physics_dt']);n=len(z['root']);a=round(m['schedule_s'][0]/dt);b=a+round(m['schedule_s'][1]/dt)
        assert len(z['torques'])==n*dec
        kin=UrdfFootKinematics(m['urdf'],m['dof_names'],m['foot_names'])
        assert np.allclose(kin.radii,.02)
        fk,J=kin.evaluate(z['joint_pos'],z['root']);err=np.linalg.norm(fk-z['feet'],axis=-1)
        assert err.max()<1e-4, 'Coordinate or joint-order mismatch'
        max_fk=max(max_fk,float(err.max()))
        # Last substep torque and pre-integration velocity: force samples are
        # control-step net contact forces; do not infer exact substep dynamics.
        motor=z['torques'].reshape(n,dec,3,12)[:,-1]
        qd=z['joint_velocities'].reshape(n,dec,3,12)[:,-1]
        snapshot={}
        for tol in [.005,.008,.012]:
            result=classify_stair_contact(z['feet'],z['foot_force'],m['geometry'],tolerance=tol)
            labels=result['labels'];wall_tau=contact_joint_torque(J,result['wall_force'])
            support_tau=contact_joint_torque(J,result['support_force'])
            # Keep only motor effort that opposes the external normal load;
            # cap by both measured actuator effort and mapped normal load.
            opposed=wall_tau*motor[:, :, None, :]<0
            attributable=np.where(opposed,np.minimum(np.abs(wall_tau),np.abs(motor[:,:,None,:])),0.)
            slow=np.abs(qd[:,:,None,:])<.2
            quiet=np.linalg.norm(z['feet_vel'],axis=-1)<.03
            candidate=(attributable>3.)&slow&quiet[...,None]
            moving=z['commands'][:,:,0]>.1
            blocked=(np.abs(z['body_lin'][:,:,0])<.2*np.abs(z['commands'][:,:,0]))&moving
            active=sustained(candidate&blocked[:,:,None,None],dt,.2)
            # Diagnostic only, in Nm^2. Mixed-contact load deliberately excluded.
            cost=np.square(np.maximum(attributable-3.,0))*active
            for i,pol in enumerate(m['policies']):
                lab=labels[a:b,i];contact=lab!=0;den=max(int(contact.sum()),1)
                row=dict(case=c['label'],policy=pol['label'],lane=i,speed=c['speed'],tolerance_m=tol,
                    contact_foot_seconds=float(contact.sum()*dt),
                    class_foot_seconds={name:float((lab==j).sum()*dt) for j,name in enumerate(CLASS_NAMES)},
                    pure_riser_fraction_of_contact=float((lab==RISER).sum()/den),
                    mixed_fraction_of_contact=float((lab==MIXED).sum()/den),
                    external_wall_joint_load_absmax_Nm=float(np.abs(wall_tau[a:b,i]).max()),
                    support_joint_load_absmax_Nm=float(np.abs(support_tau[a:b,i]).max()),
                    blocked_seconds=float(blocked[a:b,i].sum()*dt),
                    sustained_wall_stall_seconds=float(active[a:b,i].any(axis=(-1,-2)).sum()*dt),
                    diagnostic_cost_mean_Nm2=float(cost[a:b,i].sum(axis=(-1,-2)).mean()),
                    fk_error_max_m=float(err[:,i].max()))
                rows.append(row)
            # Additional conditional proxy for mixed contacts; labels unchanged.
            for mu in [.5,1.,1.5]:
                mix_force=result['wall_force'].copy()
                mix_force[...,0]-=np.where(labels==MIXED,mixed_wall_normal_lower_bound(z['foot_force'],mu),0.)
                ext=contact_joint_torque(J,mix_force)
                eff=np.where(ext*motor[:,:,None,:]<0,np.minimum(np.abs(ext),np.abs(motor[:,:,None,:])),0.)
                cond=(eff>3.)&slow&quiet[...,None]&blocked[:,:,None,None]
                flag=sustained(cond,dt,.2)
                for i,pol in enumerate(m['policies']):
                    mixed_rows.append(dict(case=c['label'],policy=pol['label'],lane=i,speed=c['speed'],tolerance_m=tol,friction_upper_bound=mu,
                        sustained_seconds=float(flag[a:b,i].any(axis=(-1,-2)).sum()*dt),
                        conditional_cost_Nm2=float((np.maximum(eff[a:b,i]-3.,0)**2*flag[a:b,i]).sum(axis=(-1,-2)).mean())))
                if tol==.008 and mu==1.5:
                    snapshot.update(mixed_wall_normal_proxy_force=mix_force,conditional_wall_external_joint_torque=ext,conditional_stall=flag)
            if tol==.008:
                snapshot.update(dict(labels=labels,wall_external_joint_torque=wall_tau,
                    support_external_joint_torque=support_tau,
                    opposed_actuator_effort_proxy=attributable,
                    sustained_wall_stall=active,diagnostic_cost=cost,
                    wall_geometry=result['wall_geometry'],ground_geometry=result['ground_geometry']))
        np.savez_compressed(args.output/(c['label']+'.npz'),**snapshot)
        inputs.append(dict(case=c['label'],samples_sha256=hashlib.sha256((d/'samples.npz').read_bytes()).hexdigest(),metadata_sha256=hashlib.sha256((d/'metadata.json').read_bytes()).hexdigest(),urdf_sha256=m['urdf_sha256']))
        expected+=3
    assert len(rows)==expected*3
    (args.output/'mixed_friction_results.json').write_text(json.dumps(mixed_rows,indent=2,allow_nan=False))
    (args.output/'results.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    (args.output/'validation.json').write_text(json.dumps(dict(policy_trials=expected,tolerance_variants=3,finite=True,max_fk_error_m=max_fk,inputs=inputs,parameters=dict(force_threshold_N=5,foot_radius_m=.02,joint_speed_threshold_rad_s=.2,foot_speed_threshold_m_s=.03,torque_threshold_Nm=3,hold_s=.2,progress_fraction=.2),source_hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(__file__).with_name('minich_contact_surface.py')]}),indent=2))
    lines=['# 足端接触表面与等效关节力矩判别试验','',
        '仅对已记录的精确 +X 楼梯回放分析；训练奖励与配置未修改。8个场景、24条策略轨迹，三个几何容差共72组分析。',
        '', '方法：URDF足端球半径20 mm，计算球心到有限踏面/立面的距离，默认容差8 mm；至少5 N接触力，踏面要求+Fz，立面要求−Fx。两种几何条件同时成立则标记mixed，禁止强行二分。与表面不匹配的有力接触标记unknown。',
        '', '力矩：用URDF的关节轴及足端位置计算世界系足端平移雅可比。纯踏面只投影+Fz，纯立面只投影−Fx；输出JᵀF为外部广义力矩，静态平衡执行器分量符号相反。mixed不分解。该值不等于执行器全部力矩，遗漏摩擦切向力、足端接触力偶、其他身体接触及动力学项。',
        '', '静止大力矩诊断：取执行器力矩与立面外部力矩符号相反且幅值的较小值作为出力代理；该值>3 Nm、关节速度<0.2 rad/s、足端世界速度<0.03 m/s，同时前进指令>0.1 m/s而机体速度绝对值小于指令20%，持续0.2秒。诊断代价为超3 Nm部分平方和，只是试验阈值，尚未标定训练权重。',
        '', '局限：净接触力无法唯一还原逐点接触法向和摩擦，纯立面Fx中也可能混入摩擦；mixed必须保留不确定。静止在棱边的情况可能漏检，因此诊断值是保守代理。接触数据按控制步记录，不能据此证明物理子步力矩分解精确。此实验没有独立逐接触点真值，以下比例是检测分布，不能称为准确率/召回率。',
        '', '验证：11项单元测试涵盖高摩擦地面、带竖直摩擦的立面、棱边、悬空、踏面/平台、错误力方向、网格外/背面、雅可比有限差分与虚功，以及混合接触摩擦下界（4000组可行合力）、持续时间触发/释放。回放FK与日志足端最大误差 %.6f mm。'% (max_fk*1000),
        '', '## 默认8 mm容差结果（每种策略三个位置取平均）','',
        '|指令 m/s|策略|纯立面占接触比例|混合占接触比例|持续受阻秒数|诊断代价 Nm²|',
        '|---|---|---|---|---|---|']
    for speed in [.2,.3,.5,1.]:
        for pol in ['baseline','latest']:
            rr=[r for r in rows if r['speed']==speed and r['policy']==pol and r['tolerance_m']==.008]
            mean=lambda k:float(np.mean([r[k] for r in rr]))
            lines.append('|%g|%s|%.1f%%|%.1f%%|%.3f|%.3f|'%(speed,pol,100*mean('pure_riser_fraction_of_contact'),100*mean('mixed_fraction_of_contact'),mean('sustained_wall_stall_seconds'),mean('diagnostic_cost_mean_Nm2')))
    lines+=['','## 几何容差敏感性','', '|容差 mm|策略|纯立面占接触比例|混合占接触比例|持续受阻秒数总和|','|---|---|---|---|---|']
    for tol in [.005,.008,.012]:
        for pol in ['baseline','latest']:
            rr=[r for r in rows if r['policy']==pol and r['tolerance_m']==tol]
            lines.append('|%g|%s|%.1f%%|%.1f%%|%.3f|'%(tol*1000,pol,100*np.mean([r['pure_riser_fraction_of_contact'] for r in rr]),100*np.mean([r['mixed_fraction_of_contact'] for r in rr]),sum(r['sustained_wall_stall_seconds'] for r in rr)))
    lines+=['','## 混合接触的摩擦下界补充试验','',
        '当同时可能接触踏面与立面时，假定两表面有效库仑摩擦系数都不超过mu，则立面法向力满足 Nwall >= max(0,(-Fx-mu*Fz)/(1+mu²))。这是条件下界，不能替代逐接触点真值；元数据只有地形摩擦配置，未记录实际材料组合后的系数，因此mu=0.5/1/1.5是敏感性试验，未宣称其为已验证物理上界。将该力沿−X映射为条件力矩代理并沿用静止阈值；此处力矩不是完整外部力矩的数学下界。', '',
        '|几何容差 mm|假定mu上界|上次持续受阻总秒数|最新持续受阻总秒数|',
        '|---|---|---|---|']
    for tol in [.005,.008,.012]:
        for mu in [.5,1.,1.5]:
            totals=[sum(r['sustained_seconds'] for r in mixed_rows if r['policy']==pol and r['tolerance_m']==tol and r['friction_upper_bound']==mu) for pol in ['baseline','latest']]
            lines.append('|%g|%g|%.3f|%.3f|'%(tol*1000,mu,*totals))
    lines+=['','## 结论','',
        '原型能区分几何明确的踏面与立面，并输出分开的等效关节负载；0.3 m/s最新策略纯立面接触比例10.1%，上次4.8%，混合接触分别41.7%和24.5%。但严格持续受阻检出对几何容差高度敏感（最新总计17/0.24/0.02秒，对应5/8/12 mm），摩擦下界没有消除该敏感性。因此当前原型适合诊断，尚不适合作为精准受阻惩罚直接接入训练；不能把上次零触发解释为已证明零误判。',
        '', '后续需记录实际接触点/法向、物理子步关节速度/力矩及有效材料摩擦；目前GPU净力日志不足以标定棱边力分解。任意训练地形需从真实碰撞网格查询表面，不能直接使用本例固定楼梯几何。',
        '', '引擎数据边界参考：[NVIDIA关于Isaac Gym净接触力与逐接触点限制的说明](https://forums.developer.nvidia.com/t/getting-contact-points-state-from-isaacgym-environment/193849)。',
        '', 'baseline = Oct23_19-43-17/model_500.pt；latest = Oct26_20-08-58/model_1000.pt。接触比例分母是有≥5N合力的足端样本；foot_seconds按足累加，四足可能同时计时。持续受阻秒数按整机任一关节触发计时。','','复现：','```bash','/opt/miniconda3/envs/gym4/bin/python legged_gym/legged_gym/scripts/analyze_minich_contact_surface.py --input legged_gym/logs/policy_eval_20261027_stairs_compare --output /tmp/minich_contact_surface_replay','/opt/miniconda3/envs/gym4/bin/python -m unittest discover -s legged_gym/legged_gym/tests -p test_minich_contact_surface.py -v','```']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[12:]))

if __name__=='__main__':main()

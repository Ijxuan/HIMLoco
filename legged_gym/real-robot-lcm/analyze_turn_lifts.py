#!/usr/bin/env python3
"""Offline kinematic foot-lift proxy; no contact truth or robot connection."""
import json,struct,csv
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation
from analyze_joint_log import events,decode
from detect_imu_turns import spans
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'record.lcm.00_analysis/turn_lifts'
OUT.mkdir(exist_ok=True)
legs=['FR','FL','RR','RL']; feedback=[]; states=[];origin=None
for _,ts,ch,p in events(ROOT/'record.lcm.00'):
    if origin is None:origin=ts
    t=(ts-origin)*1e-6
    if ch=='leg_control_data':
        d=decode(ch,p);feedback.append([t,*d['p'],*d['q']])
    elif ch=='state_estimator':
        if p[:8].hex()!='32a7a58b9685d099' or len(p)!=132:raise ValueError('state schema')
        states.append([t,*struct.unpack('>31f',p[8:])])
f=np.asarray(feedback);s=np.asarray(states); t=f[:,0]
idx=np.searchsorted(s[:,0],t);idx=np.clip(idx,0,len(s)-1)
# Published feedback immediately precedes state-estimator message in the cycle.
assert np.max(np.abs(s[idx,0]-t))<.005
q=s[idx,22:26] # state quat starts at float index21, plus time column
assert np.max(np.abs(np.linalg.norm(q,axis=1)-1))<.01
rot=Rotation.from_quat(q[:,[1,2,3,0]])
hip=np.array([[.20275,-.049,0],[.20275,.049,0],[-.20275,-.049,0],[-.20275,.049,0]])
p=f[:,1:13].reshape(-1,4,3);joint=f[:,13:].reshape(-1,4,3)
# Verify published p against the recording branch's forward kinematics.
a,b,c=joint.transpose(2,0,1);side=np.array([-1,1,-1,1]);l1=.072;l2=.2085;l3=.195
fk=np.stack([l3*np.sin(b+c)+l2*np.sin(b),l1*side*np.cos(a)+(l3*np.cos(b+c)+l2*np.cos(b))*np.sin(a),l1*side*np.sin(a)-(l3*np.cos(b+c)+l2*np.cos(b))*np.cos(a)],axis=-1)
fk_error=float(np.max(np.abs(fk-p)))
assert fk_error < 1e-5, "Published p disagrees with recording-branch FK"
z=np.einsum('ni,nfi->nf',rot.as_matrix()[:,2,:],p+hip)
# Stationary standing calibration removes fixed per-leg kinematic offsets only.
cal=(t>=19)&(t<20);base=np.median(z[cal],axis=0);offset=base-np.median(base)
z-=offset
# Two lowest feet define an instantaneous horizontal support-height proxy.
ground=np.sort(z,axis=1)[:,:2].mean(axis=1)
h=z-ground[:,None]
h=median_filter(h,size=(15,1),mode='nearest') # about30ms at500Hz
windows=[('negative',22.3,26.1),('positive',27.2,31.4)]
result={'method':{'height':'gravity-aligned p+hip, fixed standing offsets, subtract mean lowest two feet','standing_calibration_s':[19,20],'offsets_m':offset.tolist(),'thresholds_peak_mm':[5,10,15],'release_fraction':.5,'minimum_event_s':.05,'merge_gap_s':.04,'smooth_samples':15},'verification':{'max_pair_dt_s':float(np.max(abs(s[idx,0]-t))),'fk_max_abs_error_m':fk_error},'windows':[]}
all_events=[]
for name,start,end in windows:
    sel=(t>=start)&(t<=end);tt=t[sel];hh=h[sel]
    window={'name':name,'start_s':start,'end_s':end,'feet':{},'sensitivity':{}}
    for threshold in (.005,.010,.015):
        perfoot={};evs=[]
        for foot,leg in enumerate(legs):
            height=hh[:,foot];mask=height>=threshold*.5
            for a,b in spans(~mask):
                if a>0 and b<len(mask) and tt[b]-tt[a]<=.04:mask[a:b]=True
            detected=[]
            for a,b in spans(mask):
                finish=tt[min(b,len(tt)-1)];duration=finish-tt[a]
                if duration<.05 or height[a:b].max()<threshold:continue
                event={'window':name,'leg':leg,'start_s':float(tt[a]),'end_s':float(finish),'duration_s':float(duration),'peak_mm':float(height[a:b].max()*1000),'censored':bool(a==0 or b==len(mask))}
                detected.append(event);evs.append(event)
            full=[e for e in detected if not e['censored']]
            perfoot[leg]={'count_complete':len(full),'count_censored':len(detected)-len(full),'duration_median_s':float(np.median([e['duration_s'] for e in full])) if full else None,'peak_median_mm':float(np.median([e['peak_mm'] for e in full])) if full else None,'height_p95_mm':float(np.percentile(height,95)*1000),'events':detected}
        window['sensitivity'][str(round(threshold*1000))]=perfoot
        if threshold==.010:window['feet']=perfoot;all_events+=evs
    result['windows'].append(window)
result['peak_timing']={}
for win in result['windows']:
    selected=(t>=win['start_s'])&(t<=win['end_s']);tt=t[selected];hh=h[selected];peaks={}
    for j,leg in enumerate(legs):
        inds,_=find_peaks(hh[:,j],height=.01,prominence=.006,distance=125)
        peaks[leg]=tt[inds].tolist()
    pair=('FL','RR') if win['name']=='negative' else ('FR','RL')
    ref=np.array(peaks[pair[0]]);other=np.array(peaks[pair[1]])
    nearest=np.array([min(abs(other-x)) for x in ref]) if len(other) else np.array([])
    result['peak_timing'][win['name']]={'legs':pair,'peak_times_s':peaks,'nearest_peak_median_s':float(np.median(nearest)) if len(nearest) else None,'matched_within_80ms':int((nearest<=.08).sum()),'reference_peak_count':len(ref),'cycle_median_s':{leg:float(np.median(np.diff(values))) if len(values)>1 else None for leg,values in peaks.items()}}
(OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
with (OUT/'events.csv').open('w') as out:
    writer=csv.DictWriter(out,fieldnames=['window','leg','start_s','end_s','duration_s','peak_mm','censored']);writer.writeheader();writer.writerows(all_events)
np.savez_compressed(OUT/'samples.npz',time_s=t,height_proxy_m=h,gravity_aligned_z_m=z)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(4,2,figsize=(13,9),sharex='col',sharey=True)
for col,(name,start,end) in enumerate(windows):
    sel=(t>=start)&(t<=end)
    for foot,leg in enumerate(legs):
        ax=axes[foot,col];ax.plot(t[sel],h[sel,foot]*1000);ax.axhline(10,color='k',ls='--',lw=.7);ax.axhline(5,color='0.5',ls=':',lw=.7);ax.set_ylabel(leg+' height proxy (mm)');ax.grid(alpha=.3)
        for e in all_events:
            if e['window']==name and e['leg']==leg:ax.axvspan(e['start_s'],e['end_s'],alpha=.2,color='orange')
    axes[0,col].set_title(name);axes[-1,col].set_xlabel('Time since log start (s)')
fig.tight_layout();fig.savefig(OUT/'foot_lifts.png',dpi=150)
print(json.dumps(result['verification']));print('offset',offset)
for w in result['windows']:
 print(w['name'])
 for th,feet in w['sensitivity'].items():print(th,{l:{k:v for k,v in f.items() if k!='events'} for l,f in feet.items()})

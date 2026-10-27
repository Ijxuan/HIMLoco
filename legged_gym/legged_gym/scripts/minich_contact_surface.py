"""Offline prototype for the exact +X stair mesh, not arbitrary heightfields.

Net body forces cannot uniquely separate friction and multiple contacts. Geometry
therefore gates a conservative normal-force proxy; ambiguous edges stay mixed.
All positions, forces and translational Jacobians use the world frame.
"""
import xml.etree.ElementTree as ET
import numpy as np

AIR, SUPPORT, RISER, MIXED, UNKNOWN = range(5)
CLASS_NAMES = ('air', 'support', 'riser', 'mixed', 'unknown')


def classify_stair_contact(feet, forces, geometry, radius=.02, tolerance=.008,
                           force_threshold=5., y_bounds=(-4., 6.), ground_back=4.):
    """Return geometry/force evidence and exclusive labels, shape (..., feet).

    RISER requires -Fx, SUPPORT requires +Fz. A high friction force alone never
    creates a wall classification. MIXED includes top/bottom edges, where net
    force cannot be attributed uniquely. Force values are proxies, not a solve
    of individual contact normals. The staircase rises toward world +X.
    """
    pos=np.asarray(feet); f=np.asarray(forces)
    if pos.shape!=f.shape or pos.shape[-1]!=3 or not np.isfinite(pos).all() or not np.isfinite(f).all():
        raise ValueError('Require finite matching (...,3) positions and forces')
    if radius<=0 or tolerance<=0 or force_threshold<=0:
        raise ValueError('Require positive radius, tolerance and force threshold')
    x,y,z=np.moveaxis(pos,-1,0)
    start=geometry['STAIR_START_X_M']; depth=geometry['STAIR_DEPTH_M']
    height=geometry['STAIR_HEIGHT_M']; count=geometry['STAIR_COUNT']
    top=start+count*depth
    if depth<=0 or height<=0 or count<1: raise ValueError('Invalid stair geometry')
    ground_geom=np.zeros(x.shape,bool); wall_geom=ground_geom.copy()
    # Distance to finite horizontal patches and vertical patches in the XZ plane.
    # Reject the underside/back of these one-sided surfaces.
    for level in range(count+1):
        lo=-ground_back if level==0 else start+(level-1)*depth
        hi=start if level==0 else (geometry['TOP_LANDING_END_X_M'] if level==count else start+level*depth)
        dx=x-np.clip(x,lo,hi); dz=z-level*height
        ground_geom |= (np.abs(np.hypot(dx,dz)-radius)<=tolerance)&(dz>=-tolerance)
    for step in range(count):
        face=start+step*depth
        dx=x-face; dz=z-np.clip(z,step*height,(step+1)*height)
        wall_geom |= (np.abs(np.hypot(dx,dz)-radius)<=tolerance)&(dx<=tolerance)
    in_y=(y>=y_bounds[0]+radius)&(y<=y_bounds[1]-radius)
    ground_geom &= in_y; wall_geom &= in_y
    contact=np.linalg.norm(f,axis=-1)>=force_threshold
    ground=contact&ground_geom&(f[...,2]>=force_threshold)
    wall=contact&wall_geom&(-f[...,0]>=force_threshold)
    # Mark geometry ambiguity even if only one force component clears threshold.
    mixed=contact&ground_geom&wall_geom
    labels=np.where(contact,UNKNOWN,AIR).astype(np.int8)
    labels[ground&~wall_geom]=SUPPORT
    labels[wall&~ground_geom]=RISER
    labels[mixed]=MIXED
    wall_force=np.zeros_like(f); support_force=np.zeros_like(f)
    # Pure, high-confidence surface proxies only. Mixed contacts excluded.
    wall_force[...,0]=np.where(labels==RISER,np.minimum(f[...,0],0),0)
    support_force[...,2]=np.where(labels==SUPPORT,np.maximum(f[...,2],0),0)
    return dict(labels=labels,wall_force=wall_force,support_force=support_force,
                ground_geometry=ground_geom,wall_geometry=wall_geom,
                wall_evidence=wall,support_evidence=ground)


def contact_joint_torque(jacobian, force):
    """External generalized load J^T F (Nm); balancing actuator term is -J^T F.

    J shape (...,3,joints), force (...,3). No claim that this equals actuator
    torque: gravity, inertia, other contacts and foot contact moments are absent.
    """
    return np.einsum('...ij,...i->...j',jacobian,force)


def rotation(axis, angle):
    axis=np.asarray(axis,float);axis=axis/np.linalg.norm(axis)
    x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
    angle=np.asarray(angle)
    return np.eye(3)+np.sin(angle)[...,None,None]*K+(1-np.cos(angle))[...,None,None]*(K@K)


def quaternion_rotation(q):
    q=np.asarray(q,float);q=q/np.linalg.norm(q,axis=-1,keepdims=True)
    x,y,z,w=np.moveaxis(q,-1,0)
    return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
                     2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
                     2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(q.shape[:-1]+(3,3))


class UrdfFootKinematics:
    """URDF chain FK and geometric Jacobian about the foot link origin.

    Supports fixed/revolute joints, URDF origin xyz/rpy and arbitrary axes.
    Root poses are poses of the root URDF link, xyzw quaternion convention.
    """
    def __init__(self, path, dof_names, foot_names):
        tree=ET.parse(path).getroot(); self.dof_names=list(dof_names); self.foot_names=list(foot_names)
        parents={}
        for joint in tree.findall('joint'):
            origin=joint.find('origin'); axis=joint.find('axis')
            vec=lambda key: np.fromstring(origin.get(key,'0 0 0'),sep=' ') if origin is not None else np.zeros(3)
            xyz=vec('xyz');r,p,y=vec('rpy')
            R=rotation([0,0,1],y)@rotation([0,1,0],p)@rotation([1,0,0],r)
            kind=joint.get('type')
            if kind not in ('fixed','revolute','continuous'):raise ValueError('Unsupported joint '+kind)
            name=joint.get('name'); child=joint.find('child').get('link'); parent=joint.find('parent').get('link')
            parents[child]=(parent,xyz,R,np.fromstring(axis.get('xyz'),sep=' ') if axis is not None else np.array([1.,0,0]),None if kind=='fixed' else self.dof_names.index(name))
        self.chains=[]
        for foot in foot_names:
            chain=[];link=foot
            while link in parents:
                entry=parents[link];chain.append(entry);link=entry[0]
            self.chains.append(chain[::-1])
        self.radii=[]
        for foot in foot_names:
            sphere=tree.find("./link[@name='%s']/collision/geometry/sphere"%foot)
            if sphere is None:raise ValueError('Require spherical foot collision')
            self.radii.append(float(sphere.get('radius')))

    def evaluate(self, q, root):
        q=np.asarray(q);root=np.asarray(root)
        R0=quaternion_rotation(root[...,3:7]);positions=[];jacobians=[]
        for chain in self.chains:
            R=R0.copy();pos=root[...,:3].copy();axes=[];origins=[];indices=[]
            for _,offset,origin_R,axis,index in chain:
                pos=pos+np.einsum('...ij,j->...i',R,offset);R=R@origin_R
                if index is not None:
                    axes.append(np.einsum('...ij,j->...i',R,axis));origins.append(pos.copy());indices.append(index)
                    R=R@rotation(axis,q[...,index])
            J=np.zeros(q.shape[:-1]+(3,len(self.dof_names)))
            for axis,origin,index in zip(axes,origins,indices):J[...,index]=np.cross(axis,pos-origin)
            positions.append(pos);jacobians.append(J)
        return np.stack(positions,axis=-2),np.stack(jacobians,axis=-3)


def mixed_wall_normal_lower_bound(force, friction_upper_bound):
    """Necessary wall-normal force for a ground+wall contact, in newtons.

    D=-Fx=Nwall+ground_tangent_x; U=Fz=Nground+wall_tangent_z.
    |ground_tangent_x|<=mu*Nground, |wall_tangent_z|<=mu*Nwall
    imply Nwall >= max(0,(D-mu*U)/(1+mu**2)).
    Only valid for +Z ground / -X wall normals and a valid upper bound on both
    effective Coulomb coefficients; it is NOT a unique contact decomposition.
    Apply only after geometry confirms a possible two-surface contact.
    """
    f=np.asarray(force)
    if not np.isfinite(f).all() or not np.isfinite(friction_upper_bound) or friction_upper_bound<0:
        raise ValueError('Require finite force and nonnegative friction bound')
    mu=friction_upper_bound
    return np.maximum(0.,(-f[...,0]-mu*f[...,2])/(1+mu*mu))

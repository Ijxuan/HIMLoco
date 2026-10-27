"""Surface discrimination and virtual-work checks without importing Isaac Gym."""
import importlib.util
from pathlib import Path
import unittest
import sys
sys.path.insert(0,str(Path(__file__).parents[1]/"scripts"))
from analyze_minich_contact_surface import sustained
import numpy as np

path=Path(__file__).parents[1]/'scripts/minich_contact_surface.py'
spec=importlib.util.spec_from_file_location('surface',path)
surface=importlib.util.module_from_spec(spec);spec.loader.exec_module(surface)
G=dict(STAIR_START_X_M=.5,STAIR_DEPTH_M=.2,STAIR_HEIGHT_M=.08,STAIR_COUNT=20,TOP_LANDING_END_X_M=15.)

class SurfaceTests(unittest.TestCase):
    def classify(self,p,f):return surface.classify_stair_contact(np.array(p,float),np.array(f,float),G)

    def test_ground_friction_is_not_wall(self):
        r=self.classify([[.2,0,.02]],[[-100,0,40]])
        self.assertEqual(r['labels'][0],surface.SUPPORT)
        self.assertFalse(r['wall_force'].any())

    def test_vertical_face_and_upward_friction(self):
        r=self.classify([[.48,0,.04]],[[-80,0,40]])
        self.assertEqual(r['labels'][0],surface.RISER)
        np.testing.assert_equal(r['wall_force'],[[-80,0,0]])
        self.assertFalse(r['support_force'].any())

    def test_edge_is_explicitly_ambiguous(self):
        r=self.classify([[.48,0,.02]],[[-80,0,40]])
        self.assertEqual(r['labels'][0],surface.MIXED)
        self.assertFalse(r['wall_force'].any())

    def test_air_and_unexplained_force(self):
        r=self.classify([[.2,0,.3],[.2,0,.3]],[[0,0,0],[-80,0,0]])
        np.testing.assert_equal(r['labels'],[surface.AIR,surface.UNKNOWN])

    def test_tread_and_top_landing(self):
        r=self.classify([[.6,0,.10],[5.,0,1.62]],[[0,0,40],[0,0,40]])
        np.testing.assert_equal(r['labels'],[surface.SUPPORT,surface.SUPPORT])

    def test_wall_rejects_wrong_force_direction(self):
        r=self.classify([[.48,0,.04]],[[80,0,0]])
        self.assertEqual(r['labels'][0],surface.UNKNOWN)

    def test_outside_mesh_and_wrong_side(self):
        r=self.classify([[.48,8,.04],[.52,0,.04]],[[-80,0,0],[-80,0,0]])
        np.testing.assert_equal(r['labels'],[surface.UNKNOWN,surface.UNKNOWN])

    def test_mixed_friction_lower_bound(self):
        # All horizontal load explainable by ground friction: no wall claim.
        self.assertEqual(surface.mixed_wall_normal_lower_bound([-30,0,40],1.),0.)
        self.assertEqual(surface.mixed_wall_normal_lower_bound([-100,0,40],1.),30.)
        # Necessary lower bound holds for feasible random normal/tangent loads.
        rng=np.random.default_rng(1)
        for mu in [0.,.5,1.,1.5]:
            ng=rng.uniform(0,100,1000);nw=rng.uniform(0,100,1000)
            tg=rng.uniform(-1,1,1000)*mu*ng;tw=rng.uniform(-1,1,1000)*mu*nw
            f=np.stack([-nw-tg,np.zeros(1000),ng+tw],axis=-1)
            lower=surface.mixed_wall_normal_lower_bound(f,mu)
            self.assertTrue((lower<=nw+1e-10).all())

    def test_short_contact_and_reset_do_not_trigger(self):
        mask=np.array([[True],[True],[False],[True],[False]])
        self.assertFalse(sustained(mask,.02,.06).any())

    def test_sustained_contact_activation_and_release(self):
        mask=np.array([[True],[True],[True],[True],[False],[True]])
        np.testing.assert_equal(sustained(mask,.02,.06)[:,0],
                                [False,False,True,True,False,False])

    def test_jacobian_and_virtual_work(self):
        import xml.etree.ElementTree as ET
        urdf=Path(__file__).parents[2]/'resources/robots/mini_cheetah/urdf/mini_cheetah.urdf'
        dofs=[j.get('name') for j in ET.parse(urdf).getroot().findall('joint') if j.get('type')=='revolute']
        feet=['FL_foot','FR_foot','RL_foot','RR_foot'];kin=surface.UrdfFootKinematics(urdf,dofs,feet)
        rng=np.random.default_rng(42);q=rng.normal(size=(5,12))*.3
        root=np.zeros((5,13));root[:,6]=1;root[:,2]=.3
        # Non-identity world rotation catches frame mistakes.
        root[:,3:7]=[0,0,np.sin(.4),np.cos(.4)]
        pos,J=kin.evaluate(q,root);eps=1e-6
        for j in range(12):
            qp=q.copy();qm=q.copy();qp[:,j]+=eps;qm[:,j]-=eps
            numeric=(kin.evaluate(qp,root)[0]-kin.evaluate(qm,root)[0])/(2*eps)
            np.testing.assert_allclose(J[...,j],numeric,atol=2e-9)
        F=rng.normal(size=pos.shape);dq=rng.normal(size=q.shape)
        tau=surface.contact_joint_torque(J,F)
        velocity=np.einsum('...fij,...j->...fi',J,dq)
        np.testing.assert_allclose((tau*dq[:,None]).sum(-1),(F*velocity).sum(-1),atol=1e-12)

if __name__=='__main__':unittest.main()

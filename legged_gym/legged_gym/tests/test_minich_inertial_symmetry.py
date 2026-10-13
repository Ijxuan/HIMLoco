"""Validate leg inertial parameters without loading Isaac Gym.

Physical realizability is audited separately: the existing base and thigh
tensors violate the principal-moment triangle inequality and need CAD data.
"""
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import numpy as np


URDF = Path(__file__).resolve().parents[2] / "resources/robots/mini_cheetah/urdf/mini_cheetah.urdf"


def tensor(inertial):
    a = {key: float(value) for key, value in inertial.find("inertia").attrib.items()}
    return np.array([[a['ixx'], a['ixy'], a['ixz']],
                     [a['ixy'], a['iyy'], a['iyz']],
                     [a['ixz'], a['iyz'], a['izz']]])


class TestMiniChInertialSymmetry(unittest.TestCase):
    def test_leg_mirror_and_positive_definiteness(self):
        links = {link.get('name'): link for link in ET.parse(URDF).getroot().findall('link')}
        reflection = np.diag([1., -1., 1.])
        for end in ('F', 'R'):
            for part in ('hip', 'thigh', 'calf', 'foot'):
                with self.subTest(end=end, part=part):
                    left, right = [links[f'{end}{side}_{part}'].find('inertial') for side in ('L', 'R')]
                    for inertial in (left, right):
                        # These URDF inertial axes coincide with their link axes.
                        np.testing.assert_allclose(np.fromstring(inertial.find('origin').get('rpy', '0 0 0'), sep=' '), 0)
                        moments = np.linalg.eigvalsh(tensor(inertial))
                        self.assertGreater(moments[0], 0)
                    self.assertEqual(float(left.find('mass').get('value')), float(right.find('mass').get('value')))
                    com_left, com_right = [np.fromstring(i.find('origin').get('xyz'), sep=' ') for i in (left, right)]
                    np.testing.assert_allclose(com_right, reflection @ com_left, atol=1e-12)
                    np.testing.assert_allclose(tensor(right), reflection @ tensor(left) @ reflection, atol=1e-12)


if __name__ == '__main__':
    unittest.main()

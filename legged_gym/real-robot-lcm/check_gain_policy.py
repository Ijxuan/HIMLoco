"""Run the established policy/load diagnostic with explicit simulation gains."""
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('--calf_kp', type=float, required=True)
parser.add_argument('--calf_kd', type=float, required=True)
gains, remaining = parser.parse_known_args()
sys.argv = [sys.argv[0]] + remaining
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'legged_gym/scripts'))
import measure_minich_joint_loads as diagnostic

cfg, _ = diagnostic.task_registry.get_cfgs('minich')
cfg.control.stiffness = {'joint': 17.0, 'calf_joint': gains.calf_kp}
cfg.control.damping = {'joint': 0.4, 'calf_joint': gains.calf_kd}
diagnostic.main()

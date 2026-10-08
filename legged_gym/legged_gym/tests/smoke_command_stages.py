"""16-environment, 12-second command scheduling smoke test; no training."""
import json
import math
from pathlib import Path

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
import torch


def main():
    args = get_args()
    args.task, args.num_envs, args.seed = 'minich', 16, 2026
    cfg, _ = task_registry.get_cfgs('minich')
    cfg.env.num_envs = 16
    cfg.env.episode_length_s = 30.
    cfg.terrain.mesh_type = 'plane'
    cfg.terrain.curriculum = False
    cfg.noise.add_noise = False
    for name in dir(cfg.domain_rand):
        if isinstance(getattr(cfg.domain_rand, name), bool):
            setattr(cfg.domain_rand, name, False)
    cfg.commands.curriculum = False
    cfg.commands.enable_turn_to_stand = True
    cfg.commands.in_place_turn_probability = 1.
    cfg.commands.turn_to_stand_probability = 1.
    env, _ = task_registry.make_env(name='minich', args=args, env_cfg=cfg)
    print('Environment source:', __import__('inspect').getfile(type(env)), flush=True)
    env.reset()
    policy = torch.jit.load(str(Path(__file__).resolve().parents[2] /
        'logs/rough_minich/Sep18_18-36-28_smoke_minich/exported/policies/model_1000.jit'),
        map_location=env.device).eval()
    resets = []
    turns_to_stand = [0] * 16
    stands_to_turn = [0] * 16
    successful = [False] * 16
    obs = env.get_observations()
    with torch.no_grad():
        for step in range(math.ceil(12. / env.dt)):
            previous = env.command_stage.clone()
            obs, _, _, _, _, reset_ids, _ = env.step(policy(obs))
            reset_set = set(reset_ids.cpu().tolist())
            for i in range(16):
                if i in reset_set:
                    resets.append({'step': step + 1, 'env': i})
                    turns_to_stand[i] = stands_to_turn[i] = 0
                    continue
                if previous[i] == env.COMMAND_TURN and env.command_stage[i] == env.COMMAND_STAND:
                    turns_to_stand[i] += 1
                if previous[i] == env.COMMAND_STAND and env.command_stage[i] == env.COMMAND_TURN:
                    stands_to_turn[i] += 1
                    successful[i] = successful[i] or turns_to_stand[i] > 0
            stand = env.command_stage == env.COMMAND_STAND
            assert torch.all(env.commands[stand, :3] == 0)
    result = {'simulated_seconds': 12., 'environments': 16, 'seed': 2026,
              'reset_count': len(resets), 'resets': resets,
              'uninterrupted_turn_stand_turn_seen': successful,
              'note': 'Old policy used to exercise scheduler, not validate newly trained stopping.'}
    print(json.dumps(result), flush=True)
    assert all(successful), 'Some environments never completed an uninterrupted turn/stand/turn sequence'
    env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()

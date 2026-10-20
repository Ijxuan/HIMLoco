"""Event-weighted gait diagnostics from reset-batch sufficient statistics."""
import torch


def summarize_gait_episode_metrics(ep_infos):
    groups = {}
    names = {'gait_height': 'height', 'gait_retraction': 'retraction', 'retraction_other': 'retraction_other'}
    for info in ep_infos:
        for key, value in info.items():
            parts = key.split('/')
            if len(parts) == 3 and parts[0] in names:
                group = f'Gait/{names[parts[0]]}/{parts[1]}'
                groups.setdefault(group, {}).setdefault(parts[2], []).append(value)
    output = {}
    for group, metrics in groups.items():
        totals = {}
        for key, values in metrics.items():
            device = next((v.device for v in values if isinstance(v, torch.Tensor)), 'cpu')
            totals[key] = torch.stack([torch.as_tensor(v, device=device).sum() for v in values]).sum().item()
        # Totals make missing/rare events visible. Never average per-batch ratios.
        output.update({f'{group}/{key}': value for key, value in totals.items()})
        count = totals.get('count', 0)
        if count > 0:
            for key in ('under', 'over', 'in_band', 'short', 'valid_landing'):
                if key in totals:
                    output[f'{group}/{key}_rate'] = totals[key] / count
            for key, label in [('peak_sum_cm', 'mean_peak_cm'),
                               ('world_peak_sum_cm', 'mean_world_peak_cm'),
                               ('effective_peak_sum_cm', 'mean_effective_peak_cm'),
                               ('body_backward_sum_cm', 'mean_body_backward_cm'),
                               ('score_sum', 'mean_raw_event_score')]:
                if key in totals:
                    output[f'{group}/{label}'] = totals[key] / count
        started = totals.get('started', 0)
        if started > 0:
            for key in ('timeout', 'cancelled'):
                if key in totals:
                    output[f'{group}/{key}_per_started'] = totals[key] / started
        seconds = totals.get('enabled_seconds', 0)
        if seconds > 0:
            if 'weighted_score_sum' in totals:
                output[f'{group}/reward_per_enabled_second'] = totals['weighted_score_sum'] / seconds
            output[f'{group}/events_per_enabled_second'] = count / seconds
            for key, value in totals.items():
                if key.endswith('_reward_sum'):
                    output[f'{group}/{key[:-4]}_per_enabled_second'] = value / seconds
    return output

"""Apply the preregistered pilot gate at the fixed training endpoint."""
from .guard import total_variation


def compare_pilot(frozen, adapted, behavior, cfg):
    task = frozen['task']
    if frozen['prompt'] != adapted['prompt'] or adapted['task'] != task:
        raise ValueError('pilot scores must share the task and prompt')
    if {name: row['n'] for name,row in frozen['modalities'].items()} != {
            name:row['n'] for name,row in adapted['modalities'].items()}:
        raise ValueError('pilot validation counts differ')
    a,b = frozen['modalities'][task],adapted['modalities'][task]
    predicted_a = {letter:a['predicted_marginal'][letter] for letter in 'ABCD'}
    predicted_b = {letter:b['predicted_marginal'][letter] for letter in 'ABCD'}
    tv = total_variation(predicted_a,predicted_b)
    gains = {'greedy_vs_greedy':b['greedy']-a['greedy'],'greedy_vs_maj16':b['greedy']-a['maj16'],
             'maj16':b['maj16']-a['maj16'],'pass16':b['pass16']-a['pass16'],
             'other_greedy':adapted['other_modalities_mean_greedy']-frozen['other_modalities_mean_greedy']}
    checks = {'greedy_vs_greedy':gains['greedy_vs_greedy'] >= cfg['min_greedy_gain_over_frozen_greedy'],
              'greedy_vs_maj16':gains['greedy_vs_maj16'] >= cfg['min_greedy_gain_over_frozen_maj16'],
              'prediction_tv':tv is not None and tv <= cfg['max_prediction_tv'],
              'pass16':gains['pass16'] >= -cfg['max_pass16_drop'],
              'other_greedy':gains['other_greedy'] >= -cfg['max_other_tasks_mean_accuracy_drop'],
              'brake':not behavior['brake_stop']}
    return {'passed':all(checks.values()),'checks':checks,'gains':gains,'prediction_tv':tv,
            'supplemental_maj16':gains['maj16'] >= cfg['supplemental_maj16_gain']}

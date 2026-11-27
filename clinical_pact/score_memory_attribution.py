"""Pair constrained MedBook decisions on replay and non-replay training rows."""
import json
import os
from pathlib import Path

from .score_retained_checkpoints import transitions, write_json


def evaluate(destination, protocol, rank):
    import torch

    from .answer_decisions import choice_logits, choice_token_ids, choice_view
    from .data import load_split
    from .semantic_replay import known_views
    from .validate_development import load_checkpoint

    root = Path(os.environ['ROOT'])
    saved = Path(os.environ['REFERENCE'])
    checkpoint = protocol['checkpoints'][rank]
    known = known_views(saved)
    trial = root / protocol['reference_run']
    retained = json.loads((trial / 'private' / 'reference.json').read_text())
    indices = sorted(map(int, retained))
    if len(indices) != 582 or any(indices[i + 1] != indices[i] + 1 or indices[i] % 2 for i in range(0, 582, 2)):
        raise ValueError('expected 291 complete pairs of calibrated answer views')
    training = list(load_split(root / 'data/prepared/MedicalVQA_MedBook', 'train'))
    seen = {item['id'] for item, _ in known}
    train_ids = {item['id'] for item in training}
    if len(training) != len(train_ids) or len(known) != 2 * len(seen) or not seen <= train_ids:
        raise ValueError('historical references do not match the prepared training split')
    never = [item for item in training if item['id'] not in seen]
    if len(never) != protocol['common_nonmemory_n']:
        raise ValueError('non-memory denominator differs from protocol')
    sets = {'retained': [known[index] for index in indices],
            'never_replayed': [choice_view(item, rotation) for item in never for rotation in (0, 1)]}
    model, processor = load_checkpoint(root / checkpoint['path'], rank, protocol['seed'])
    candidates = choice_token_ids(processor.tokenizer)
    data = {}
    with torch.inference_mode():
        for name, views in sets.items():
            correct = []
            for i, (item, target) in enumerate(views, 1):
                scores = choice_logits(model, processor, item, rank, candidates)
                correct.append(int(int(scores.argmax()) == target))
                if i % 200 == 0 or i == len(views):
                    write_json(destination / f'worker{rank}_progress.json', {'set': name, 'views': i})
            data[name] = correct
    write_json(destination / 'private' / f'worker{rank}.json', data)


def aggregate(destination, protocol):
    status = json.loads((destination / 'status.json').read_text())
    if status['reason'] != 'completed' or status['codes'] != [0] * protocol['worker_count']:
        raise ValueError('memory attribution did not complete')
    checkpoints = [checkpoint['name'] for checkpoint in protocol['checkpoints']]
    rows = {name: json.loads((destination / 'private' / f'worker{rank}.json').read_text())
            for rank, name in enumerate(checkpoints)}
    start = rows[checkpoints[0]]
    if sum(start['retained']) != 582:
        raise ValueError('retained anchor scoring differs from original calibration')
    report = {}
    for name, sets in rows.items():
        report[name] = {}
        for cohort, data in sets.items():
            expected = 582 if cohort == 'retained' else 2 * protocol['common_nonmemory_n']
            if len(data) != expected:
                raise ValueError('incomplete or duplicate-scored cohort')
            initial = start[cohort]
            both = [min(data[i:i+2]) for i in range(0, len(data), 2)]
            initial_both = [min(initial[i:i+2]) for i in range(0, len(initial), 2)]
            report[name][cohort] = {'views': len(data), 'view_correct': sum(data),
                                    'questions': len(both), 'both_correct': sum(both),
                                    'view_transitions': transitions(initial, data),
                                    'both_transitions': transitions(initial_both, both)}
    write_json(destination / 'aggregate.json', {'status': status, 'checkpoints': report})


if __name__ == '__main__':
    directory = Path(os.environ['OUT']) / 'validation'
    config = json.loads((directory / 'protocol.json').read_text())
    if 'RANK' in os.environ:
        evaluate(directory, config, int(os.environ['RANK']))
    else:
        aggregate(directory, config)

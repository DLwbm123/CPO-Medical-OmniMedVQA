"""Single-example validation of retained checkpoints, with private correctness bits."""
from collections import Counter
import json
import os
from pathlib import Path


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def transitions(before, after):
    masks = ('wrong', 'correct')
    pairs = Counter((first, second) for first, second in zip(before, after, strict=True))
    return {f'{masks[first]}_to_{masks[second]}': pairs[first, second]
            for first in range(2) for second in range(2)}


def evaluate(destination, protocol, rank):
    import torch
    from .data import load_split
    from .validate_development import load_checkpoint, predict_answer

    root = Path(os.environ['ROOT'])
    selected = protocol['checkpoints'][rank]
    control = protocol['checkpoints'][1]
    seed = protocol['seed']
    token_limit = protocol['max_new_tokens']
    data = root / 'data' / 'prepared'
    task, path_task = protocol['tasks']
    medbook = list(load_split(data / task['name'], 'validation'))
    if len(medbook) != task['n'] or len({item['id'] for item in medbook}) != task['n']:
        raise ValueError('MedBook validation differs from the frozen denominator')

    model, processor = load_checkpoint(root / selected['path'], rank, seed)
    medbook_result = {'checkpoint': selected['name'], 'n': len(medbook)}
    bits = {}
    for mode in ('image', 'text_only'):
        counts = Counter(correct=0, generated_tokens=0, eos_terminated=0)
        bits[mode] = []
        for item in medbook:
            result = predict_answer(model, processor, item, rank, token_limit, include_image=(mode == 'image'))
            counts.update(result)
            bits[mode].append(result['correct'])
        medbook_result[mode] = dict(counts)
    write_json(destination / 'private' / f'worker{rank}_correctness.json', bits)
    write_json(destination / f'worker{rank}_medbook.json', medbook_result)

    if selected != control:
        del model, processor
        torch.cuda.empty_cache()
        model, processor = load_checkpoint(root / control['path'], rank, seed)
    path_vqa = list(load_split(data / path_task['name'], 'validation'))
    if len(path_vqa) != path_task['n'] or len({item['id'] for item in path_vqa}) != path_task['n']:
        raise ValueError('PathVQA validation differs from the frozen denominator')
    shards = protocol['worker_count']
    counts = Counter(correct=0, generated_tokens=0, eos_terminated=0, n=0)
    assigned = path_vqa[rank::shards]
    for i, item in enumerate(assigned, start=1):
        counts.update(predict_answer(model, processor, item, rank, token_limit))
        counts['n'] += 1
        if i % 200 == 0 or i == len(assigned):
            write_json(destination / f'worker{rank}_pathvqa.json', counts)


def aggregate(destination, protocol):
    status = json.loads((destination / 'status.json').read_text())
    if status['reason'] != 'completed' or status['codes'] != [0] * protocol['worker_count']:
        raise ValueError('workers did not all complete')
    medbook = {}
    bits = {}
    path_counts = Counter()
    path_n = protocol['tasks'][1]['n']
    shards = protocol['worker_count']
    for rank in range(shards):
        metric = json.loads((destination / f'worker{rank}_medbook.json').read_text())
        checkpoint = protocol['checkpoints'][rank]['name']
        if metric['checkpoint'] != checkpoint or metric['n'] != protocol['tasks'][0]['n']:
            raise ValueError('MedBook checkpoint/validation pairing differs from protocol')
        medbook[checkpoint] = metric
        bits[checkpoint] = json.loads((destination / 'private' / f'worker{rank}_correctness.json').read_text())
        for mode, values in bits[checkpoint].items():
            if len(values) != metric['n'] or not set(values) <= {0, 1} or sum(values) != metric[mode]['correct']:
                raise ValueError('private correctness does not reconcile with MedBook aggregates')
        path = json.loads((destination / f'worker{rank}_pathvqa.json').read_text())
        if path['n'] != len(range(rank, path_n, shards)):
            raise ValueError('incomplete PathVQA shard')
        path_counts.update(path)
    initial = protocol['checkpoints'][0]['name']
    paired = {}
    for checkpoint, correct in bits.items():
        paired[checkpoint] = {'image_to_text_only': transitions(correct['image'], correct['text_only'])}
        if checkpoint != initial:
            paired[checkpoint]['initial_to_final_with_image'] = transitions(bits[initial]['image'], correct['image'])
    write_json(destination / 'aggregate.json', {
        'status': status, 'medbook': medbook, 'medbook_transitions': paired,
        'pathvqa_plain_gspo': dict(path_counts),
    })


if __name__ == '__main__':
    destination = Path(os.environ['OUT']) / 'validation'
    protocol = json.loads((destination / 'protocol.json').read_text())
    if 'RANK' in os.environ:
        evaluate(destination, protocol, int(os.environ['RANK']))
    else:
        aggregate(destination, protocol)

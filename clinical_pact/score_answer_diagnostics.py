"""Frozen paired validation for answer-format and option-order diagnostics."""
from collections import Counter
import json
import os
from pathlib import Path

from .answer_decisions import LABELS, answer_candidate, choice_logits, choice_token_ids, choice_view
from .score_retained_checkpoints import transitions, write_json
from .scoring import normalize


def evaluate(destination, protocol, rank):
    import torch

    from .data import load_split
    from .validate_development import generate_answer, load_checkpoint

    root = Path(os.environ['ROOT'])
    checkpoint = protocol['checkpoints'][rank]
    model, processor = load_checkpoint(root / checkpoint['path'], rank, protocol['seed'])
    ids = choice_token_ids(processor.tokenizer)
    report = {'checkpoint': checkpoint['name'], 'subsets': {}}
    private = {}
    for task in protocol['tasks']:
        rows = list(load_split(root / 'data/prepared' / task['name'], 'validation'))
        if len(rows) != task['n'] or len({item['id'] for item in rows}) != task['n']:
            raise ValueError('unexpected validation cardinality')
        for i, item in enumerate(rows, 1):
            text, scored = generate_answer(model, processor, item, rank, protocol['max_new_tokens'])
            candidate = answer_candidate(text)
            if task['name'] == 'MedicalVQA_MedBook':
                group = 'medbook'
                scored['valid_answer_type'] = int(candidate.upper() in tuple(LABELS))
                with torch.inference_mode():
                    predictions = []
                    for rotation in (0, 1):
                        view, target = choice_view(item, rotation)
                        logits = choice_logits(model, processor, view, rank, ids)
                        predictions.append(int(logits.argmax()) == target)
                scored.update(forced_correct=int(predictions[0]), rotated_correct=int(predictions[1]),
                              both_views_correct=int(all(predictions)))
            else:
                group = 'pathvqa_yes_no' if normalize(item['answer']) in ('yes', 'no') else 'pathvqa_other'
                if group == 'pathvqa_yes_no':
                    scored['valid_answer_type'] = int(candidate in ('yes', 'no'))
            scored['n'] = 1
            report['subsets'].setdefault(group, Counter()).update(scored)
            private.setdefault(group, []).append(scored)
            if i % 200 == 0 or i == len(rows):
                write_json(destination / f'worker{rank}_progress.json', report)
    write_json(destination / 'private' / f'worker{rank}.json', private)
    write_json(destination / f'worker{rank}.json', report)


def aggregate(destination, protocol):
    status = json.loads((destination / 'status.json').read_text())
    ranks = range(protocol['worker_count'])
    if status['reason'] != 'completed' or status['codes'] != [0] * len(ranks):
        raise ValueError('diagnostic workers did not complete')
    reports, records = {}, {}
    for rank in ranks:
        model = json.loads((destination / f'worker{rank}.json').read_text())
        report = model['subsets']
        if [report[group]['n'] for group in ('medbook', 'pathvqa_yes_no', 'pathvqa_other')] != [130, 3125, 3134]:
            raise ValueError('diagnostic subset size changed')
        key = model['checkpoint']
        reports[key] = report
        records[key] = json.loads((destination / 'private' / f'worker{rank}.json').read_text())
    paired = {}
    for key in ('plain_gspo_final', 'pact_med_final'):
        paired[key] = {}
        for name, metric in (('medbook', 'correct'), ('medbook', 'forced_correct'),
                             ('medbook', 'both_views_correct'), ('pathvqa_yes_no', 'correct'),
                             ('pathvqa_other', 'correct')):
            start = records['shared_medbook_start'][name]
            final = records[key][name]
            paired[key][f'{name}:{metric}'] = transitions([r[metric] for r in start], [r[metric] for r in final])
    write_json(destination / 'aggregate.json', {'status': status, 'subsets': reports, 'paired_from_start': paired})


if __name__ == '__main__':
    destination = Path(os.environ['OUT']) / 'validation'
    protocol = json.loads((destination / 'protocol.json').read_text())
    if 'RANK' in os.environ:
        evaluate(destination, protocol, int(os.environ['RANK']))
    else:
        aggregate(destination, protocol)

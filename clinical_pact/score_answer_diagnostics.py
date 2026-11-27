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
    if protocol.get('sharded_checkpoint'):
        from .semantic_replay import known_views

        run = Path(os.environ['OUT'])
        known = known_views(Path(os.environ['REFERENCE']))
        reference = json.loads((run / 'private' / 'reference.json').read_text())
        audit = Counter(n=0, flips=0)
        with torch.inference_mode():
            for index in list(reference)[rank::protocol['worker_count']]:
                view, target = known[int(index)]
                logits = choice_logits(model, processor, view, rank, ids)
                audit.update(n=1, flips=int(int(logits.argmax()) != target))
        report['memory_audit'] = dict(audit)
    for task in protocol['tasks']:
        rows = list(load_split(root / 'data/prepared' / task['name'], 'validation'))
        if len(rows) != task['n'] or len({item['id'] for item in rows}) != task['n']:
            raise ValueError('unexpected validation cardinality')
        assigned = list(enumerate(rows))
        if protocol.get('sharded_checkpoint'):
            assigned = assigned[rank::protocol['worker_count']]
        for i, (row_index, item) in enumerate(assigned, 1):
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
            private.setdefault(group, []).append({**scored, 'row_index': row_index})
            if i % 200 == 0 or i == len(assigned):
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


def aggregate_shards(destination, protocol):
    reports = {}
    metrics = {}
    memory = Counter()
    for rank in range(protocol['worker_count']):
        summary = json.loads((destination / f'worker{rank}.json').read_text())
        memory.update(summary['memory_audit'])
        for key, counter in summary['subsets'].items():
            reports.setdefault(key, Counter()).update(counter)
        rows = json.loads((destination / 'private' / f'worker{rank}.json').read_text())
        for key, values in rows.items():
            metrics.setdefault(key, []).extend(values)
    for key, n in [('medbook', 130), ('pathvqa_yes_no', 3125), ('pathvqa_other', 3134)]:
        if reports[key]['n'] != n or len({record['row_index'] for record in metrics[key]}) != n:
            raise ValueError('missing or duplicate validation rows')
        metrics[key].sort(key=lambda record: record['row_index'])
    write_json(destination / 'private' / 'merged.json', metrics)
    expected = json.loads((Path(os.environ['OUT']) / 'calibration.json').read_text())['eligible_questions'] * 2
    if memory['n'] != expected:
        raise ValueError('incomplete final memory audit')
    write_json(destination / 'aggregate.json', {'subsets': reports, 'memory_audit': dict(memory)})


if __name__ == '__main__':
    destination = Path(os.environ.get('EVAL_DIR', Path(os.environ['OUT']) / 'validation'))
    protocol = json.loads((destination / 'protocol.json').read_text())
    if 'RANK' in os.environ:
        evaluate(destination, protocol, int(os.environ['RANK']))
    elif protocol.get('sharded_checkpoint'):
        aggregate_shards(destination, protocol)
    else:
        aggregate(destination, protocol)

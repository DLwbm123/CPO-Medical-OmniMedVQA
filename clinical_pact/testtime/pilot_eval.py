"""Frozen and fixed-step checkpoint evaluation on the stage B validation cohorts."""
from copy import deepcopy
import json
import os
from pathlib import Path
import random
import time

from .choices import extract_letter
from .probe import input_for, sample_answers, summarize
from .tt_data import load_labeled_eval
from ..score_retained_checkpoints import write_json
from ..supervise_semantic_replay import run_workers


def evaluation_cohorts(root, spec):
    for name, n in spec['validation_counts'].items():
        rows = load_labeled_eval(root/f'MedicalVQA_{name}', 'validation')
        if len(rows) != n:
            raise ValueError(f'{name} validation denominator differs from the frozen pilot')
        rows = sorted(rows, key=lambda pair: pair[0].id)
        if name != spec['task']:
            rows = random.Random(spec['seed']).sample(rows, spec['other_validation_n'])
        yield name, rows


def greedy_letter(model, tokenizer, inputs, letters, generation):
    tokens = model.generate(**inputs, generation_config=generation, use_model_defaults=False, synced_gpus=False)
    response = tokens[0, inputs['input_ids'].shape[1]:]
    return extract_letter(tokenizer.decode(response, skip_special_tokens=True), letters)


def evaluate(out, spec):
    import torch
    from ..validate_development import load_checkpoint

    started = time.monotonic()
    checkpoint = os.environ['TT_CHECKPOINT']
    model, processor = load_checkpoint(checkpoint, 0, spec['seed'])
    tokenizer = processor.tokenizer
    greedy = deepcopy(model.generation_config)
    for key, val in spec['sampling'].items():
        setattr(greedy, key, val)
    greedy.do_sample = False
    cohorts = list(evaluation_cohorts(Path(os.environ['TT_DATA']), spec))
    done = 0
    total = sum(len(rows) for _, rows in cohorts)
    result = {}
    with torch.inference_mode():
        for mi, (name, rows) in enumerate(cohorts):
            full_rollouts = name == spec['task']
            samples, hits, invalid = [], 0, 0
            for i, (item, target) in enumerate(rows):
                inputs = input_for(processor, item, spec['prompt'])
                letter = greedy_letter(model, tokenizer, inputs, item.letters, greedy)
                hits += int(letter == target)
                invalid += int(letter is None)
                if full_rollouts:
                    seed = spec['seed'] + 10_000*mi + i
                    samples.append(sample_answers(model, tokenizer, inputs, item.letters, target, spec, seed))
                done += 1
                if (i+1) % 25 == 0 or i+1 == len(rows):
                    elapsed = time.monotonic()-started
                    projection = total*elapsed/done
                    write_json(out/'progress.json', {
                        'modality': name, 'processed': i+1, 'modality_total': len(rows),
                        'total_processed': done, 'total': total, 'elapsed_seconds': elapsed,
                        'projected_wall_seconds': projection})
                    if projection > spec['projected_wall_limit_seconds']:
                        raise RuntimeError('projected pilot evaluation exceeds its frozen time limit')
            stats = summarize(samples, spec['agreement_band_votes']) if full_rollouts else {'n': len(rows)}
            stats.update(greedy=hits/len(rows), greedy_correct=hits, greedy_invalid=invalid)
            result[name] = stats
            write_json(out/'validation_partial.json', result)
    other = [x['greedy'] for key, x in result.items() if key != spec['task']]
    write_json(out/'validation_result.json', {
        'task': spec['task'], 'prompt': spec['prompt'], 'split': 'validation',
        'modalities': result, 'other_modalities_mean_greedy': sum(other)/len(other),
        'elapsed_seconds': time.monotonic()-started, 'peak_allocated_gpu_bytes': torch.cuda.max_memory_allocated()})


def main():
    out = Path(os.environ['OUT'])
    spec = json.loads((out/'protocol.json').read_text())
    if spec['stage'] != 'B' or spec['split'] != 'validation' or ',' in os.environ['CUDA_VISIBLE_DEVICES']:
        raise ValueError('stage B pilot evaluator requires validation and one GPU')
    if os.environ.get('TT_EVAL_WORKER') == '1':
        evaluate(Path(os.environ.get('TT_EVAL_OUT',out)),spec)
        return
    previous_clock = out/'original_eval_clock.json'
    if previous_clock.exists():
        original = json.loads(previous_clock.read_text())
        started = original['started']
        deadline = min(original['deadline'], started + min(spec['max_wall_seconds'], 23*3600))
    else:
        started = time.time()
        deadline = started + min(spec['max_wall_seconds'], 23*3600)
    if time.time() >= deadline:
        raise ValueError('original pilot evaluation deadline has expired')
    with (out/'clock.json').open('x') as stream:
        json.dump({'started':started, 'deadline':deadline},stream)
    result = run_workers(out,spec,0,[{'MODULE': 'clinical_pact.testtime.pilot_eval','TT_EVAL_WORKER': '1'}],deadline)
    write_json(out/'status.json',result)
    if result['reason'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

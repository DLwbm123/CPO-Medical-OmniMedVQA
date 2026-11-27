"""One evaluator for the validation matrix and paired decision probes."""
from collections import Counter
import json
import os
from pathlib import Path
import random

from .answer_decisions import answer_candidate
from .score_retained_checkpoints import transitions, write_json
from .scoring import normalize
from .v3_memory import TASKS, candidate_tokens, decision_views, fp32_scores, memory_pool


def categorical_kl(actor_logits, reference_logits):
    actor = actor_logits.float().log_softmax(-1)
    reference = reference_logits.float().log_softmax(-1)
    return (actor.exp() * (actor - reference)).sum(-1).clamp_min(0)


def evaluate_kl(model, processor, reference, items, rank):
    from .multimodal import make_inputs

    energy = 0.
    tokens = 0
    for item in items:
        prompt = make_inputs(processor, item, device=rank)
        ids = model.generate(**prompt, do_sample=False, num_beams=1, max_new_tokens=128,
                             repetition_penalty=1.0, use_cache=True, synced_gpus=False)
        n = ids.shape[1] - prompt['input_ids'].shape[1]
        if not n:
            raise RuntimeError('empty completion in stage KL diagnostic')
        args = {**prompt, 'input_ids':ids, 'attention_mask':ids.new_ones(ids.shape),
                'logits_to_keep':n+1, 'use_cache':False}
        a = model(**args).logits[:, :-1]
        b = reference(**args).logits[:, :-1]
        per_token = categorical_kl(a, b)
        energy += float(per_token.double().sum())
        tokens += n
    return {'questions':len(items), 'tokens':tokens, 'kl_sum':energy}


def evaluate(out, config, rank):
    import torch

    from .data import load_split
    from .validate_development import generate_answer, load_checkpoint

    root, reference = (Path(os.environ[k]) for k in ('ROOT', 'REFERENCE'))
    torch.set_float32_matmul_precision('highest')
    model, processor = load_checkpoint(root/config['checkpoint'], rank, config['seed'])
    size = config['worker_count']
    result = {'validation': {}, 'decisions': {}}
    with torch.inference_mode():
        for task in config['tasks']:
            items = list(load_split(root/'data/prepared'/TASKS[task], 'validation'))
            if len(items) != config['validation_sizes'][task]:
                raise ValueError('unexpected validation count')
            totals = Counter(n=0, correct=0, generated_tokens=0, eos_terminated=0)
            for i, item in enumerate(items[rank::size], 1):
                text, stats = generate_answer(model, processor, item, rank, 128)
                totals.update(stats)
                totals['n'] += 1
                if task == 'pathvqa':
                    label = normalize(item['answer'])
                    if label in ('yes','no'):
                        answer = answer_candidate(text)
                        answer = answer if answer in ('yes','no') else 'other'
                        totals[f'gold_{label}_pred_{answer}'] += 1
                    else:
                        totals.update(other_n=1, other_correct=stats['correct'])
                if i % 250 == 0:
                    write_json(out/f'w{rank}_progress.json', {'task':task,'n':i})
            result['validation'][task] = dict(totals)
            if task == config.get('kl_task'):
                boundary, _ = load_checkpoint(root/config['kl_reference'], rank, config['seed'])
                cohort = random.Random(100).sample(items, config['kl_samples'])
                result['stage_kl'] = evaluate_kl(model, processor, boundary, cohort[rank::size], rank)
                del boundary

        for task in config['probes']:
            pool, outside = memory_pool(root, reference, task)
            spec = candidate_tokens(processor.tokenizer, task)
            result['decisions'][task] = {}
            for label, rows in (('candidate', pool),('unreplayed', outside)):
                views = decision_views(rows, task, 4)
                bits = []
                for i in range(rank,len(views),size):
                    item,target = views[i]
                    logits = fp32_scores(model, processor, item, rank, spec)
                    bits.append([i,int(logits.argmax())==target])
                    if len(bits) % 250 == 0:
                        write_json(out/f'w{rank}_progress.json', {'task':task,'probe':label,'n':len(bits)})
                result['decisions'][task][label] = bits
    write_json(out/'private'/f'w{rank}.json',result)


def aggregate(out, config):
    stats, decisions = {}, {}
    kl = Counter()
    for rank in range(config['worker_count']):
        data = json.loads((out/'private'/f'w{rank}.json').read_text())
        kl.update(data.get('stage_kl', {}))
        for k, v in data['validation'].items():
            stats.setdefault(k, Counter()).update(v)
        for k, groups in data['decisions'].items():
            for name, bits in groups.items():
                decisions.setdefault(k, {}).setdefault(name, []).extend(bits)
    for k, v in stats.items():
        if v['n'] != config['validation_sizes'][k]:
            raise ValueError('incomplete sharded validation')
        v['accuracy'] = v['correct'] / v['n']
        if k == 'pathvqa':
            total = {gt: sum(v[f'gold_{gt}_pred_{p}'] for p in ('yes', 'no','other')) for gt in ('yes','no')}
            if list(total.values()) != [1712,1413] or v['other_n'] != 3134:
                raise ValueError('PathVQA answer-type denominator changed')
            v['yes_no_balanced_accuracy'] = sum(v[f'gold_{gt}_pred_{gt}']/n for gt,n in total.items()) / 2
            v['predicted_yes_fraction'] = sum(v[f'gold_{gt}_pred_yes'] for gt in total)/sum(total.values())
    for k, groups in decisions.items():
        for group, pairs in groups.items():
            ordered = sorted(pairs)
            expected = config['probe_sizes'][k][group] * (4 if k=='medbook' else 1)
            if [p[0] for p in ordered] != list(range(expected)):
                raise ValueError('missing or duplicate decision-probe rows')
            groups[group] = [int(p[1]) for p in ordered]
    write_json(out/'private'/'decisions.json', decisions)
    probes = {k:{g:{'views':len(bits),'correct':sum(bits)} for g,bits in gg.items()} for k,gg in decisions.items()}
    report = {'tasks':stats,'probes':probes}
    if config.get('kl_task'):
        if kl['questions'] != config['kl_samples'] or not kl['tokens']:
            raise ValueError('incomplete stage KL diagnostic')
        report['stage_kl'] = {'task':config['kl_task'], 'questions':kl['questions'], 'tokens':kl['tokens'],
                              'mean_token_kl':kl['kl_sum'] / kl['tokens']}
    write_json(out/'aggregate.json', report)


def stages_summary(out, config):
    start = [out/f'e{i}' for i in (1,2,3)]
    stages = [json.loads((p/'aggregate.json').read_text()) for p in start]
    decisions = [json.loads((p/'private'/'decisions.json').read_text()) for p in start]
    reference_file = out/'private'/'s3.json'
    memory = json.loads(reference_file.read_text())['records'] if reference_file.is_file() else None
    changes = {}
    for task, initial in [('medbook',0), ('pathvqa',1)]:
        before,after = decisions[initial][task],decisions[2][task]
        if memory is None:
            width = 4 if task == 'medbook' else 1
            bits = before['candidate']
            indexes = [i for start in range(0,len(bits),width) if all(bits[start:start+width])
                       for i in range(start,start+width)]
        else:
            indexes = [r[0] for r in memory[task]]
            if task == 'medbook' and not config['rotations']:
                indexes = [4*i for i in indexes]
        retain = [after['candidate'][i] for i in indexes]
        changes[task] = {'memory':{'n':len(retain),'flips':len(retain)-sum(retain)},
                         'unreplayed_view_transitions': transitions(before['unreplayed'],after['unreplayed'])}
        if task == 'medbook':
            to_bits = lambda x:[int(all(x[i:i+4])) for i in range(0,len(x),4)]
            changes[task]['unreplayed_all_orders_transitions'] = transitions(to_bits(before['unreplayed']),to_bits(after['unreplayed']))
    matrix = {f's{i}':{k:v['accuracy'] for k,v in stage['tasks'].items()} for i,stage in enumerate(stages,1)}
    last = matrix['s3']
    final_minus_peak = {k:last[k]-max(row[k] for row in matrix.values() if k in row) for k in ('medbook','pathvqa')}
    learning_stage = {'medbook':'s1','pathvqa':'s2','slake':'s3'}
    backward_transfer = sum(last[k]-matrix[learning_stage[k]][k] for k in ('medbook','pathvqa'))/2
    report = {'validation_matrix':matrix,'stages':stages,'old_task_changes':changes,
        'final_average_accuracy':sum(last.values())/len(last),'final_minus_peak':final_minus_peak,
        'backward_transfer':backward_transfer,'plasticity':{k:matrix[v][k] for k,v in learning_stage.items()}}
    if config.get('r1_alpha') or config.get('fisher_tr'):
        report['retention_extensions'] = {}
        for i in (2, 3):
            train = json.loads((out/f's{i}'/'training.json').read_text())
            stage = {}
            if config.get('fisher_tr'):
                stage['fisher_omega_final'] = train['fisher_omega_final']
            if config.get('r1_alpha'):
                stage['r1_sampled_log_ratio'] = [{'step':row['step'], 'mean':row['r1/sampled_log_ratio'],
                    'tie_split_fraction':row['r1/all_correct_tie_split_fraction']}
                    for row in train['logs'] if 'r1/sampled_log_ratio' in row]
            report['retention_extensions'][f's{i}'] = stage
    write_json(out/'aggregate.json', report)


if __name__ == '__main__':
    dst = Path(os.environ['EVAL_DIR'])
    config = json.loads((dst/'protocol.json').read_text())
    if 'RANK' in os.environ:
        evaluate(dst,config,int(os.environ['RANK']))
    else:
        aggregate(dst,config)

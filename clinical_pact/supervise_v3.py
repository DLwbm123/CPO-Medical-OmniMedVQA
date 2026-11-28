"""Resume the frozen first stage and run the two subsequent main-method stages."""
import json
import os
from pathlib import Path
import resource
import socket
import subprocess
import time

from .score_retained_checkpoints import write_json
from .supervise_semantic_replay import run_workers
from .v3_eval import medbook_cache_signature, restore_medbook_cache, save_medbook_cache, stages_summary


def retained_metrics(aggregate):
    tasks = aggregate['tasks']
    if 'pathvqa' in tasks and 'yes_no_accuracy' not in tasks['pathvqa']:
        tb = tasks['pathvqa']
        totals = [sum(tb.get(f'gold_{gt}_pred_{pred}',0) for pred in ('yes','no','other')) for gt in ('yes','no')]
        if totals != [1712,1413]:
            raise ValueError('retained PathVQA yes/no counts differ from the frozen cohort')
        tb['yes_no_accuracy'] = (tb.get('gold_yes_pred_yes',0) + tb.get('gold_no_pred_no',0)) / sum(totals)
    return tasks


def main():
    out, root, reference = (Path(os.environ[k]) for k in ('OUT','ROOT','REFERENCE'))
    p = json.loads((out/'protocol.json').read_text())
    smoke = p.get('memory_smoke', False)
    retained_run = out.parent/p['frozen_report_of'] if p.get('frozen_report_of') else None
    any_extension = any(p.get(k) for k in ('r1_beta','fisher_tr','dynamic_sampling','batched_same_prompt_generation'))
    if p.get('r1_alpha'):
        raise ValueError('R1 now uses r1_beta for the native KL loss')
    if p.get('mode') == 'no_memory' and any(p.get(k) for k in ('fisher_tr','dynamic_sampling')):
        raise ValueError('these extensions require the v3 training arm')
    if smoke and (p['updates'] != 1 or not p.get('r1_beta') or not p.get('fisher_tr') or not p.get('fisher_gamma')):
        raise ValueError('memory smoke requires one update with R1 and an explicit test-only Fisher gamma')
    if p.get('single_gpu') and p.get('r1_beta') and p.get('fisher_tr') and not smoke:
        receipt = out.parent/p['memory_smoke_receipt']
        test = json.loads((receipt/'protocol.json').read_text())
        status = json.loads((receipt/'status.json').read_text())
        timings = json.loads((receipt/'s2/timing.json').read_text())
        if (status['reason'] != 'completed' or not test.get('memory_smoke') or not test.get('single_gpu')
                or not test.get('fisher_tr') or test.get('r1_beta') != p['r1_beta']
                or len(timings) != 1 or not timings[0]['peak_allocated_bytes']):
            raise ValueError('single-GPU R1+C3 needs a completed one-update memory receipt')
    if retained_run and json.loads((retained_run/'status.json').read_text())['reason'] != 'completed':
        raise ValueError('frozen report must use a completed run')
    if p.get('fisher_tr') and p['seed'] != 100 and not (root/p['fisher_gamma_reference']).is_file():
        raise ValueError('publish the seed-100 Fisher calibration before launching another seed')
    started = time.time()
    deadline = started + min(p['max_wall_seconds'], 86_400)
    if 'original_clock' in p:
        original = json.loads((out.parent/p['original_clock']).read_text())
        deadline = min(deadline, original['deadline'])
        started = original['started']
    ranks = 1 if p.get('single_gpu') else 2
    with (out/'clock.json').open('x') as f:
        json.dump({'started':started, 'deadline':deadline}, f)
    resource.setrlimit(resource.RLIMIT_FSIZE, (10_000_000_000, 10_000_000_000))
    history = []

    def phase(env):
        index = len(history)
        write_json(out/'progress.json',{'phase':index,'state':'running'})
        result = run_workers(out,p,index,env,deadline)
        history.append(result)
        write_json(out/'history.json',history)
        return result['reason']=='completed'

    def launch(module, **options):
        gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
        available = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                                            '--format=csv,noheader,nounits'],text=True)
        free = {k.strip():int(v) for k,v in (line.split(',') for line in available.splitlines())}
        if len(gpus)!=ranks or any(free[g]<60000 for g in gpus):
            history.append({'phase':len(history),'reason':'insufficient_free_memory'})
            return False
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            port = str(sock.getsockname()[1])
        base = dict(MODULE=module, WORLD_SIZE=str(ranks), MASTER_ADDR='127.0.0.1', MASTER_PORT=port, **options)
        return phase([dict(base,RANK=str(rank),LOCAL_RANK=str(rank)) for rank in range(ranks)])

    def train(stage):
        if p.get('fisher_tr'):
            if not launch('clinical_pact.fisher_boundary', STAGE=str(stage)):
                return False
            if stage == 2 and not smoke:
                settings = {'MODULE':'clinical_pact.fisher_boundary','FISHER_PHASE':'calibrate',
                            'CUDA_VISIBLE_DEVICES':''}
                if not phase([settings]):
                    return False
        return launch('clinical_pact.v3_train', STAGE=str(stage))

    def evaluate(stage):
        dst = out/f'e{stage}'
        (dst/'private').mkdir(parents=True)
        source = reference/'stage1/model' if stage==1 else (retained_run or out)/f's{stage}'/'model'
        include = ['medbook','pathvqa','slake'][:stage]
        evaluation = {'checkpoint':str(source.relative_to(root)), 'seed':100 if stage == 1 else p['seed'],
                      'tasks':include,'probes':include[:2], 'worker_count':ranks,
                      'validation_sizes':p['validation_sizes'],'probe_sizes':p['probe_sizes']}
        if stage > 1 and (any_extension or p.get('stage_kl_diagnostic')) and p.get('stage_kl_samples'):
            anchor = reference/'stage1/model' if stage == 2 else out/'s2/model'
            evaluation.update(kl_task=include[-1], kl_reference=str(anchor.relative_to(root)),
                              kl_samples=p['stage_kl_samples'])
        if retained_run:
            old = retained_run/f'e{stage}'
            evaluation.update(probes=[], retained_probes=str(old/'private/decisions.json'),
                              precomputed_tasks=retained_metrics(json.loads((old/'aggregate.json').read_text())))
        write_json(dst/'protocol.json', evaluation)
        cache = root/p['shared_medbook_eval_cache'] if stage == 1 and p.get('shared_medbook_eval_cache') else None
        signature = medbook_cache_signature(root, reference, evaluation) if cache else None
        if cache and restore_medbook_cache(cache, dst, signature):
            return True
        base = dict(MODULE='clinical_pact.v3_eval', EVAL_DIR=str(dst))
        complete = phase([dict(base,RANK=str(rank)) for rank in range(ranks)]) and phase([base])
        if complete and cache:
            save_medbook_cache(cache, dst, signature)
        return complete

    if time.time() >= deadline:
        write_json(out/'status.json', {'reason':'time_limit'})
        return
    if retained_run:
        complete = all(evaluate(stage) for stage in (1,2,3))
    else:
        valid = (ranks != 1 and not p.get('fisher_tr')) or launch('clinical_pact.projection_check')
        complete = valid and train(2)
        if complete and not smoke:
            complete = evaluate(2) and train(3) and evaluate(3) and evaluate(1)
    if complete and not smoke:
        stages_summary(out,p,retained_run=retained_run)
    write_json(out/'status.json',{'reason':'completed' if complete else history[-1]['reason'],'phases':history})


if __name__=='__main__':
    main()

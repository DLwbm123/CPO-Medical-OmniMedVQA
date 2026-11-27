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


def main():
    out, root, reference = (Path(os.environ[k]) for k in ('OUT','ROOT','REFERENCE'))
    p = json.loads((out/'protocol.json').read_text())
    any_extension = any(p.get(k) for k in ('r1_beta','fisher_tr','dynamic_sampling','batched_same_prompt_generation'))
    if p.get('r1_alpha'):
        raise ValueError('R1 now uses r1_beta for the native KL loss')
    if p.get('mode') == 'no_memory' and any(p.get(k) for k in ('fisher_tr','dynamic_sampling')):
        raise ValueError('these extensions require the v3 training arm')
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
            if stage == 2:
                settings = {'MODULE':'clinical_pact.fisher_boundary','FISHER_PHASE':'calibrate',
                            'CUDA_VISIBLE_DEVICES':''}
                if not phase([settings]):
                    return False
        return launch('clinical_pact.v3_train', STAGE=str(stage))

    def evaluate(stage):
        dst = out/f'e{stage}'
        (dst/'private').mkdir(parents=True)
        source = reference/'stage1/model' if stage==1 else out/f's{stage}'/'model'
        include = ['medbook','pathvqa','slake'][:stage]
        evaluation = {'checkpoint':str(source.relative_to(root)), 'seed':100 if stage == 1 else p['seed'],
                      'tasks':include,'probes':include[:2], 'worker_count':ranks,
                      'validation_sizes':p['validation_sizes'],'probe_sizes':p['probe_sizes']}
        if stage > 1 and (any_extension or p.get('stage_kl_diagnostic')) and p.get('stage_kl_samples'):
            anchor = reference/'stage1/model' if stage == 2 else out/'s2/model'
            evaluation.update(kl_task=include[-1], kl_reference=str(anchor.relative_to(root)),
                              kl_samples=p['stage_kl_samples'])
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
    valid = (ranks != 1 and not p.get('fisher_tr')) or launch('clinical_pact.projection_check')
    complete = valid and train(2) and evaluate(2) and train(3) and evaluate(3) and evaluate(1)
    if complete:
        stages_summary(out,p)
    write_json(out/'status.json',{'reason':'completed' if complete else history[-1]['reason'],'phases':history})


if __name__=='__main__':
    main()

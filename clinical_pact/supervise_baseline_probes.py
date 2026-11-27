"""Finish the baseline's four-order and binary probes after the SLAKE job exits."""
import json
import os
from pathlib import Path
import subprocess
import time

from .score_retained_checkpoints import write_json
from .supervise_semantic_replay import run_workers
from .v3_eval import medbook_cache_signature, restore_medbook_cache, save_medbook_cache, stages_summary


def read(path):
    return json.loads(path.read_text())


def generation_counts(subsets):
    tasks = {k:{f:subsets[k][f] for f in ('n','correct','generated_tokens','eos_terminated')}
             for k in ('medbook','slake') if k in subsets}
    if 'pathvqa_yes_no' in subsets:
        yes_no, other = (subsets[k] for k in ('pathvqa_yes_no','pathvqa_other'))
        if (yes_no['n'], other['n']) != (3125,3134):
            raise ValueError('cached PathVQA answer type denominators changed')
        tasks['pathvqa'] = {f:yes_no[f]+other[f] for f in ('n','correct','generated_tokens','eos_terminated')}
        tasks['pathvqa']['yes_no_accuracy'] = yes_no['correct'] / yes_no['n']
    return tasks


def main():
    out, root = (Path(os.environ[k]) for k in ('OUT','ROOT'))
    spec = read(out/'protocol.json')
    baseline = root/spec['baseline_run']
    reference = Path(os.environ['REFERENCE'])
    before_path = root/spec['pathvqa_run']/'s1'
    status_file = baseline/'status.json'
    clock = read(baseline/'clock.json')
    # The already-started SLAKE job keeps its original deadline. The end-to-end
    # comparison starting at the same clock is below the 24-hour hard cap.
    deadline = clock['started'] + spec['max_wall_seconds']
    while not status_file.exists() and time.time() < clock['deadline']:
        time.sleep(30)
    if not status_file.exists() or read(status_file)['reason'] != 'completed':
        write_json(out/'status.json',{'reason':'upstream_run_incomplete'});return
    gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    query = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                                     '--format=csv,noheader,nounits'],text=True)
    free = {g.strip():int(v) for g,v in (line.split(',') for line in query.splitlines())}
    if any(free[g]<16000 for g in gpus):
        write_json(out/'status.json',{'reason':'insufficient_free_memory'});return
    phases = []
    pairs = [(reference/'stage1/model',{}),
             (before_path/'model',generation_counts(read(before_path/'validation/aggregate.json')['subsets'])),
             (baseline/'s1/model',generation_counts(read(baseline/'s1/validation/aggregate.json')['subsets']))]
    for stage,(model,precomputed) in enumerate(pairs,1):
        dst=out/f'e{stage}'
        (dst/'private').mkdir(parents=True)
        tasks = ['medbook','pathvqa','slake'][:stage]
        probes = ['medbook'] if stage == 1 else ['pathvqa'] if stage == 2 else ['medbook','pathvqa']
        config = {'checkpoint':str(model.relative_to(root)),'seed':100,
                  'tasks':tasks,'probes':probes,'worker_count':len(gpus),'precomputed_tasks':precomputed,
                  'validation_sizes':spec['validation_sizes'],'probe_sizes':spec['probe_sizes']}
        write_json(dst/'protocol.json',config)
        cache = root/spec['shared_medbook_eval_cache'] if stage == 1 and spec.get('shared_medbook_eval_cache') else None
        signature = medbook_cache_signature(root, reference, config) if cache else None
        if cache and restore_medbook_cache(cache, dst, signature):
            continue
        env = dict(MODULE='clinical_pact.v3_eval', EVAL_DIR=str(dst))
        for workers in ([dict(env,RANK=str(rank)) for rank in range(len(gpus))],[env]):
            part = run_workers(out,spec,len(phases),workers,deadline)
            phases.append(part)
            write_json(out/'history.json', phases)
            if part['reason']!='completed':
                write_json(out/'status.json', {'reason':part['reason'],'phases':phases});return
        if cache:
            save_medbook_cache(cache, dst, signature)
    stages_summary(out,spec)
    write_json(out/'status.json',{'reason':'completed','phases':phases})


if __name__ == '__main__':
    main()

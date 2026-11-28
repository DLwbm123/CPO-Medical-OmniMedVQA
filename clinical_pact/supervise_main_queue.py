"""Dispatch the next matched seeds only after the frozen first-round gate."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .score_retained_checkpoints import write_json


def read(path):
    return json.loads(path.read_text())


def first_gate(study, main_run, baseline_run):
    a = study/main_run
    b = study/baseline_run
    control = read(b/'aggregate.json')
    main = read(a/'aggregate.json')
    if 'semantic_validation_matrix' not in main or 'semantic_validation_matrix' not in control:
        raise ValueError('require the corrected semantic validation before deciding at the gate')
    rates = [[],[]]
    for task, stage in [('medbook',1),('pathvqa',2)]:
        values = [[read(run/f'e{i}'/'private'/'decisions.json')[task]['unreplayed'] for i in (stage,3)]
                  for run in (b,a)]
        if task == 'medbook':
            values = [[[int(all(data[j:j+4])) for j in range(0,len(data),4)] for data in group] for group in values]
        old_b,new_b=values[0];old_a,new_a=values[1]
        common = [j for j,(left,right) in enumerate(zip(old_b,old_a,strict=True)) if left and right]
        if not common:
            raise ValueError('no common initially correct examples for a semantic probe')
        for k,end in enumerate((new_b,new_a)):
            rates[k].append(sum(not end[j] for j in common)/len(common))
    b_rate,a_rate = (sum(task_rates)/2 for task_rates in rates)
    strong = b_rate > 0 and a_rate <= .6*b_rate
    a_path = main['validation_matrix']['s2']['pathvqa']
    b_path = control['validation_matrix']['s2']['pathvqa']
    plasticity = a_path >= b_path
    route = 'repeat_v3' if strong and plasticity else 'try_r1_c3'
    return {'baseline_forgetting_gate':control['measurable_forgetting_gate'],
            'baseline_common_flip_rate':b_rate,'v3_common_flip_rate':a_rate,
            'v3_retention_gate':strong,'pathvqa_plasticity_gate':plasticity,
            'baseline_pathvqa_acquisition':b_path,'v3_pathvqa_acquisition':a_path,'route':route}


def main():
    out,root = (Path(os.environ[k]) for k in ('OUT','ROOT'))
    cfg = read(out/'protocol.json'); study=out.parent
    main_run = cfg['main_run']
    original_status = study/main_run/'status.json'
    deadline = read(study/main_run/'clock.json')['deadline']
    while time.time() < deadline:
        if original_status.exists():
            break
        time.sleep(30)
    if not original_status.exists() or read(original_status)['reason'] != 'completed':
        write_json(out/'status.json',{'reason':'dependency_incomplete'});return
    if cfg['lanes'] != [[5]]:
        raise ValueError('this queue is restricted to GPU 5')
    defaults = read(Path(__file__).with_name('v3_main_protocol.json'))
    template = {**defaults, **read(study/main_run/'protocol.json')}
    template.pop('original_clock',None)
    template['first_round_gate'] = defaults['first_round_gate']
    if not template.get('single_gpu'):
        raise ValueError('seed-100 comparison requires the existing single-GPU v3 run')
    finished = []

    def run(job):
        dst = study/job['run']
        status = dst/'status.json'
        run_spec = {**template, **{k:v for k,v in job.items() if k != 'run'}}
        if status.exists():
            previous = read(dst/'protocol.json')
            if any(previous.get(k) != v for k,v in run_spec.items() if k not in ('fisher_gamma','fisher_actual_samples')):
                raise ValueError('existing queued run uses a different protocol')
            report = read(status)
            finished.append({'run':job['run'],'report':report,'reused':True})
            if report['reason'] == 'completed':
                return True
            why = 'single_gpu_memory_smoke_failed' if job.get('memory_smoke') else 'run_failure'
            write_json(out/'status.json',{'reason':why,'runs':finished})
            return False
        idle_limit = time.time()+14400
        while time.time() < idle_limit:
            free = subprocess.check_output(['nvidia-smi','--id=5','--query-gpu=memory.free',
                                           '--format=csv,noheader,nounits'],text=True)
            if int(free.strip()) >= 60000:
                break
            time.sleep(30)
        else:
            write_json(out/'status.json',{'reason':'waiting_for_resources','pending':job,'runs':finished})
            return False
        dst.mkdir()
        for sub in ('cache/tmp','cache/triton','private'):
            (dst/sub).mkdir(parents=True)
        shutil.copytree(out/'source',dst/'source')
        write_json(dst/'protocol.json',run_spec)
        env={**os.environ,'OUT':str(dst),'CUDA_VISIBLE_DEVICES':'5',
             'PYTHONPATH':str(dst/'source')+':'+str(root/'env/lib/python3.10/site-packages'),
             'HF_HOME':str(dst/'cache'),'TMPDIR':str(dst/'cache/tmp'),'TRITON_CACHE_DIR':'cache/triton',
             'MODULE':'clinical_pact.supervise_v3'}
        args=['python','-c','import os,runpy;runpy.run_module(os.environ["MODULE"],run_name="__main__")']
        with (dst/'launcher.log').open('x') as stream:
            process=subprocess.Popen(args,cwd=dst,env=env,start_new_session=True,stdout=stream,stderr=subprocess.STDOUT)
            (dst/'launcher.pid').write_text(str(process.pid)+'\n')
            write_json(out/'queue.json',{'running':job,'finished':finished})
            code = process.wait()
        report = read(status) if status.exists() else None
        finished.append({'run':job['run'],'exit_code':code,'report':report})
        ok = code == 0 and report and report['reason'] == 'completed'
        if not ok:
            why = 'single_gpu_memory_smoke_failed' if job.get('memory_smoke') else 'run_failure'
            write_json(out/'status.json',{'reason':why,'runs':finished})
        return ok

    rescore = cfg['main_rescore']
    baseline = cfg['matched_baseline']
    if not all(run(job) for job in (rescore,baseline)):
        return
    gate = first_gate(study,rescore['run'],baseline['run'])
    write_json(out/'gate.json',gate)
    jobs = cfg['queue'] if gate['route'] == 'repeat_v3' else cfg['retention_fallback']
    if all(run(job) for job in jobs):
        write_json(out/'status.json',{'reason':'completed','runs':finished})


if __name__=='__main__':
    main()

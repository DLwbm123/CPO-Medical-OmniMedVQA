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
    return {'baseline_forgetting_gate':control['measurable_forgetting_gate'],
            'baseline_common_flip_rate':b_rate,'v3_common_flip_rate':a_rate,
            'v3_retention_gate':strong,'continue':bool(control['measurable_forgetting_gate'] or strong)}


def main():
    out,root = (Path(os.environ[k]) for k in ('OUT','ROOT'))
    cfg = read(out/'protocol.json'); study=out.parent
    main_run,baseline_run = cfg['main_run'],cfg['baseline_report_run']
    dependencies = [study/d/'status.json' for d in (main_run,baseline_run)]
    deadline=max(read(study/main_run/'clock.json')['deadline'],read(study/cfg['baseline_training_run']/'clock.json')['started']+57600)
    while time.time() < deadline:
        statuses = [read(p)['reason'] if p.exists() else None for p in dependencies]
        if any(s not in ('completed',None) for s in statuses) or all(statuses):
            break
        time.sleep(30)
    if not all(p.exists() and read(p)['reason']=='completed' for p in dependencies):
        write_json(out/'status.json',{'reason':'dependency_incomplete'});return
    gate=first_gate(study,main_run,baseline_run)
    write_json(out/'gate.json',gate)
    if not gate['continue']:
        write_json(out/'status.json',{'reason':'needs_stronger_sequence'});return
    pending=list(cfg['queue']);running={};receipt=[]
    template=read(study/main_run/'protocol.json')
    template.pop('original_clock',None)
    if template.get('single_gpu') and cfg['lanes'] != [[5]]:
        raise ValueError('the current study is restricted to GPU 5')
    idle_limit=time.time()+14400
    while pending or running:
        for lane, (job, process) in list(running.items()):
            if process.poll() is None:
                continue
            dst=study/job['run']; latest=dst/'status.json'
            receipt.append({'run':job['run'],'seed':job['seed'],'mode':job['mode'],
                            'exit_code':process.returncode,'report':read(latest) if latest.exists() else None})
            del running[lane]
            idle_limit=time.time()+14400
        if pending:
            query=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                                           '--format=csv,noheader,nounits'],text=True)
            free={g.strip():int(v) for g,v in (x.split(',') for x in query.splitlines())}
            for lane,gpus in enumerate(cfg['lanes']):
                if not pending or lane in running or any(free[str(g)]<60000 for g in gpus):
                    continue
                job=pending.pop(0);dst=study/job['run']
                dst.mkdir()
                for sub in ('cache/tmp','cache/triton','private'):
                    (dst/sub).mkdir(parents=True)
                shutil.copytree(out/'source',dst/'source')
                run_spec={**template,'seed':job['seed'],'mode':job['mode']}
                write_json(dst/'protocol.json',run_spec)
                env={**os.environ,'OUT':str(dst),'CUDA_VISIBLE_DEVICES':','.join(map(str,gpus)),
                     'PYTHONPATH':str(dst/'source')+':'+str(root/'env/lib/python3.10/site-packages'),
                     'HF_HOME':str(dst/'cache'),'TMPDIR':str(dst/'cache/tmp'),'TRITON_CACHE_DIR':'cache/triton',
                     'MODULE':'clinical_pact.supervise_v3'}
                args=['python','-c','import os,runpy;runpy.run_module(os.environ["MODULE"],run_name="__main__")']
                with (dst/'launcher.log').open('x') as stream:
                    process=subprocess.Popen(args,cwd=dst,env=env,start_new_session=True,stdout=stream,stderr=subprocess.STDOUT)
                (dst/'launcher.pid').write_text(str(process.pid)+'\n')
                running[lane]=(job,process)
        write_json(out/'queue.json',{'running':{str(k):v[0] for k,v in running.items()},'pending':pending,'finished':receipt})
        if pending and not running and time.time()>=idle_limit:
            write_json(out/'status.json',{'reason':'waiting_for_resources','pending':pending});return
        if pending or running:
            time.sleep(30)
    success=all(x['exit_code']==0 and x['report'] and x['report']['reason']=='completed' for x in receipt)
    write_json(out/'status.json',{'reason':'completed' if success else 'run_failure','runs':receipt})


if __name__=='__main__':
    main()

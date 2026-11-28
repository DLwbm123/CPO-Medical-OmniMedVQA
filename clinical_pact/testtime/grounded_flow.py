"""Run P0 and its two gated pilots only after the old replication releases the cards."""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from .tt_data import load_unlabeled
from ..score_retained_checkpoints import write_json
from ..supervise_semantic_replay import run_workers


def wait_for_replication(out, replication):
    clock = json.loads((replication/'clock.json').read_text())
    write_json(out/'progress.json',{'phase':'waiting_for_replication','gpu_workers':0})
    while time.time() < clock['deadline']:
        final = replication/'amended_status.json'
        if final.exists():
            status = json.loads(final.read_text())
            if status['reason']!='completed_with_cancelled_arm':
                raise RuntimeError('the preserved replication arm did not reach a verified endpoint')
            return
        time.sleep(10)
    raise RuntimeError('replication exceeded its original deadline')


def free_cards(gpus, minimum, deadline):
    while time.time()<deadline:
        command = ['nvidia-smi','-i',','.join(gpus)]
        result = subprocess.check_output([*command,'--query-gpu=memory.free',
                                          '--format=csv,noheader,nounits'],text=True)
        busy = subprocess.check_output([*command,'--query-compute-apps=pid',
                                        '--format=csv,noheader,nounits'],text=True)
        free = [int(x) for x in result.splitlines()]
        if not busy.strip() and len(free)==len(gpus) and min(free)>=minimum:
            return
        time.sleep(15)
    raise RuntimeError('the authorized GPUs did not have enough available memory')


def audit_summary(destination):
    records = [json.loads(line) for line in (destination/'audit.jsonl').read_text().splitlines()]
    if not records:
        raise ValueError('grounded training must produce a nonempty audit')
    questions = [q for row in records for q in row['questions']]
    weights = sorted(q['weight'] for q in questions)
    return {'updates':len(records),'questions':len(questions),
            'm1_skips':sum(q['m1_skip'] for q in questions),
            'm2_skips':sum(q['m2_skip'] for q in questions),
            'source_fallbacks':sum(q['source_fallback'] for q in questions),
            'weight_min_median_max':[weights[0],weights[len(weights)//2],weights[-1]],
            'optimizer_steps':sum(row['applied_optimizer_expected'] for row in records),
            'dual_audits':[{'step':r['step'],'lambda_b':r['lambda_b'],'next_lambda_b':r['next_lambda_b'],
                            'blind_kl_nats':r['audited_blind_kl']}
                           for r in records if 'audited_blind_kl' in r]}


def run(out):
    replication = Path(os.environ['TT_REPLICATION_OUT']).resolve(strict=True)
    source = Path(__file__).parent
    gpus = os.environ['TT_GPU_ASSIGNMENT'].split(',')
    if gpus != ['3','5']:
        raise ValueError('grounded flow is limited to the two previously assigned cards')
    configs = {name:json.loads((out/name).read_text()) for name in
        ('precheck_protocol.json','dev_pilot_protocol.json','confirm_pilot_protocol.json','pilot_protocol.json')}
    pre,dev,conf = (configs[k] for k in ('precheck_protocol.json','dev_pilot_protocol.json','confirm_pilot_protocol.json'))
    wait_for_replication(out,replication)
    start = time.time()
    deadline = start+min(pre['max_wall_seconds'],dev['max_wall_seconds'],conf['max_wall_seconds'],23*3600)
    clock = {'started':start,'deadline':deadline}
    with (out/'clock.json').open('x') as stream:
        json.dump(clock,stream)
    phases = []
    public = {}
    limits = {'output_guard_bytes':min(dev['output_guard_bytes'],conf['output_guard_bytes'],48_000_000_000)}
    def phase(name,workers,cards,minimum=24000,storage_limit=None):
        write_json(out/'progress.json',{'phase':name,'state':'waiting_for_memory'})
        free_cards(cards,minimum,deadline)
        write_json(out/'progress.json',{'phase':name,'state':'running'})
        phase_limits = limits if storage_limit is None else {'output_guard_bytes':storage_limit}
        result = run_workers(out,phase_limits,len(phases),workers,deadline)
        phases.append({'name':name,**result})
        write_json(out/'status.json',{'reason':'running' if result['reason']=='completed' else result['reason'],
                                     'phases':phases})
        if result['reason']!='completed':
            raise SystemExit(1)
    def finish(reason,decision):
        write_json(out/'grounded_result.json',{'outcome':reason,'decision':decision,**public})
        write_json(out/'status.json',{'reason':'completed','outcome':reason,'decision':decision,'phases':phases})
        write_json(out/'progress.json',{'phase':'completed','outcome':reason,'decision':decision})

    model = os.environ['TT_CHECKPOINT']
    root = os.environ['TT_DATA']
    def phase_dir(name,cfg):
        folder = out/name
        folder.mkdir(mode=0o700)
        write_json(folder/'protocol.json',cfg)
        write_json(folder/'clock.json',clock)
        for key in ('pilot_protocol.json','dev_pilot_protocol.json'):
            write_json(folder/key,configs[key])
        return folder

    p0 = phase_dir('p0',pre)
    phase('precheck',[{'MODULE':'clinical_pact.testtime.precheck_grounding','TT_P0_WORKER':'1',
                       'OUT':str(p0),'CUDA_VISIBLE_DEVICES':'5'}],['5'],
                       storage_limit=pre['output_guard_bytes'])
    p0_result = p0/'precheck_result.json'
    choice = json.loads(p0_result.read_text())
    public['precheck'] = choice
    if not choice['passed']:
        finish('p0_failed','stop_and_report')
        return

    protected = p0/'private'/'forbidden_images.json'
    def environment(folder,arm,gpu,stream):
        data = load_unlabeled(stream,'train',dev['seed'])
        manifest = folder/'private_images.json'
        write_json(manifest,[{'id':x.id,'image':str(x.image),'source_dataset':x.source_dataset} for x in data])
        with socket.socket() as listener:
            listener.bind(('127.0.0.1',0))
            port = str(listener.getsockname()[1])
        return {'OUT':str(folder),'TT_ARM':arm,'CUDA_VISIBLE_DEVICES':gpu,'WORLD_SIZE':'1',
                'RANK':'0','LOCAL_RANK':'0','MASTER_ADDR':'127.0.0.1','MASTER_PORT':port,
                'TT_REDACTED_TASK':str(stream),
                'TT_COUNTERFACTUAL_MANIFEST':str(manifest),'TT_FORBIDDEN_IMAGES':str(protected),
                'TT_P0_RESULT':str(p0_result),'TT_CHECKPOINT':model}

    def train(env,steps=0):
        return {**env,'MODULE':'clinical_pact.testtime.gc_train','TT_DATA':'',
                'TT_ACCEPTANCE_STEPS':str(steps)}

    def evaluate(env,arm,checkpoint,adapter=None):
        return {**env,'MODULE':'clinical_pact.testtime.confirm_eval','TT_DATA':root,'TT_ARM':arm,
                'TT_CHECKPOINT':checkpoint,'TT_ADAPTER':str(adapter) if adapter is not None else ''}

    fundus = Path(os.environ['TT_FUNDUS_FLOW']).resolve(strict=True)
    verification = phase_dir('s0',dev)
    smoke = environment(verification,'tt_gc','5',fundus)
    phase('verify_grounded_real_model',[train(smoke,3)],['5'],minimum=40000)
    original = json.loads((verification/'initial_reference.json').read_text())
    check = json.loads((verification/'training_result.json').read_text())
    diagnostic = audit_summary(verification)
    timing = json.loads((verification/'timing.json').read_text())
    peak = max(row['peak_reserved_bytes'] for row in timing)
    verification_summary = {'reference':original,'training':check,'grounding':diagnostic,
                            'peak_reserved_bytes':peak,'seconds_per_step':[row['seconds'] for row in timing]}
    write_json(verification/'verification_summary.json',verification_summary)
    if original['kl_nats']!=0 or not original['choice_log_probabilities_bitwise_equal'] or check['steps']!=3:
        raise ValueError('TT-GC real-model verification failed')
    if peak>65_000_000_000:
        raise RuntimeError('single-GPU acceptance did not leave enough memory headroom')
    single_card_minimum = max(40000, int(peak/2**20)+4096)
    public['verification'] = verification_summary

    developed = phase_dir('s1',dev)
    env = environment(developed,'tt_gc','5',fundus)
    shutil.copyfile(source/dev['development_reference_aggregates'][0],
                    developed/dev['development_reference_aggregates'][0])
    phase('development_tt_gc',[train(env)],['5'],minimum=single_card_minimum)
    phase('development_evaluation',[evaluate(env,'tt_gc',model,developed/'model')],['5'])
    write_json(developed/'grounding_summary.json',audit_summary(developed))
    d = json.loads((developed/'gate_result.json').read_text())
    full_ref = replication.parent/dev['full_parameter_reference_run']/'model'
    ref_workers,ref_dirs = [],[]
    for index,(arm,checkpoint,gpu) in enumerate((('frozen',model,'3'),('full_tt_v0',str(full_ref),'5'))):
        folder = phase_dir(f'd{index}',dev)
        ref_dirs.append(folder)
        ref_workers.append(evaluate({**env,'OUT':str(folder),'CUDA_VISIBLE_DEVICES':gpu},arm,checkpoint))
    phase('development_reference_counterfactual',ref_workers,gpus)
    public['development'] = {'gate':d,'tt_gc':json.loads((developed/'evaluation.json').read_text()),
                             'grounding':audit_summary(developed),
                             'reference_counterfactuals':[json.loads((folder/'evaluation.json').read_text())
                                                         for folder in ref_dirs]}
    if not d['passed']:
        finish('development_failed','stop_and_report_no_self_retuning')
        return

    stream = replication/'private'/'stream'
    arms = conf['training_arms']
    folders = [phase_dir(f's{k+2}',conf) for k in range(len(arms))]
    environments = [environment(folder,arm,gpu,stream)
                    for folder,arm,gpu in zip(folders,arms,gpus,strict=True)]
    phase('confirmation_training',[train(env) for env in environments],gpus,minimum=single_card_minimum)
    phase('confirmation_evaluation',[evaluate(env,arm,model,folder/'model')
          for env,arm,folder in zip(environments,arms,folders,strict=True)],gpus)
    grounded = folders[arms.index(conf['decision_arm'])]
    write_json(grounded/'grounding_summary.json',audit_summary(grounded))
    result = json.loads((grounded/'gate_result.json').read_text())
    public['confirmation'] = {'gate':result,'arms':[json.loads((folder/'evaluation.json').read_text())
                                                   for folder in folders],
                              'grounding':audit_summary(grounded)}
    finish('confirmation_passed' if result['passed'] else 'confirmation_failed',
           conf['on_success'] if result['passed'] else conf['on_failure'])


def main():
    out = Path(os.environ['OUT'])
    try:
        run(out)
    except Exception as error:
        write_json(out/'failure.json',{'type':type(error).__name__,'message':str(error)})
        raise


if __name__ == '__main__':
    main()

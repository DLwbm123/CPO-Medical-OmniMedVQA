"""Run one pilot arm under an independent resource envelope."""
import json
import os
from pathlib import Path
import socket
import time

from ..score_retained_checkpoints import write_json
from ..supervise_semantic_replay import run_workers


def main():
    out = Path(os.environ['OUT'])
    spec = json.loads((out/'protocol.json').read_text())
    arm = os.environ['TT_ARM']
    gpus = os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
    if spec['stage'] != 'B' or arm not in ('ttrl','tt_v0') or len(gpus) != spec['gpu_count']:
        raise ValueError('pilot training must use the frozen GPU count')
    if gpus not in (['5'],['3','5']):
        raise ValueError('pilot training is limited to the authorized GPUs')
    original = out/'original_acceptance_clock.json'
    if spec.get('acceptance_updates') and original.exists():
        clock = json.loads(original.read_text())
        started = clock['started']
        deadline = min(clock['deadline'],started + min(spec['max_wall_seconds'],23*3600))
    else:
        started = time.time()
        deadline = started + min(spec['max_wall_seconds'],23*3600)
    if time.time() >= deadline:
        raise ValueError('pilot time budget is already exhausted')
    with (out/'clock.json').open('x') as stream:
        json.dump({'started':started,'deadline':deadline},stream)
    worker_limits = {**spec,'output_guard_bytes':min(spec['training_output_guard_bytes'],48_000_000_000)}
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0))
        port = str(listener.getsockname()[1])
    workers = [dict(MODULE='clinical_pact.testtime.pilot_train',TT_DATA='',
                    WORLD_SIZE=str(len(gpus)),RANK=str(rank),LOCAL_RANK=str(rank),
                    MASTER_ADDR='127.0.0.1',MASTER_PORT=port) for rank in range(len(gpus))]
    training = run_workers(out,worker_limits,0,workers,deadline)
    phases = [training]
    if training['reason'] == 'completed' and not spec.get('acceptance_updates'):
        before = out/'validation'
        before.mkdir()
        phases.append(run_workers(out,worker_limits,1,[{'MODULE':'clinical_pact.testtime.pilot_eval',
            'TT_EVAL_WORKER':'1','TT_CHECKPOINT':str(out/'model'),'TT_EVAL_OUT':str(before),
            'CUDA_VISIBLE_DEVICES':'5','WORLD_SIZE':'1','RANK':'0','LOCAL_RANK':'0'}],deadline))
    write_json(out/'status.json',{'reason':phases[-1]['reason'],'phases':phases})
    if phases[-1]['reason'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

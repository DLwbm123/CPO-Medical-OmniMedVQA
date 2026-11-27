"""Bounded sequential calibration, training, and checkpoint evaluation."""
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import time

from .score_retained_checkpoints import write_json

BOOTSTRAP = 'import os,runpy;runpy.run_module(os.environ["MODULE"],run_name="__main__")'


def run_workers(out, protocol, index, environments, deadline):
    jobs = []
    started = time.monotonic()
    base = os.environ.copy()
    for worker, overrides in enumerate(environments):
        with (out / f'p{index}_w{worker}.log').open('x') as stream:
            jobs.append(subprocess.Popen(['python', '-c', BOOTSTRAP], env={**base, **overrides},
                                         start_new_session=True, stdout=stream, stderr=subprocess.STDOUT))
    reason = 'completed'
    while any(job.poll() is None for job in jobs):
        if time.time() >= deadline:
            reason = 'time_limit'
        elif any(job.poll() not in (None, 0) for job in jobs):
            reason = 'worker_failure'
        else:
            try:
                size = sum(path.stat().st_blocks * 512 for path in out.rglob('*') if path.is_file())
                if size >= protocol['output_guard_bytes']:
                    reason = 'storage_limit'
            except OSError:
                reason = 'storage_stat_failed'
        if reason != 'completed':
            break
        time.sleep(4)
    for job in jobs:
        if job.poll() is None:
            try:
                os.killpg(job.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    codes = []
    for job in jobs:
        try:
            codes.append(job.wait(timeout=10))
        except subprocess.TimeoutExpired:
            os.killpg(job.pid, signal.SIGKILL)
            codes.append(job.wait())
    if any(codes) and reason == 'completed':
        reason = 'worker_failure'
    return {'phase': index, 'reason': reason, 'codes': codes, 'seconds': time.monotonic() - started}


def main():
    out, root = (Path(os.environ[key]) for key in ('OUT', 'ROOT'))
    protocol = json.loads((out / 'protocol.json').read_text())
    with (out / 'clock.json').open('x') as stream:
        json.dump({'started': time.time(), 'deadline': time.time() + protocol['max_wall_seconds']}, stream)
    deadline = json.loads((out / 'clock.json').read_text())['deadline']
    resource.setrlimit(resource.RLIMIT_FSIZE, (10_000_000_000, 10_000_000_000))
    history = []
    train_module = 'clinical_pact.semantic_replay'
    eval_module = 'clinical_pact.score_answer_diagnostics'
    physical = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if len(physical) != 2:
        raise ValueError('requires exactly two assigned physical GPUs')

    def phase(workers):
        index = len(history)
        write_json(out / 'progress.json', {'phase': index, 'state': 'running'})
        result = run_workers(out, protocol, index, workers, deadline)
        history.append(result)
        write_json(out / 'history.json', history)
        return result['reason'] == 'completed'

    if phase([{'MODULE': train_module, 'PHASE': 'calibrate', 'CUDA_VISIBLE_DEVICES': physical[0]}]):
        for index, arm in enumerate(protocol['arms'], 1):
            if time.time() >= deadline:
                break
            free = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free',
                                            '--format=csv,noheader,nounits'], text=True)
            sizes = {gpu.strip(): int(size) for gpu, size in (line.split(',') for line in free.splitlines())}
            if any(sizes[gpu] < 60000 for gpu in physical):
                history.append({'phase': len(history), 'reason': 'insufficient_free_memory'})
                break
            stage = f's{index}'
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                port = str(listener.getsockname()[1])
            settings = dict(MODULE=train_module, PHASE='train', ARM=arm, STAGE_DIR=stage,
                            WORLD_SIZE='2', MASTER_ADDR='127.0.0.1', MASTER_PORT=port)
            if not phase([dict(settings, RANK=str(rank), LOCAL_RANK=str(rank)) for rank in range(2)]):
                break
            destination = out / stage / 'validation'
            (destination / 'private').mkdir(parents=True)
            settings = dict(MODULE=eval_module, EVAL_DIR=str(destination))
            evaluation = {'checkpoints': [{'name': stage, 'path': str((out / stage / 'model').relative_to(root))}] * 2,
                          'tasks': [{'name': 'MedicalVQA_MedBook', 'n': 130},
                                    {'name': 'MedicalVQA_Pathology', 'n': 6259}],
                          'seed': protocol['seed'], 'max_new_tokens': 128, 'worker_count': 2, 'sharded_checkpoint': True}
            write_json(destination / 'protocol.json', evaluation)
            if not phase([dict(settings, RANK=str(rank)) for rank in range(2)]) or not phase([settings]):
                break
    expected = 1 + 3 * len(protocol['arms'])
    reason = 'completed' if len(history) == expected and history[-1]['reason'] == 'completed' else history[-1]['reason']
    if len(history) != expected and reason == 'completed':
        reason = 'time_limit'
    write_json(out / 'status.json', {'reason': reason, 'phases': history})


if __name__ == '__main__':
    main()

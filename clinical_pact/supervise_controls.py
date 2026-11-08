"""One hard clock and byte budget for both control arms and validation."""
import json
import os
from pathlib import Path
import resource
import signal
import socket
import subprocess
import time

root = Path(os.environ['OUT'])
protocol = json.loads((root / 'protocol.json').read_text())
resource.setrlimit(resource.RLIMIT_FSIZE, (10000000000, 10000000000))
started = time.time()
with (root / 'clock.json').open('x') as stream:
    json.dump({'first_start_epoch': started, 'absolute_deadline_epoch': started+protocol['max_wall_seconds']}, stream)
history = []
reason = 'completed'
for phase, arm in [('training', 'plain_gspo'), ('training', 'fixed_margin'), ('validation', None)]:
    free = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'], text=True)
    available = {int(line.split(',')[0]): int(line.split(',')[1]) for line in free.strip().splitlines()}
    if any(available[gpu] < 60000 for gpu in protocol['GPUs']):
        reason = 'insufficient_free_memory'
        break
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    port = listener.getsockname()[1]
    listener.close()
    jobs = []
    label = arm or 'validation'
    for rank in (0, 1):
        env = os.environ.copy()
        env.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE='2', MASTER_ADDR='127.0.0.1',
                   MASTER_PORT=str(port), MODULE='clinical_pact.matched_controls' if arm else 'clinical_pact.validate_development')
        if arm:
            env['ARM'] = arm
        log = (root / f'{label}_rank{rank}.log').open('x')
        jobs.append(subprocess.Popen(['python', '-c', 'import os,runpy;runpy.run_module(os.environ["MODULE"],run_name="__main__")'],
                                     env=env, start_new_session=True, stdout=log, stderr=subprocess.STDOUT))
    while any(job.poll() is None for job in jobs):
        if time.time()-started >= protocol['max_wall_seconds']:
            reason = 'timeout'
            break
        try:
            size = sum(os.stat(os.path.join(directory, name)).st_blocks*512
                       for directory, _, names in os.walk(root) for name in names)
        except OSError:
            reason = 'storage_stat_failed'
            break
        if size >= protocol['output_guard_bytes']:
            reason = 'storage_limit'
            break
        if any(job.poll() not in (None, 0) for job in jobs):
            reason = 'worker_failure'
            break
        time.sleep(1)
    if reason == 'completed' and any(job.poll() not in (None, 0) for job in jobs):
        reason = 'worker_failure'
    if reason != 'completed':
        for job in jobs:
            if job.poll() is None:
                try:
                    os.killpg(job.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for job in jobs:
            try:
                job.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(job.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                job.wait()
    history.append({'phase': phase, 'arm': arm, 'codes': [job.wait() for job in jobs]})
    with (root / 'phase_history.json').open('w') as stream:
        json.dump(history, stream, indent=2)
    if reason != 'completed':
        break
with (root / 'status.json').open('x') as stream:
    json.dump({'reason': reason, 'phases': history, 'elapsed_seconds': time.time()-started}, stream)

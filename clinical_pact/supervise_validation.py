"""Aggregate-only validation extension under the original experiment limits."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

root = Path(os.environ['OUT'])
destination = root / 'validation'
protocol = json.loads((destination / 'protocol.json').read_text())
resource.setrlimit(resource.RLIMIT_FSIZE, (100000000, 100000000))
jobs = []
started = time.time()
assert started < protocol['absolute_deadline_epoch']
for rank in (0, 1):
    env = os.environ.copy()
    env['RANK'] = str(rank)
    log = (destination / f'worker{rank}.log').open('x')
    jobs.append(subprocess.Popen(['python', '-c', 'import os,runpy;runpy.run_module(os.environ["MODULE"],run_name="__main__")'],
                                 env=env, start_new_session=True, stdout=log,
                                 stderr=subprocess.STDOUT))
reason = 'completed'
while any(job.poll() is None for job in jobs):
    if time.time() >= protocol['absolute_deadline_epoch']:
        reason = 'timeout'
        break
    try:
        total = sum(os.stat(os.path.join(directory, name)).st_blocks*512
                    for directory, _, names in os.walk(root) for name in names)
        local = sum(file.stat().st_blocks*512 for file in destination.rglob('*') if file.is_file())
    except OSError:
        reason = 'storage_stat_failed'
        break
    if total >= 40000000000 or local >= 1000000000:
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
with (destination / 'status.json').open('x') as stream:
    json.dump({'reason': reason, 'codes': [job.wait() for job in jobs],
               'elapsed_seconds': time.time()-started}, stream)

"""Single-worker bounded supervisor for the frozen validation probe."""
import json
import os
from pathlib import Path
import time

from ..score_retained_checkpoints import write_json
from ..supervise_semantic_replay import run_workers
from .probe import VOTES


def main():
    out = Path(os.environ['OUT'])
    protocol = json.loads((out/'protocol.json').read_text())
    if protocol['stage'] != 'A' or protocol['split'] != 'validation' or protocol['votes'] != VOTES:
        raise ValueError('the probe supervisor accepts only the frozen stage A validation protocol')
    if ',' in os.environ['CUDA_VISIBLE_DEVICES']:
        raise ValueError('stage A requires exactly one assigned GPU')
    resuming = os.environ.get('TT_RESUME') == '1'
    clock_path = out/'clock.json'
    if resuming:
        clock = json.loads(clock_path.read_text())
        deadline = min(clock['deadline'], clock['started'] + protocol['max_wall_seconds'], clock['started'] + 23*3600)
        if time.time() >= deadline:
            raise ValueError('original stage A time budget has expired')
    else:
        started = time.time()
        deadline = started + min(protocol['max_wall_seconds'], 23*3600)
        with clock_path.open('x') as stream:
            json.dump(dict(started=started, deadline=deadline), stream)
    module = 'clinical_pact.testtime.probe'
    result = run_workers(out, protocol, int(resuming), [{'MODULE': module}], deadline)
    write_json(out/'status.json', result)
    if result['reason'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

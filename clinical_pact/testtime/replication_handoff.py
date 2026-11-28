"""Finalize the amended one-arm replication once its frozen supervisor exits."""
import json
import os
from pathlib import Path
import time

from ..score_retained_checkpoints import write_json


def main():
    out = Path(os.environ['OUT']).resolve()
    original = int((out/'launcher.pid').read_text())
    clock = json.loads((out/'clock.json').read_text())
    cancellation = json.loads((out/'s2'/'cancelled.json').read_text())
    if cancellation['arm'] != 'image_gate' or (out/'s2'/'training_result.json').exists():
        raise ValueError('only the unstarted image-gate arm may be cancelled')
    while time.time() < clock['deadline']:
        try:
            if Path(f'/proc/{original}/cwd').resolve(strict=True) != out:
                break
        except FileNotFoundError:
            break
        time.sleep(10)
    else:
        raise RuntimeError('live replication exceeded its original deadline')
    original_status = json.loads((out/'status.json').read_text())
    aggregate = out/'tt_v0_aggregate.json'
    if (not aggregate.is_file()
            or not any(row['name']=='eval_tt_v0' and row['reason']=='completed'
                       for row in original_status['phases'])):
        write_json(out/'amended_status.json',{'reason':'tt_v0_not_completed',
                   'original_status':original_status,'cancellation':cancellation})
        return
    result = json.loads(aggregate.read_text())
    write_json(out/'amended_replication_result.json', {
        'task':'Dermoscopy','role':'baseline_and_mechanism_diagnostic',
        'stage_b_original_decision':'failed','direction':'deferred_to_grounded_protocol',
        'split':json.loads((out/'split_summary.json').read_text()),'arms':{'tt_v0':result},
        'cancelled_arms':{'image_gate':cancellation}})
    write_json(out/'amended_status.json',{'reason':'completed_with_cancelled_arm',
               'original_completed_phases':original_status['phases'],'cancellation':cancellation})


if __name__ == '__main__':
    main()

"""Offline conversion into the whitelist-only training input format."""
import json
import os
from pathlib import Path

from .tt_data import load_unlabeled


def redact(source, target, seed):
    records = json.loads((source/'jsons'/'train'/'data.json').read_text())
    clean = []
    for record in records:
        prompt = record['conversations'][0]
        if prompt['from'] != 'human':
            raise ValueError('training records must start with the user question')
        clean.append({'id': str(record['id']), 'image': record['image'], 'question': prompt['value'],
                      'source_dataset': record['source_dataset']})
    output = target/'unlabeled'/'train'
    output.mkdir(parents=True, exist_ok=False)
    (target/'images').symlink_to((source/'images').resolve(strict=True),target_is_directory=True)
    with (output/'data.json').open('x') as stream:
        json.dump(clean,stream,ensure_ascii=False)
    return len(load_unlabeled(target,'train',seed))


if __name__ == '__main__':
    source = Path(os.environ['TT_RAW_TASK'])
    target = Path(os.environ['TT_REDACTED_TASK'])
    print({'redacted_items': redact(source,target,100)})

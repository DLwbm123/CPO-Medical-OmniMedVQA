"""Private local dataset access. No silent image/path fallback is allowed."""
import json
from pathlib import Path


def load_split(task_directory, split):
    if split not in {'train', 'validation', 'test'}:
        raise ValueError('unknown split')
    root = Path(task_directory).resolve()
    image_root = (root / 'images').resolve()
    with (root / 'jsons' / split / 'data.json').open(encoding='utf-8') as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise ValueError('split must be a list')
    for row in rows:
        if not isinstance(row.get('image'), str) or not row['image']:
            raise ValueError('each sample must have an image path')
        image = (image_root / row['image']).resolve()
        if not image.is_relative_to(image_root) or not image.is_file():
            raise ValueError('image is missing or escapes the dataset image directory')
        conversations = row.get('conversations')
        if not isinstance(conversations, list) or len(conversations) != 2:
            raise ValueError('expected question and answer conversations')
        if not all(isinstance(item, dict) and isinstance(item.get('value'), str) for item in conversations):
            raise ValueError('invalid conversation')
        yield {'id': row['id'], 'image': image, 'question': conversations[0]['value'], 'answer': conversations[1]['value']}

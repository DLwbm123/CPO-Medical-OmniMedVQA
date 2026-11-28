"""Separate redacted training input from labeled evaluation input."""
from dataclasses import dataclass
import json
from pathlib import Path
import random
import re

from ..scoring import normalize


OPTION = re.compile(r'(?m)^([A-D])\. +(?=\S)')


@dataclass(frozen=True)
class UnlabeledItem:
    id: str
    image: Path
    question: str
    options: tuple[str, ...]
    letters: str
    source_dataset: str

    @property
    def duplicate_options(self):
        return len(set(map(normalize, self.options))) != len(self.options)


def parse_options(question):
    markers = list(OPTION.finditer(question))
    letters = ''.join(m[1] for m in markers)
    if len(letters) not in (2,3,4) or letters != 'ABCD'[:len(letters)]:
        raise ValueError('expected two to four consecutive answer options')
    ends = [m.start() for m in markers[1:]] + [len(question)]
    options = tuple(question[m.end():end].strip() for m,end in zip(markers,ends,strict=True))
    if not all(options):
        raise ValueError('expected nonempty option text')
    return options, letters


def _item(row, root, question):
    image = (root/row['image']).resolve()
    if not image.is_relative_to(root) or not image.is_file():
        raise ValueError('image missing or outside its frozen root')
    options, letters = parse_options(question)
    return UnlabeledItem(str(row['id']),image,question,options,letters,row['source_dataset'])


def load_unlabeled(task_dir, split, seed, exclude_ids=()):
    if split not in ('train','validation'):
        raise ValueError('redacted input supports only train and validation')
    root = (task_dir/'images').resolve()
    path = task_dir/'unlabeled'/split/'data.json'
    rows = json.loads(path.read_text())
    fields = {'id','image','question','source_dataset'}
    excluded = set(exclude_ids)
    result = []
    for row in rows:
        if set(row) != fields:
            raise ValueError('redacted input must contain only its four whitelisted fields')
        if str(row['id']) not in excluded:
            result.append(_item(row,root,row['question']))
    if len({x.id for x in result}) != len(result):
        raise ValueError('duplicate unlabeled question IDs')
    random.Random(seed).shuffle(result)
    return result


def load_labeled_eval(task_dir, split):
    if split not in ('validation','test'):
        raise ValueError('evaluation can read validation or test labels only')
    root = (task_dir/'images').resolve()
    rows = json.loads((task_dir/'jsons'/split/'data.json').read_text())
    result = []
    for row in rows:
        question,gold = (turn['value'] for turn in row['conversations'])
        item = _item(row,root,question)
        letter = normalize(gold).upper()
        if len(letter)!=1 or letter not in item.letters:
            raise ValueError('gold is outside the legal option letters')
        result.append((item,letter))
    if len({x.id for x,_ in result}) != len(result):
        raise ValueError('duplicate evaluation question IDs')
    return result


def reserve_reference_ids(items, seed=100, count=1024):
    ids = sorted(item.id for item in items)
    if len(ids)!=len(set(ids)):
        raise ValueError('cannot reserve IDs from a duplicated stream')
    return sorted(random.Random(seed).sample(ids,count))

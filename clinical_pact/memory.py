"""Immutable per-stage greedy decision records and bounded task-balanced memory."""
from dataclasses import dataclass
from collections import defaultdict
import math


@dataclass(frozen=True)
class DecisionRecord:
    task: str
    record_id: str
    image_ref: str
    prompt_fingerprint: str
    response_ids: tuple[int, ...]
    margins: tuple[float, ...]
    eos_id: int
    priority: int
    stratum: str
    correctly_scored: bool
    complete: bool
    path_aligned: bool

    def __post_init__(self):
        if not self.task or not self.record_id or not self.image_ref or not self.prompt_fingerprint:
            raise ValueError('identity and input references are required')
        if not (self.correctly_scored and self.complete and self.path_aligned):
            raise ValueError('reference must be correct, complete and path-aligned')
        if not self.response_ids or self.response_ids[-1] != self.eos_id:
            raise ValueError('response must end with EOS')
        if len(self.response_ids) != len(self.margins):
            raise ValueError('one margin required per response token including EOS')
        if any(not math.isfinite(m) or m < 0.05 for m in self.margins):
            raise ValueError('reference margins must be finite and >= 0.05')


class DecisionMemory:
    def __init__(self, capacity=512):
        if capacity <= 0:
            raise ValueError('capacity must be positive')
        self.capacity = capacity
        self.tasks = {}

    def add_task(self, task, records):
        if task in self.tasks:
            raise ValueError('immutable task references cannot be replaced')
        records = list(records)
        if not records or any(r.task != task for r in records):
            raise ValueError('task needs eligible records with matching task ID')
        if len({r.record_id for r in records}) != len(records):
            raise ValueError('duplicate record ID')
        self.tasks[task] = records
        quota = self.capacity // len(self.tasks)
        if quota == 0:
            del self.tasks[task]
            raise ValueError('capacity too small for task-balanced memory')
        for key in self.tasks:
            self.tasks[key] = self._select(self.tasks[key], quota)

    @staticmethod
    def _select(records, quota):
        groups = defaultdict(list)
        for record in records:
            groups[record.stratum].append(record)
        for values in groups.values():
            values.sort(key=lambda x: (x.priority, x.record_id))
        selected = []
        while len(selected) < quota:
            progress = False
            for key in sorted(groups):
                if groups[key] and len(selected) < quota:
                    selected.append(groups[key].pop(0))
                    progress = True
            if not progress:
                break
        return selected

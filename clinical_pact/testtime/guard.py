"""Sliding letter-distribution guard for label-free stage B updates."""
from collections import Counter, deque


def total_variation(first, second):
    a, b = sum(first.values()), sum(second.values())
    if not a or not b:
        return None
    return 0.5 * sum(abs(first.get(x,0)/a-second.get(x,0)/b) for x in 'ABCD')


class Guard:
    def __init__(self, reference_counts, *, window=50, limit=.15, consecutive_limit=20):
        if not sum(reference_counts.values()):
            raise ValueError('guard reference has no legal forced choices')
        self.reference = reference_counts
        self.window = deque(maxlen=window)
        self.limit = limit
        self.consecutive_limit = consecutive_limit
        self.consecutive = 0

    def observe_update(self, vote_counts):
        self.window.extend(vote_counts)
        counts = Counter()
        for question in self.window:
            counts.update(question)
        tv = total_variation(counts,self.reference)
        brake = tv is not None and tv > self.limit
        self.consecutive = self.consecutive + 1 if brake else 0
        return tv, brake, self.consecutive > self.consecutive_limit

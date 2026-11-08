"""Deterministic strategy-card allocation and training-only candidate selection."""
from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class Candidate:
    text: str
    token_length: int
    scores: tuple[int, ...]

    def __post_init__(self):
        if self.token_length < 0 or self.token_length > 96:
            raise ValueError('strategy must be at most 96 tokens')
        if not self.scores or any(x not in (0, 1) for x in self.scores):
            raise ValueError('candidate requires binary training scores')

    @property
    def tie_key(self):
        return self.token_length, hashlib.sha256(self.text.encode('utf-8')).hexdigest()


def allocate_group(group_id, canonical, card_a, card_b):
    """Assign exactly four canonical, two A and two B conditions to one occurrence."""
    if not group_id or not canonical:
        raise ValueError('group identity and canonical prompt are required')
    return [(group_id, tag, text) for tag, text, count in (
        ('canonical', canonical, 4), ('card_a', card_a, 2), ('card_b', card_b, 2)
    ) for _ in range(count)]


def choose_cards(canonical, candidates):
    """Choose by accuracy and complementary training coverage; empty means fallback."""
    candidates = list(candidates)
    n = len(canonical.scores)
    if any(len(candidate.scores) != n for candidate in candidates):
        raise ValueError('candidate scores must use identical training questions')
    baseline = sum(canonical.scores)
    eligible = [c for c in candidates if sum(c.scores) > baseline and c.text.strip()]
    if not eligible:
        return '', ''
    first = min(eligible, key=lambda c: (-sum(c.scores), c.tie_key))
    rest = [c for c in eligible if c.text != first.text]
    if not rest:
        return first.text, ''
    second = min(rest, key=lambda c: (-sum(a or b for a, b in zip(first.scores, c.scores)), c.tie_key))
    return first.text, second.text

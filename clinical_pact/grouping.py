"""Validate cross-prompt rollout identity before computing group advantages."""
from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Rollout:
    group_id: tuple
    strategy_id: str
    prompt_fingerprint: str
    logprob_prompt_fingerprint: str
    reward: float


def validate_groups(rollouts):
    groups = defaultdict(list)
    for r in rollouts:
        if not r.group_id or not r.prompt_fingerprint or r.prompt_fingerprint != r.logprob_prompt_fingerprint:
            raise ValueError('rollout and logprob contexts must match')
        if r.reward not in (0.0, 1.0):
            raise ValueError('frozen reward must be binary')
        groups[r.group_id].append(r)
    if not groups:
        raise ValueError('no rollout groups')
    for group in groups.values():
        if Counter(r.strategy_id for r in group) != {'canonical': 4, 'card_a': 2, 'card_b': 2}:
            raise ValueError('each question occurrence needs canonical/card_a/card_b counts 4/2/2')
    return groups

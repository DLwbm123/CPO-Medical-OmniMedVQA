"""Construct an audit-only sample of a previously consumed unlabeled stream."""
import random


def audit_items(used_items, replay_ids, count, seed):
    """Choose a fixed cohort without labels, disjoint from the replay bank."""
    excluded = set(replay_ids)
    candidates = [item for item in used_items if item.id not in excluded and not item.duplicate_options]
    ids = {item.id for item in candidates}
    if len(ids) != len(candidates):
        raise ValueError('audit candidates must have unique item IDs')
    return tuple(random.Random(seed).sample(sorted(candidates,key=lambda item: item.id),
                                            min(count,len(candidates))))

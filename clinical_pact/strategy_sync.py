"""Rank-zero strategy refresh synchronization at a drained optimizer boundary.

The caller must collectively gather/unshard policy parameters beforehand and
clear consumed rollout buffers afterward. The callbacks must not invoke
collectives themselves. This helper does not change Trainer internals.
"""
import torch.distributed as dist
from .strategy_cycle import StrategyCycle


def synchronized_refresh(cycle, update, *, callback_factory=None, **callbacks):
    if not dist.is_initialized():
        candidate = StrategyCycle.restore(cycle.checkpoint())
        candidate.refresh(update, **(callback_factory() if callback_factory is not None else callbacks))
        return candidate
    local = {}
    try:
        local['state'] = cycle.checkpoint()
    except Exception as error:
        local['error_type'] = type(error).__name__
    states = [None] * dist.get_world_size()
    dist.all_gather_object(states, local)
    if any('error_type' in state for state in states):
        raise ValueError('a rank has undrained or invalid strategy state')
    if any(state != states[0] for state in states):
        raise ValueError('strategy state differs between ranks')
    result = [None]
    if dist.get_rank() == 0:
        try:
            candidate = StrategyCycle.restore(states[0]['state'])
            candidate.refresh(update, **(callback_factory() if callback_factory is not None else callbacks))
            result[0] = {'state': candidate.checkpoint()}
        except Exception as error:
            # All ranks receive the failure and leave together. Do not silently
            # accept a partial cycle or retry a proposer with uncharged calls.
            result[0] = {'error_type':type(error).__name__}
    dist.broadcast_object_list(result, src=0)
    if 'error_type' in result[0]:
        raise ValueError('rank-zero refresh failed: '+result[0]['error_type'])
    return StrategyCycle.restore(result[0]['state'])

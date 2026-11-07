"""Small collective primitives for PACT group and task synchronization.

These helpers do not make the TRL adapter distributed-ready. All ranks must
enter the same collective in the same order, with identical controller state.
"""
import torch
import torch.distributed as dist
from .core import question_group_advantages


def distributed_group_advantages(rewards, group_ids, expected_group_size=8):
    if rewards.ndim != 1 or rewards.numel() != len(group_ids):
        raise ValueError('one group ID per local reward required')
    if not dist.is_initialized():
        return question_group_advantages(rewards, group_ids, expected_group_size=expected_group_size)
    local = (rewards.detach().float().cpu().tolist(), list(group_ids))
    parts = [None] * dist.get_world_size()
    dist.all_gather_object(parts, local)
    all_rewards = [r for values, _ in parts for r in values]
    all_ids = [i for _, ids in parts for i in ids]
    values = question_group_advantages(torch.tensor(all_rewards), all_ids, expected_group_size=expected_group_size)
    offset = sum(len(parts[i][0]) for i in range(dist.get_rank()))
    return values[offset:offset + len(local[0])].to(rewards.device)


def synchronized_task(controller, generator):
    """Rank 0 samples one task; all ranks verify the full distribution."""
    ids, probs = controller.sampling_distribution()
    if not ids:
        raise ValueError('no old tasks available')
    local = (ids, probs.tolist(), [controller.states[k].weight for k in ids])
    if not dist.is_initialized():
        index = torch.multinomial(probs, 1, generator=generator).item()
        return ids[index], float(probs[index])
    parts = [None] * dist.get_world_size()
    dist.all_gather_object(parts, local)
    if any(item != parts[0] for item in parts):
        raise ValueError('controller state differs between ranks')
    selected = [None]
    if dist.get_rank() == 0:
        selected[0] = torch.multinomial(probs, 1, generator=generator).item()
    dist.broadcast_object_list(selected, src=0)
    index = selected[0]
    return ids[index], float(probs[index])

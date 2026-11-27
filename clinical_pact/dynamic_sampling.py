"""Synchronized binary-reward group rejection for the optional GSPO branch."""
import torch
import torch.distributed as dist


class UniformGroup(Exception):
    """Restart the rollout before computing a retention reward or advantages."""


def uniform_rewards(raw, device):
    if not raw or any(value not in (0, 1) for value in raw):
        raise ValueError('dynamic sampling requires a nonempty group of binary verifier rewards')
    seen = torch.tensor([int(0 in raw), int(1 in raw)], device=device)
    dist.all_reduce(seen, op=dist.ReduceOp.MAX)
    return not bool(seen.all())

"""Sparse answer Fisher, anchored quadratic energy, and ZeRO-3 gradient injection."""
import math

import torch
import torch.distributed as dist


def global_top_indices(scores, fraction):
    """Select exact global top-k coordinates, resolving ties by rank and local index."""
    if scores.ndim != 1 or not 0 < fraction <= 1:
        raise ValueError('expected a flat Fisher diagonal and a valid fraction')
    if not torch.isfinite(scores).all() or (scores < 0).any():
        raise FloatingPointError('invalid Fisher diagonal')
    counts = [None] * dist.get_world_size()
    dist.all_gather_object(counts, scores.numel())
    k = math.ceil(fraction * sum(counts))
    if k == 0:
        raise ValueError('empty Fisher diagonal')
    local = scores.topk(min(k, scores.numel()), sorted=False).values
    if len(counts) == 1:
        cutoff = local.min()
    else:
        candidates = scores.new_full((k,), -float('inf'))
        candidates[:local.numel()] = local
        gathered = scores.new_empty((k * len(counts),))
        dist.all_gather_into_tensor(gathered, candidates)
        cutoff = gathered.kthvalue(gathered.numel() - k + 1).values
    strict = (scores > cutoff).nonzero(as_tuple=True)[0]
    tied = (scores == cutoff).nonzero(as_tuple=True)[0]
    counts = [None] * len(counts)
    dist.all_gather_object(counts, (strict.numel(), tied.numel()))
    n = max(0, k - sum(x for x, _ in counts) - sum(y for _, y in counts[:dist.get_rank()]))
    selected = torch.cat((strict, tied[:n])).sort().values
    total = scores[selected].sum(dtype=torch.float64)
    dist.all_reduce(total)
    mean = total / k
    if not bool(torch.isfinite(mean) & (mean > 0)):
        raise ValueError('selected Fisher coordinates have zero or invalid mean')
    return selected, mean


def merge_fisher(previous, current, weights, selected, offset):
    """Merge overlap and anchor every retained coordinate at the latest endpoint."""
    if previous is None:
        indexes, diagonal = selected, weights
    else:
        prior_indices, prior_weights, _ = previous
        all_indices = torch.cat((prior_indices.to(selected.device).long() - offset, selected))
        indexes, reverse = all_indices.unique(sorted=True, return_inverse=True)
        diagonal = weights.new_zeros(len(indexes))
        n = prior_indices.numel()
        # Each source has unique coordinates, so neither assignment has duplicate indices.
        diagonal[reverse[:n]] = prior_weights.to(weights.device)
        diagonal[reverse[n:]] += weights
    if indexes.numel() and (int(indexes.min()) < 0 or int(indexes.max()) >= current.numel()):
        raise ValueError('Fisher coordinates do not belong to the expected partition')
    return (indexes.add(offset).to(dtype=torch.int32, device='cpu'),
            diagonal.cpu(), current[indexes].float().cpu())


def quadratic(delta, weights):
    energy = torch.dot(weights.double(), delta.double().square())
    return energy, 2 * weights * delta


class FisherPenalty:
    def __init__(self, path, gamma):
        self.state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        self.gamma = float(gamma)
        if not math.isfinite(self.gamma) or self.gamma <= 0:
            raise ValueError('Fisher calibration requires a positive finite gamma')
        if self.state['rank'] != dist.get_rank() or self.state['world_size'] != dist.get_world_size():
            raise ValueError('Fisher shard must match the training rank layout')

    @torch.no_grad()
    def apply(self, engine, coefficient=0, *, audit=False):
        from deepspeed.utils import safe_get_local_fp32_param, safe_get_local_grad, safe_set_local_grad

        rank = dist.get_rank()
        parameters = dict(engine.module.named_parameters())  # Tied parameters occur once.
        energy = torch.zeros((), device=engine.device, dtype=torch.float64)
        for name, (positions, diagonal, anchors) in self.state['entries'].items():
            p = parameters[name]
            current = safe_get_local_fp32_param(p).flatten()
            grad = safe_get_local_grad(p).flatten() if coefficient else None
            for indices, f, anchor in zip(positions.split(1_048_576), diagonal.split(1_048_576),
                                          anchors.split(1_048_576), strict=True):
                local = indices.to(engine.device, dtype=torch.long).sub_(rank * current.numel())
                f, anchor = f.to(engine.device), anchor.to(engine.device)
                delta = current[local] - anchor
                if audit:
                    energy += quadratic(delta, f)[0]
                if grad is not None:
                    grad[local] = grad[local].add_(2 * f * delta, alpha=coefficient * self.gamma)
            if grad is not None:
                safe_set_local_grad(p, grad)
        if audit:
            dist.all_reduce(energy)
            if not torch.isfinite(energy):
                raise FloatingPointError('nonfinite Fisher energy')
            return float(energy)

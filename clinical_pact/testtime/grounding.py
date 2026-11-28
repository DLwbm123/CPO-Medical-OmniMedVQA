"""Grounding weights and blind-input control on legal option logits."""
import math


def evidence_weight(real_logp, blind_logp, ratio, temperature):
    evidence = real_logp-blind_logp
    z = (evidence-math.log(ratio))/temperature
    weight = 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))
    return evidence,weight


def grounded_advantages(batch, weights, confidences, width, min_weight, confidence_cap):
    import torch

    incoming = batch['advantages']
    if len(weights)*width != incoming.numel() or len(confidences) != len(weights):
        raise ValueError('expected one grounding weight and confidence per question')
    factor = [weight if weight >= min_weight and p < confidence_cap else 0.
              for weight,p in zip(weights,confidences,strict=True)]
    incoming.mul_(torch.tensor(factor,dtype=incoming.dtype,device=incoming.device).repeat_interleave(width))
    return not bool(incoming.count_nonzero())


def blind_kl(current, reference):
    if current.shape != reference.shape:
        raise ValueError('current and deployment choices must share the legal option space')
    return (current.exp()*(current-reference.detach())).sum(dim=-1)


def update_dual(current, estimate, cfg):
    budget = cfg['blind_kl_budget_nats']
    if estimate > budget:
        current *= cfg['dual_increase_factor']
    elif estimate < budget*cfg['dual_low_budget_fraction']:
        current *= cfg['dual_decrease_factor']
    return min(cfg['dual_max'],max(cfg['dual_min'],current))

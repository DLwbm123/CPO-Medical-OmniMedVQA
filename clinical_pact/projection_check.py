"""Validate BF16 ZeRO-3 against analytic gradients, including padded partitions."""
import json
import os
from pathlib import Path

import deepspeed
from deepspeed.utils import (safe_get_full_grad, safe_get_local_fp32_param, safe_get_local_grad,
                             safe_set_local_fp32_param)
import torch
import torch.distributed as dist

from .decision_fisher import FisherPenalty, shard_bounds
from .shard_projection import combine_replay_gradient


def analytic_grad(weights, x):
    first, second = weights
    x = x.float().flatten()
    return [second.sum(0).outer(x).flatten(), (first @ x).repeat(second.size(0))]


def check_fisher(engine, params, weights, rank, world, dtype, out):
    inputs = [torch.tensor([[i+1, 0.5*i, 1-i]], device=engine.device, dtype=dtype) for i in range(4)]
    gradients = list(zip(*(analytic_grad(weights, x) for x in inputs), strict=True))
    diagonal = [torch.zeros_like(safe_get_local_fp32_param(p)) for _,p in params]
    for x in inputs:
        engine.optimizer.zero_grad(set_to_none=True)
        engine.set_gradient_accumulation_boundary(True)
        engine.backward(engine(x).sum(), scale_wrt_gas=False)
        for (_,p), f in zip(params, diagonal, strict=True):
            f.add_(safe_get_local_grad(p).square() / len(inputs))

    entries, targets, padding, tails = {}, {}, {}, {}
    expected_force_sq = 0.
    for (name,p), local_f, samples in zip(params, diagonal, gradients, strict=True):
        start, end = shard_bounds(p, rank)
        size = local_f.numel()
        valid = end - start
        expected_diagonal = torch.stack(samples).square().mean(0)
        torch.testing.assert_close(local_f[:valid], expected_diagonal[start:end], rtol=0, atol=0)
        tails[name] = safe_get_local_grad(p)[valid:].clone()
        padding[name] = size * world - p.ds_numel
        master = safe_get_local_fp32_param(p).clone()
        changed = master.clone()
        changed[:valid].add_(0.00002)
        safe_set_local_fp32_param(p, changed)
        local_force = torch.zeros_like(master)
        local_force[:valid] = (changed[:valid]-master[:valid]) * local_f[:valid] * (2*.5*2048.)
        slots = torch.empty(size * world, device=engine.device)
        dist.all_gather_into_tensor(slots, local_force)
        force = slots[:p.ds_numel]
        expected_force_sq += float(force.double().square().sum())
        targets[name] = (safe_get_full_grad(p).flatten() + force).to(dtype).float()
        indices = torch.arange(start, end, dtype=torch.int32)
        entries[name] = (indices, local_f[:valid].cpu(), master[:valid].cpu())

    if world == 2 and padding != {'0.weight':1, '1.weight':0}:
        raise ValueError('expected exactly one padded coordinate in the 15-element parameter')
    path = out/f'test_shard_{rank}.pt'
    torch.save({'rank':rank,'world_size':world,'entries':entries}, path)
    penalty = FisherPenalty(path, gamma=2048.)
    measured_energy = penalty.apply(engine, audit=True)
    force_norm = penalty.apply(engine, .5)
    if measured_energy <= 0:
        raise ValueError('Fisher must register a nonzero master-weight displacement')
    torch.testing.assert_close(force_norm, expected_force_sq**.5)
    error = 0.
    for name,p in params:
        target = targets[name]
        actual = safe_get_full_grad(p).flatten()
        error = max(error, float((actual-target).abs().max()))
        torch.testing.assert_close(actual, target, rtol=0, atol=0)
        start, end = shard_bounds(p, rank)
        torch.testing.assert_close(safe_get_local_grad(p)[end-start:], tails[name], rtol=0, atol=0)
    return {'samples':len(inputs),'parameter_padding':padding,'injected_gradient_max_abs_error':error,
            'omega_f':measured_energy,'fisher_force_norm':force_norm}


def main():
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', rank)))
    torch.cuda.set_per_process_memory_fraction(0.025)
    dist.init_process_group('nccl')
    world = dist.get_world_size()
    if world not in (1, 2):
        raise ValueError('the test requires one or two actual ZeRO ranks')
    dtype = torch.bfloat16
    network = torch.nn.Sequential(torch.nn.Linear(3, 5, bias=False),
                                  torch.nn.Linear(5, 2, bias=False)).to(device='cuda', dtype=dtype)
    weights = [torch.tensor([[1,0,0],[0,1,0],[0,0,1],[-1,0,0],[0,0,1]], device='cuda', dtype=torch.float32),
               torch.full((2,5), .5, device='cuda')]
    with torch.no_grad():
        for p, w in zip(network.parameters(), weights, strict=True):
            p.copy_(w)
    opt = torch.optim.AdamW(network.parameters(), lr=0., weight_decay=0., foreach=False)
    config = {'train_micro_batch_size_per_gpu':1,'gradient_accumulation_steps':4,
              'zero_optimization':{'stage':3,'overlap_comm':False,'contiguous_gradients':True},
              'zero_allow_untested_optimizer':True,'bf16':{'enabled':True}}
    engine, *_ = deepspeed.initialize(model=network, optimizer=opt, config=config)
    params = list(engine.module.named_parameters())
    device = engine.device
    coef = .2
    full = []
    per_rank = 8 // world
    for sign, project in ((1,False),(-1,True),(1,True)):
        x = torch.tensor([[1+rank,.5,2]], device=device, dtype=dtype)
        views = [torch.tensor([[sign*(1+i+rank),1+i%3,.5*(i-2)]], device=device, dtype=dtype) for i in range(per_rank)]
        for micro in range(4):
            engine.set_gradient_accumulation_boundary(micro == 3)
            engine.backward(engine(x).sum())

        explicit_new = torch.cat(analytic_grad(weights,x))
        explicit_mem = torch.stack([torch.cat(analytic_grad(weights,view)) for view in views]).mean(0)
        for g in (explicit_new,explicit_mem):
            dist.all_reduce(g)
            g.div_(world)
        original = torch.cat([safe_get_full_grad(p).flatten() for _,p in params])
        torch.testing.assert_close(original, explicit_new)
        correction = max(0., -float(explicit_new @ explicit_mem) / float(explicit_mem @ explicit_mem)) if project else 0.
        expected = explicit_new.add(explicit_mem,alpha=coef+correction).to(dtype).float()
        outcome = combine_replay_gradient(engine,[lambda view=view:engine(view).sum() for view in views],
                                          coef,project=project)
        actual = torch.cat([safe_get_full_grad(p).flatten() for _,p in params])
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=5e-7)
        torch.testing.assert_close(outcome['memory_norm'],float(explicit_mem.norm()),rtol=2e-6,atol=5e-7)
        if outcome['projection_triggered'] != bool(correction):
            raise ValueError('known gradient geometry disagrees with the projection flag')
        full.append({'conflict':sign<0,'projection':project,'max_abs_error':float((actual-expected).abs().max())})
        engine.step()
    destination = Path(os.environ['OUT'])
    fisher = check_fisher(engine, params, weights, rank, world, dtype, destination)
    if rank == 0:
        (destination/'projection_check.json').write_text(json.dumps({'world_size':world,'gradient_accumulation_steps':4,
            'replay_losses_per_rank':per_rank,'dtype':str(dtype),'cases':full,'fisher':fisher},indent=2)+'\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

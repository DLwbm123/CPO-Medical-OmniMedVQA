"""Compare ZeRO-3 projection against explicitly assembled full gradients."""
import json
import os
from pathlib import Path

import deepspeed
from deepspeed.utils import (safe_get_full_grad, safe_get_local_fp32_param, safe_get_local_grad,
                             safe_set_local_fp32_param)
import torch
import torch.distributed as dist

from .shard_projection import combine_replay_gradient


def check_fisher(engine, param, rank, world, dtype, out):
    from .decision_fisher import FisherPenalty

    inputs = [torch.tensor([[i+1, 0.5*i, 1-i]], device=engine.device, dtype=dtype) for i in range(4)]
    expected = torch.cat(inputs).float().square().mean(0).repeat(2)
    diagonal = torch.zeros_like(safe_get_local_fp32_param(param))
    for x in inputs:
        engine.optimizer.zero_grad(set_to_none=True)
        engine.set_gradient_accumulation_boundary(True)
        engine.backward(engine(x).sum(), scale_wrt_gas=False)
        diagonal.add_(safe_get_local_grad(param).square() / len(inputs))
    size = diagonal.numel()
    slots = torch.empty(size*world, device=engine.device)
    dist.all_gather_into_tensor(slots, diagonal)
    torch.testing.assert_close(slots, expected, rtol=0, atol=0)
    ready = safe_get_local_fp32_param(param).clone()
    anchors = ready.cpu()
    displaced = ready.add(0.00002)
    safe_set_local_fp32_param(param, displaced)
    original = safe_get_full_grad(param).clone()
    path = out/f'test_shard_{rank}.pt'
    indices = torch.arange(rank*size, (rank+1)*size, dtype=torch.int32)
    torch.save({'rank':rank, 'world_size':world, 'entries':{'weight':(indices, diagonal.cpu(), anchors)}}, path)
    penalty = FisherPenalty(path, gamma=2048.)
    expected_force = (displaced-ready) * diagonal * (2*.5*2048.)
    dist.all_gather_into_tensor(slots, expected_force)
    target = (original.flatten() + slots).to(dtype).float()
    measured_energy = penalty.apply(engine, .5, audit=True)
    error = float((safe_get_full_grad(param).flatten()-target).abs().max())
    if measured_energy <= 0:
        raise ValueError('Fisher must register a nonzero master-weight displacement')
    torch.testing.assert_close(safe_get_full_grad(param).flatten(), target, rtol=0, atol=0)
    return {'samples':len(inputs), 'injected_gradient_max_abs_error':error, 'omega_f':measured_energy}


def main():
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', rank)))
    torch.cuda.set_per_process_memory_fraction(0.025)
    dist.init_process_group('nccl')
    world = dist.get_world_size()
    if world not in (1, 2):
        raise ValueError('the test requires one or two actual ZeRO ranks')
    torch.manual_seed(7)
    dtype = torch.bfloat16
    network = torch.nn.Linear(3, 2, bias=False).to(device='cuda', dtype=dtype)
    opt = torch.optim.AdamW(network.parameters(), lr=0.03, weight_decay=0, foreach=False)
    config = {
        'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 4,
        'zero_optimization': {'stage': 3, 'overlap_comm': False, 'contiguous_gradients': True},
        'zero_allow_untested_optimizer': True,
    }
    config['bf16'] = {'enabled': True}
    engine, *_ = deepspeed.initialize(model=network, optimizer=opt, config=config)
    params = list(engine.module.parameters())
    device = engine.device
    coef = 0.2
    full = []
    per_rank = 8 // world
    for sign, project in ((1, False), (-1, True), (1, True)):
        x = torch.tensor([[1 + rank, 0.5, 2]], device=device, dtype=dtype)
        views = [torch.tensor([[sign * (1 + i + rank), 1 + i % 3, 0.5 * (i - 2)]],
                              device=device, dtype=dtype) for i in range(per_rank)]
        for micro in range(4):
            engine.set_gradient_accumulation_boundary(micro == 3)
            engine.backward(engine(x).sum())
        # For sum(linear(x)), both rows of the weight gradient are exactly x.
        explicit_new = x.float().repeat(2, 1).flatten()
        explicit_mem = torch.stack(views).float().mean(0).repeat(2, 1).flatten()
        for g in (explicit_new, explicit_mem):
            dist.all_reduce(g)
            g.div_(world)
        torch.testing.assert_close(safe_get_full_grad(params[0]).flatten(), explicit_new)
        correction = max(0.0, -float(explicit_new @ explicit_mem) / float(explicit_mem @ explicit_mem)) if project else 0.
        scale = coef + correction
        expected = explicit_new.add(explicit_mem, alpha=scale).to(dtype).float()
        outcome = combine_replay_gradient(engine, [lambda view=view: engine(view).sum() for view in views],
                                          coef, project=project)
        actual = safe_get_full_grad(params[0]).flatten()
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=5e-7)
        torch.testing.assert_close(outcome['memory_norm'], float(explicit_mem.norm()), rtol=2e-6, atol=5e-7)
        if outcome['projection_triggered'] != bool(correction):
            raise ValueError('known gradient geometry disagrees with the projection flag')
        full.append({'conflict': sign < 0, 'projection':project, 'max_abs_error': float((actual - expected).abs().max())})
        engine.step()
    destination = Path(os.environ['OUT'])
    fisher = check_fisher(engine, params[0], rank, world, dtype, destination)
    if rank == 0:
        (destination/'projection_check.json').write_text(json.dumps({'world_size': world, 'gradient_accumulation_steps': 4,
                    'replay_losses_per_rank':per_rank, 'dtype':str(dtype), 'cases':full,'fisher':fisher}, indent=2) + '\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

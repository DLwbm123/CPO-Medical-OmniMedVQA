"""Compare ZeRO-3 projection against explicitly assembled full gradients."""
import json
import os
from pathlib import Path

import deepspeed
from deepspeed.utils import safe_get_full_grad
import torch
import torch.distributed as dist

from .shard_projection import combine_replay_gradient


def main():
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', rank)))
    dist.init_process_group('nccl')
    world = dist.get_world_size()
    if world not in (1, 2):
        raise ValueError('the test requires one or two actual ZeRO ranks')
    torch.manual_seed(7)
    dtype = torch.bfloat16 if world == 1 else torch.float32
    network = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Tanh(), torch.nn.Linear(5, 2)).to(device='cuda', dtype=dtype)
    opt = torch.optim.AdamW(network.parameters(), lr=0.03, weight_decay=0, foreach=False)
    config = {
        'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 4,
        'zero_optimization': {'stage': 3, 'overlap_comm': False, 'contiguous_gradients': True},
        'zero_allow_untested_optimizer': True,
    }
    if world == 1:
        config['bf16'] = {'enabled': True}
    engine, *_ = deepspeed.initialize(model=network, optimizer=opt, config=config)
    params = list(engine.module.parameters())
    device = engine.device
    coef = 0.2
    full = []
    for sign in (-1, 1):
        g_new = [torch.zeros_like(p) for p in params]
        with deepspeed.zero.GatheredParameters(params):
            g_new = [torch.zeros_like(p) for p in params]
            g_mem = [torch.zeros_like(p) for p in params]
        # Explicit full gradients are observed from two separate native DS backward passes.
        # The inputs differ by rank, so the expected values include distributed reduction.
        x = torch.tensor([[1.0 + rank, -0.5, 2.0]], device=device, dtype=dtype)
        for micro in range(4):
            engine.set_gradient_accumulation_boundary(micro == 3)
            engine.backward(engine(x).sum())
        g_new = [safe_get_full_grad(p).detach().clone() for p in params]
        engine.optimizer.zero_grad(set_to_none=True)
        engine.backward(sign * engine(x).sum(), scale_wrt_gas=False)
        g_mem = [safe_get_full_grad(p).detach().clone() for p in params]
        engine.optimizer.zero_grad(set_to_none=True)
        for micro in range(4):
            engine.set_gradient_accumulation_boundary(micro == 3)
            engine.backward(engine(x).sum())
        explicit_new = torch.cat([g.flatten() for g in g_new])
        explicit_mem = torch.cat([g.flatten() for g in g_mem])
        scale = coef + max(0.0, -float(explicit_new @ explicit_mem) / float(explicit_mem @ explicit_mem))
        expected = explicit_new.add(explicit_mem, alpha=scale).to(dtype).float()
        outcome = combine_replay_gradient(engine, [lambda: sign * engine(x).sum()], coef)
        actual = torch.cat([safe_get_full_grad(p).flatten() for p in params])
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=5e-7)
        if outcome['projection_triggered'] != (sign < 0):
            raise ValueError('known gradient geometry disagrees with the projection flag')
        full.append({'conflict': sign < 0, 'max_abs_error': float((actual - expected).abs().max())})
        engine.step()
    if rank == 0:
        destination = Path(os.environ['OUT']) / 'projection_check.json'
        destination.write_text(json.dumps({'world_size': world, 'gradient_accumulation_steps': 4,
                                            'dtype': str(dtype), 'cases': full}, indent=2) + '\n')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

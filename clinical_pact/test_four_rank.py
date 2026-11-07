"""Synthetic four-rank gradient and cross-prompt group verification; no data/model."""
import json
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from .core import TaskDualController
from .distributed import distributed_group_advantages, synchronized_task


def main():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl')
    device = torch.device('cuda', local_rank)
    rewards = torch.tensor([rank % 2, 1 - rank % 2], dtype=torch.float32, device=device)
    adv = distributed_group_advantages(rewards, [('one', 1)] * 2)
    all_rewards = torch.tensor([0., 1., 1., 0., 0., 1., 1., 0.], device=device)
    expected = (all_rewards - all_rewards.mean()) / (all_rewards.std() + 1e-4)
    if not torch.allclose(adv, expected[2 * rank:2 * rank + 2]):
        raise RuntimeError('cross-rank group advantages differ')
    controller = TaskDualController()
    controller.add_task('old')
    controller.states['old'].weight = 2.
    task, probability = synchronized_task(controller, torch.Generator().manual_seed(7))
    model = DistributedDataParallel(torch.nn.Linear(1, 1, bias=False, device=device), device_ids=[local_rank])
    with torch.no_grad(): model.module.weight.fill_(1.)
    x = torch.tensor([[float(rank + 1)]], device=device)
    # Both terms pass through the same DDP forward, as required for this check.
    prediction = model(x)
    rl = prediction.mean()
    auxiliary = controller.corrected_replay_loss(task, prediction.mean(), probability)
    (rl + auxiliary).backward()
    expected_grad = torch.tensor([[7.5]], device=device)
    error = (model.module.weight.grad - expected_grad).abs().max()
    if error.item() > 1e-6:
        raise RuntimeError('DDP combined-gradient mismatch')
    if rank == 0:
        with open(os.environ['RESULT'], 'x') as handle:
            json.dump({'ranks': dist.get_world_size(), 'advantage_groups': 'matched', 'task_sync': 'matched', 'gradient_max_abs_error': error.item(), 'scope': 'synthetic single-step DDP only; no actual model, data, generation, gradient accumulation, ZeRO, or scientific outcomes'}, handle)
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

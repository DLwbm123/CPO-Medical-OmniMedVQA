"""Combine full-update RL and replay gradients without gathering ZeRO-3 gradients."""
import torch
import torch.distributed as dist


def combine_replay_gradient(engine, replay_losses, coefficient, *, project=True):
    from deepspeed.utils import safe_get_local_grad, safe_set_local_grad

    optimizer = engine.optimizer
    if engine.zero_optimization_stage() != 3 or optimizer.offload_optimizer:
        raise ValueError('projection requires on-device ZeRO-3 gradient partitions')
    if not engine.is_gradient_accumulation_boundary() or not replay_losses:
        raise ValueError('a complete RL update and nonempty replay are required')
    params = [p for p in engine.module.parameters() if p.requires_grad]
    compact = dist.get_world_size() == 1

    def gradient(p):
        if compact:
            # Preserve the native gradient buffer dtype before the public API widens it.
            native, _ = optimizer._get_fp32_grad_state_partition(p, release_swap_buffers=False)
            return native
        return safe_get_local_grad(p)

    new = [gradient(p).detach().clone() for p in params]
    old_micro_step = optimizer.micro_step_id
    optimizer.zero_grad(set_to_none=True)
    for p, grad in zip(params, new, strict=True):
        if compact:
            gradient(p).zero_()
        else:
            safe_set_local_grad(p, torch.zeros_like(grad))
    for loss in replay_losses:
        engine.backward(loss() / len(replay_losses), scale_wrt_gas=False)
    optimizer.micro_step_id = old_micro_step

    stats = torch.zeros(3, device=engine.device, dtype=torch.float64)
    for p, grad in zip(params, new, strict=True):
        memory = gradient(p).flatten()
        grad = grad.flatten()
        chunk = 4_194_304 if compact else grad.numel()
        for g, m in zip(grad.split(chunk), memory.split(chunk), strict=True):
            g, m = g.float(), m.float()
            stats[0] += torch.dot(g, m).double()
            stats[1] += torch.dot(m, m).double()
            stats[2] += torch.dot(g, g).double()
    dist.all_reduce(stats, group=optimizer.dp_process_group)
    dot, old_sq, new_sq = stats.tolist()
    if not bool(torch.isfinite(stats).all()):
        raise FloatingPointError('nonfinite sharded gradient statistics')
    correction = -dot / old_sq if project and old_sq > 0 and dot < 0 else 0.0
    for p, grad in zip(params, new, strict=True):
        memory = gradient(p)
        if compact:
            for g, m in zip(grad.split(4_194_304), memory.split(4_194_304), strict=True):
                m.copy_(g.float().add_(m.float(), alpha=coefficient + correction))
        else:
            safe_set_local_grad(p, grad.add_(memory, alpha=coefficient + correction))
    return {'dot': dot, 'new_norm': new_sq ** 0.5, 'memory_norm': old_sq ** 0.5,
            'projection_triggered': bool(correction), 'coefficient': coefficient}

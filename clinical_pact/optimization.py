"""PACT auxiliary objective added once to an externally audited GSPO objective."""
import torch


def optimize_single_step(model, optimizer, gspo_loss, replay_loss):
    """One-step reference implementation; not a DDP/ZeRO training adapter."""
    if gspo_loss.ndim or replay_loss.ndim:
        raise ValueError('losses must be scalar tensors')
    if not bool(torch.isfinite(gspo_loss)) or not bool(torch.isfinite(replay_loss)):
        raise ValueError('loss must be finite')
    optimizer.zero_grad(set_to_none=True)
    (gspo_loss + replay_loss).backward()
    for p in model.parameters():
        if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
            optimizer.zero_grad(set_to_none=True)
            raise ValueError('nonfinite gradients; optimizer step aborted')
    optimizer.step()

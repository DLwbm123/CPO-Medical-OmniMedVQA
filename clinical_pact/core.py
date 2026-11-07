"""PACT-Med experimental loss/controller primitives.

CPU-tested building blocks, NOT an integrated Qwen/TRL trainer.
Margins apply to one fixed canonical prompt and an identical greedy decoder.
All tensor inputs named response_* must already be causally aligned, and masks
must include the actual EOS decision and exclude prompt/padding positions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Mapping, Sequence
import math
import torch
from torch import Tensor


def token_margins(logits: Tensor, target_ids: Tensor) -> Tensor:
    """Return z(target) - max_{v != target} z(v), in float32, shape [B,L].

    Two highest logits suffice to find the best non-target competitor. A fixed
    token-identity table, prompt, processor, decoder and special-token policy
    are prerequisites; do not silently change the candidate vocabulary.
    """
    if logits.ndim != 3 or logits.shape[-1] < 2:
        raise ValueError("logits must have shape [B,L,V], V >= 2")
    if target_ids.shape != logits.shape[:2] or target_ids.dtype != torch.long:
        raise ValueError("target_ids must be int64 [B,L]")
    if target_ids.device != logits.device:
        raise ValueError("target_ids and logits must share a device")
    if bool(((target_ids < 0) | (target_ids >= logits.shape[-1])).any()):
        raise ValueError("target_ids out of vocabulary range")
    z = logits.float()
    if bool(torch.isnan(z).any()) or bool(torch.isposinf(z).any()):
        raise ValueError("NaN/+inf logits are not supported")
    values, indices = z.topk(2, dim=-1)
    chosen = z.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    rival = torch.where(indices[..., 0] == target_ids, values[..., 1], values[..., 0])
    return chosen - rival


def _check_mask(mask: Tensor, shape: torch.Size) -> None:
    if mask.shape != shape or mask.dtype != torch.bool:
        raise ValueError("response_mask must be boolean [B,L]")
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("Every example needs at least one valid response token")


@torch.no_grad()
def make_target_margins(
    ref_logits: Tensor,
    target_ids: Tensor,
    response_mask: Tensor,
    *,
    retention: float = 0.5,
    reference_floor: float = 0.05,
    reference_cap: float = 2.0,
) -> Tensor:
    """Create immutable targets tau = retention * min(reference_gap, cap).

    Caller separately verifies: greedy sequence was correct under frozen scorer,
    strictly formatted, not truncated, and aligned with teacher-forced logits.
    Fail instead of forcing a non-greedy/near-tie trajectory into the memory.
    """
    if not 0 < retention <= 1 or not 0 < reference_floor <= reference_cap:
        raise ValueError("Invalid margin configuration")
    gaps = token_margins(ref_logits, target_ids)
    _check_mask(response_mask, gaps.shape)
    valid = gaps[response_mask]
    if not bool(torch.isfinite(valid).all()):
        raise ValueError("Nonfinite reference margin")
    if bool((valid < reference_floor).any()):
        raise ValueError("Reference trajectory is not eligible: near tie/non-greedy token")
    targets = retention * gaps.clamp(max=reference_cap)
    return torch.where(response_mask, targets, torch.zeros_like(targets)).detach()


def decision_deficits(
    logits: Tensor,
    target_ids: Tensor,
    response_mask: Tensor,
    target_margins: Tensor,
) -> Tensor:
    """Per-example D_j = max_l relu(1 - current_gap_l / tau_l), shape [B].

    This is a max over response positions, not a token average: one decision
    changing at a critical prefix must not be diluted by easy syntax tokens.
    Never clip D_j at 1: that would remove useful recovery gradients.
    """
    gaps = token_margins(logits, target_ids)
    _check_mask(response_mask, gaps.shape)
    if target_margins.shape != gaps.shape:
        raise ValueError("target_margins shape mismatch")
    if target_margins.device != logits.device or response_mask.device != logits.device:
        raise ValueError("All auxiliary-loss tensors must share a device")
    tau = target_margins.detach().float()
    if not bool(torch.isfinite(tau[response_mask]).all()) or bool((tau[response_mask] <= 0).any()):
        raise ValueError("Every valid target margin must be finite and positive")
    if not bool(torch.isfinite(gaps[response_mask]).all()):
        raise ValueError("Nonfinite current margin")
    safe_tau = torch.where(response_mask, tau, torch.ones_like(tau))
    safe_gap = torch.where(response_mask, gaps, torch.zeros_like(gaps))
    deficits = torch.relu(1.0 - safe_gap / safe_tau)
    deficits = torch.where(response_mask, deficits, torch.zeros_like(deficits))
    return deficits.amax(dim=-1)


def question_group_advantages(
    rewards: Tensor,
    question_ids: Sequence[Hashable],
    *,
    expected_group_size: int = 8,
    eps: float = 1e-4,
) -> Tensor:
    """Compute stopped GRPO-style group advantages across ALL prompt variants.

    Call after all_gather of rewards and stable question-group IDs. Each ID must
    identify one occurrence of a question, not merely its dataset row ID: repeated
    occurrences in different rollout batches are different groups.
    """
    if rewards.ndim != 1 or len(question_ids) != rewards.numel():
        raise ValueError("Expected 1-D rewards and one group ID per rollout")
    if expected_group_size < 2 or eps <= 0 or not bool(torch.isfinite(rewards).all()):
        raise ValueError("Invalid rewards/group configuration")
    groups: dict[Hashable, list[int]] = {}
    for i, key in enumerate(question_ids):
        groups.setdefault(key, []).append(i)
    result = torch.empty_like(rewards, dtype=torch.float32)
    for idx_list in groups.values():
        if len(idx_list) != expected_group_size:
            raise ValueError(f"Incomplete rollout group: {len(idx_list)} != {expected_group_size}")
        idx = torch.tensor(idx_list, device=rewards.device, dtype=torch.long)
        r = rewards[idx].float()
        result[idx] = (r - r.mean()) / (r.std(unbiased=True) + eps)
    return result.detach()


@dataclass
class TaskState:
    weight: float = 0.0
    ema_deficit: float | None = None
    observations: int = 0


@dataclass
class TaskDualController:
    """Controller consumes UNIFORMLY sampled, globally averaged audit estimates.

    Not a convergence guarantee. A cap may leave constraints violated; log both
    weights and measured deficits. This class has no distributed collectives:
    rank 0 updates after proper reduction and then broadcasts its state.
    """
    budget: float = 0.05
    learning_rate: float = 0.5
    weight_cap: float = 5.0
    ema_decay: float = 0.8
    exploration: float = 0.25
    risk_prior: float = 0.01
    states: dict[str, TaskState] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.budget < 0 or self.learning_rate <= 0 or self.weight_cap <= 0:
            raise ValueError("Invalid dual-controller scale")
        if not 0 <= self.ema_decay < 1 or not 0 < self.exploration <= 1 or self.risk_prior <= 0:
            raise ValueError("Invalid smoothing/exploration configuration")

    def add_task(self, task_id: str) -> None:
        if task_id in self.states:
            raise ValueError("Task already registered; do not reset an old reference")
        self.states[task_id] = TaskState()

    def observe_uniform_estimates(self, estimates: Mapping[str, float]) -> None:
        """Update only tasks actually audited. Missing observation != zero drift."""
        for task_id, raw in estimates.items():
            if task_id not in self.states:
                raise KeyError(task_id)
            value = float(raw)
            if not math.isfinite(value) or value < 0:
                raise ValueError("Deficits must be finite, nonnegative")
            s = self.states[task_id]
            s.ema_deficit = value if s.ema_deficit is None else (
                self.ema_decay * s.ema_deficit + (1 - self.ema_decay) * value
            )
            s.weight = min(self.weight_cap, max(
                0.0, s.weight + self.learning_rate * (s.ema_deficit - self.budget)
            ))
            s.observations += 1

    def sampling_distribution(self) -> tuple[list[str], Tensor]:
        ids = sorted(self.states)
        if not ids:
            return [], torch.empty(0, dtype=torch.float64)
        risks = torch.tensor([
            self.risk_prior + max((self.states[k].ema_deficit or 0.0) - self.budget, 0.0)
            for k in ids
        ], dtype=torch.float64)
        k = len(ids)
        probs = self.exploration / k + (1 - self.exploration) * risks / risks.sum()
        return ids, probs

    def corrected_replay_loss(self, task_id: str, mean_deficit: Tensor, draw_probability: float) -> Tensor:
        """Unbiased single-task estimate of (1/K) sum_k lambda_k D_k.

        Probability and lambda must be frozen before drawing the replay task.
        Do not update the controller between sampling and this call. Within a
        selected task, examples must be uniform; otherwise another correction
        is required. No importance clipping or self-normalization is performed.
        """
        if task_id not in self.states or mean_deficit.ndim != 0:
            raise ValueError("Unknown task or non-scalar replay estimate")
        if not 0 < draw_probability <= 1:
            raise ValueError("Invalid sampling probability")
        factor = self.states[task_id].weight / (len(self.states) * draw_probability)
        return mean_deficit * factor

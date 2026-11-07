"""Restricted TRL replay integration; distributed/full PACT remains gated."""
from trl import GRPOTrainer
from .decision_loss import replay_deficit


class SingleProcessReplayTrainer(GRPOTrainer):
    """TRL GSPO with one unbiased task-sampled PACT auxiliary loss per step.

    replay_provider.draw() must return (task, probability, model_inputs,
    response_start, eos_id, tau); draws must be uniform within the chosen task.
    Controller weights/probabilities must stay fixed through an optimizer step.
    This deliberately rejects DDP, accumulation, multi-iteration rollout reuse,
    and strategy cards until those implementations are validated separately.
    """
    def __init__(self, *args, controller, replay_provider, **kwargs):
        self.pact_controller = controller
        self.pact_replay_provider = replay_provider
        super().__init__(*args, **kwargs)
        if self.accelerator.num_processes != 1 or self.args.gradient_accumulation_steps != 1:
            raise ValueError('PACT replay adapter only supports one process and no gradient accumulation')
        if self.num_iterations != 1 or self.args.steps_per_generation != 1:
            raise ValueError('PACT replay adapter does not support reused generation batches')
        if self.importance_sampling_level != 'sequence':
            raise ValueError('PACT requires sequence-level GSPO importance sampling')
        if self.use_liger_kernel:
            raise ValueError('untested Liger loss is not supported')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError('TRL GRPO does not return outputs')
        loss = super().compute_loss(model, inputs, return_outputs=False, num_items_in_batch=num_items_in_batch)
        if not self.pact_controller.states:
            return loss
        task, probability, replay_inputs, response_start, eos_id, tau = self.pact_replay_provider.draw()
        deficit = replay_deficit(model, replay_inputs, response_start, eos_id, tau)
        return loss + self.pact_controller.corrected_replay_loss(task, deficit, probability)

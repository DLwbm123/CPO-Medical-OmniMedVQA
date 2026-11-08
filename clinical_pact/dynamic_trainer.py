"""Dynamic-card TRL integration at consumed-rollout optimizer boundaries.

The full PACT replay/ZeRO recovery runner is still gated. This class handles
card refresh only; the caller supplies actual training-audit and frozen-local
proposer callbacks, tracks costs and persists the cycle with all Trainer state.
"""
from trl.models.utils import unwrap_model_for_generation
from .fast_rollout import FirstStageRolloutTrainer
from .strategy_sync import synchronized_refresh


class DynamicStrategyTrainer(FirstStageRolloutTrainer):
    def __init__(self, *args, strategy_cycle, refresh_factory, **kwargs):
        self.strategy_cycle = strategy_cycle
        self.refresh_factory = refresh_factory
        super().__init__(*args, card_a=strategy_cycle.cards[0], card_b=strategy_cycle.cards[1], **kwargs)

    def _generate_and_score_completions(self, inputs):
        update = self.state.global_step
        cycle = self.strategy_cycle
        if update > cycle.last_refresh and update % cycle.period == 0:
            # num_iterations=1 and steps_per_generation=1 are enforced by the
            # base adapter. This boundary follows a completed optimizer step;
            # all old prompt-conditioned trajectories have been consumed.
            with unwrap_model_for_generation(self.model_wrapped, self.accelerator,
                                             gather_deepspeed3_params=True) as policy:
                was_training = policy.training
                policy.eval()
                try:
                    new_cycle = synchronized_refresh(
                        cycle, update,
                        callback_factory=lambda: self.refresh_factory(policy, update))
                finally:
                    policy.train(was_training)
            self.strategy_cycle = new_cycle
            self.pact_cards = new_cycle.cards
            self._buffered_inputs = None
        return super()._generate_and_score_completions(inputs)

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        if resume_from_checkpoint:
            raise ValueError('complete PACT cycle/replay/ZeRO resume has not been integrated')
        return super().train(resume_from_checkpoint=resume_from_checkpoint, *args, **kwargs)

"""PACT first-stage cross-card TRL rollout preparation.

This is an adapter primitive, not an end-to-end experiment runner. It requires
four ranks, two copies of the same question per rank, and no reused generation.
"""
from copy import deepcopy
import torch.distributed as dist
from trl import GRPOTrainer


def assign_rank_card(inputs, rank, card_a, card_b):
    if len(inputs) != 2 or rank not in range(4):
        raise ValueError('exactly two local trajectories and four ranks required')
    ids = [x.get('pact_question_id') for x in inputs]
    if not ids[0] or ids[0] != ids[1]:
        raise ValueError('both local trajectories must be the same question')
    strategy = ('', '', card_a, card_b)[rank]
    result = [deepcopy(item) for item in inputs]
    for item in result:
        messages = item.get('prompt')
        if not isinstance(messages, list) or not messages or messages[-1].get('role') != 'user':
            raise ValueError('expected conversational user prompt')
        if not isinstance(messages[-1].get('content'), str):
            raise ValueError('expected text user content before image insertion')
        if strategy:
            messages[-1]['content'] += '\nAnswering strategy:\n' + strategy
    return result, ids[0]


class FirstStageRolloutTrainer(GRPOTrainer):
    """Gated first-stage TRL rollout adapter, not the complete PACT trainer."""
    def __init__(self, *args, card_a, card_b, **kwargs):
        self.pact_cards = (card_a, card_b)
        super().__init__(*args, **kwargs)
        if self.accelerator.num_processes != 4 or self.args.per_device_train_batch_size != 2:
            raise ValueError('requires 4 ranks and 2 trajectories per rank')
        if self.num_generations != 8 or self.num_iterations != 1 or self.args.steps_per_generation != 1:
            raise ValueError('requires 8 generations and no rollout reuse')
        if self.importance_sampling_level != 'sequence' or self.use_vllm:
            raise ValueError('requires sequence GSPO and non-vLLM generation')

    def _generate_and_score_completions(self, inputs):
        rank = dist.get_rank()
        local_inputs, local_id = assign_rank_card(inputs, rank, *self.pact_cards)
        ids = [None] * dist.get_world_size()
        dist.all_gather_object(ids, local_id)
        if len(ids) != 4 or len(set(ids)) != 1:
            raise ValueError('the eight trajectories must share one question occurrence')
        return super()._generate_and_score_completions(local_inputs)

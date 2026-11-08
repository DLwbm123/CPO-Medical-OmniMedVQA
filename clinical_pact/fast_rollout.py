"""PACT first-stage cross-card TRL rollout preparation.

This is an adapter primitive, not an end-to-end experiment runner. It requires
four ranks, two copies of the same question per rank, and no reused generation.
"""
from copy import deepcopy
import torch.distributed as dist
from trl import GRPOTrainer


def assign_rank_card(inputs, rank, card_a, card_b, world_size=4):
    if world_size not in (2,4) or len(inputs) != 8//world_size or rank not in range(world_size):
        raise ValueError('requires 2 or 4 ranks dividing the same eight trajectories')
    ids = [x.get('pact_question_id') for x in inputs]
    if not ids[0] or any(value != ids[0] for value in ids):
        raise ValueError('both local trajectories must be the same question')
    strategies = ('', '', '', '', card_a, card_a, card_b, card_b)
    result = [deepcopy(item) for item in inputs]
    for local_index, item in enumerate(result):
        strategy = strategies[rank*(8//world_size)+local_index]
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
        if self.accelerator.num_processes not in (2,4) or self.args.per_device_train_batch_size != 8//self.accelerator.num_processes:
            raise ValueError('requires local trajectory count 8/world_size on 2 or 4 ranks')
        if self.num_generations != 8 or self.num_iterations != 1 or self.args.steps_per_generation != 1:
            raise ValueError('requires 8 generations and no rollout reuse')
        if self.importance_sampling_level != 'sequence' or self.use_vllm:
            raise ValueError('requires sequence GSPO and non-vLLM generation')

    def _generate_and_score_completions(self, inputs):
        rank = dist.get_rank()
        local_inputs, local_id = assign_rank_card(inputs, rank, *self.pact_cards, world_size=dist.get_world_size())
        ids = [None] * dist.get_world_size()
        dist.all_gather_object(ids, local_id)
        if len(ids) not in (2,4) or len(set(ids)) != 1:
            raise ValueError('the eight trajectories must share one question occurrence')
        return super()._generate_and_score_completions(local_inputs)

    def _generate_single_turn(self, prompts):
        if self.accelerator.num_processes != 2:
            return super()._generate_single_turn(prompts)
        if len(prompts)!=4 or self.tools:
            raise ValueError('two-rank sequential generation requires four prompts and no tools')
        prompt_ids,completion_ids=[],[]
        # Each rank enters exactly four collective gather/generate contexts.
        # Batch one avoids mixed-card left-padding cached-position errors.
        for prompt in prompts:
            p,c,logprobs,extra=super()._generate_single_turn([prompt])
            if logprobs is not None or extra:
                raise ValueError('unexpected custom rollout output')
            prompt_ids.extend(p);completion_ids.extend(c)
        return prompt_ids,completion_ids,None,{}

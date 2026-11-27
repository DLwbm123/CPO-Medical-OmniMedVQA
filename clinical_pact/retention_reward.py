"""Detached, verifier-gated trajectory shaping for the fixed-prompt v3 trainer."""
import torch
from torch.nn.utils.rnn import pad_sequence


@torch.no_grad()
def sampled_log_ratio(actor, reference, mask):
    drift = (actor.float() - reference.float()).masked_fill(~mask, 0).sum(dim=-1)
    if not torch.isfinite(drift).all():
        raise FloatingPointError('nonfinite sampled policy log-ratio')
    return drift


@torch.no_grad()
def shape_rewards(raw, drift, alpha):
    if alpha == 0:
        return raw
    return raw * (1 - alpha * drift.clamp(max=2))


def load_reference(trainer, checkpoint):
    from trl.models import prepare_deepspeed

    model = type(trainer.model).from_pretrained(
        checkpoint, dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True,
    )
    model.requires_grad_(False)
    return prepare_deepspeed(model, trainer.accelerator)


def rollout_prompt_features(trainer, prompts):
    return trainer.processing_class.apply_chat_template(
        conversation=prompts, tools=trainer.tools, chat_template=trainer.chat_template,
        add_generation_prompt=True, tokenize=True, padding=True, padding_side='left',
        return_tensors='pt', return_dict=True, **trainer.chat_template_kwargs,
    )


def unpadded_equal_prefixes(features):
    ids, mask = features['input_ids'], features['attention_mask']
    return bool(mask.bool().all()) and bool(ids.eq(ids[:1]).all())


@torch.no_grad()
def score_rollouts(trainer, prompts, prompt_ids, outputs):
    if len(prompt_ids) != len(outputs) or any(ids != prompt_ids[0] for ids in prompt_ids):
        raise ValueError('R1 requires the same unpadded prompt within the local rollout group')
    device = trainer.accelerator.device
    prefix = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    completion = [torch.tensor(ids, dtype=torch.long, device=device) for ids in outputs]
    suffix = pad_sequence(completion, batch_first=True, padding_value=trainer.pad_token_id)
    if not suffix.size(1):
        return torch.zeros(len(outputs), 0, device=device), torch.zeros(len(outputs), 0, device=device), suffix.bool()
    lengths = torch.tensor([len(ids) for ids in outputs], device=device)
    mask = torch.arange(suffix.size(1), device=device)[None, :] < lengths[:, None]
    ids = torch.cat((prefix, suffix), dim=1)
    attention = torch.cat((torch.ones_like(prefix), mask.long()), dim=1)
    num_images = [sum(part['type'] == 'image' for msg in prompt for part in msg['content']) for prompt in prompts]
    features = rollout_prompt_features(trainer, prompts)
    if not unpadded_equal_prefixes(features) or not torch.equal(features['input_ids'].to(device), prefix):
        raise RuntimeError('R1 scoring prompt differs from the actual rollout tokens')
    kwargs = {k: v.to(device) for k, v in features.items() if k not in ('input_ids', 'attention_mask')}
    if 'token_type_ids' in kwargs:
        types = kwargs['token_type_ids']
        kwargs['token_type_ids'] = torch.cat((types, types.new_zeros(suffix.shape)), dim=1)
    args = dict(batch_size=len(outputs), num_images=num_images, **kwargs)
    model = trainer.model_wrapped
    was_training = model.training
    model.eval()
    try:
        actor, _ = trainer._get_per_token_logps_and_entropies(model, ids, attention, suffix.size(1), **args)
        ref, _ = trainer._get_per_token_logps_and_entropies(trainer.retention_reference, ids, attention, suffix.size(1), **args)
    finally:
        model.train(was_training)
    return actor, ref, mask

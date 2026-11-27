"""Identical unpadded prompts are safe to pass to joint generation."""


def rollout_prompt_features(trainer, prompts):
    return trainer.processing_class.apply_chat_template(
        conversation=prompts, tools=trainer.tools, chat_template=trainer.chat_template,
        add_generation_prompt=True, tokenize=True, padding=True, padding_side='left',
        return_tensors='pt', return_dict=True, **trainer.chat_template_kwargs,
    )


def unpadded_equal_prefixes(features):
    ids, mask = features['input_ids'], features['attention_mask']
    return bool(mask.bool().all()) and bool(ids.eq(ids[:1]).all())

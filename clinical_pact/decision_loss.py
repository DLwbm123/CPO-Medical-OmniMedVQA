"""Teacher-forced PACT objective over an already-tokenized batch of size one."""
import torch
from .core import make_target_margins, decision_deficits


def aligned_response_logits(model, model_inputs, response_start):
    """Predict response tokens from preceding causal positions, including EOS.

    `model_inputs` must be the exact multimodal prompt concatenated with the
    reference response, with no right/left padding. `response_start` is the
    first response token's position in input_ids. The caller must verify the
    generated and teacher-forced image/prompt processing are identical.
    """
    ids = model_inputs['input_ids']
    if ids.ndim != 2 or ids.shape[0] != 1 or not 0 < response_start < ids.shape[1]:
        raise ValueError('expected a single nonempty prompt and response')
    attention = model_inputs.get('attention_mask')
    if attention is not None and not bool((attention == 1).all()):
        raise ValueError('padding is not supported in PACT replay')
    outputs = model(**model_inputs, use_cache=False)
    logits = outputs.logits[:, response_start - 1:-1, :]
    targets = ids[:, response_start:]
    if logits.shape[:2] != targets.shape:
        raise RuntimeError('causal response alignment failed')
    return logits, targets


def reference_targets(model, model_inputs, response_start, eos_id):
    ids = model_inputs['input_ids']
    if int(ids[0, -1]) != eos_id:
        raise ValueError('reference response must include actual EOS')
    with torch.no_grad():
        logits, targets = aligned_response_logits(model, model_inputs, response_start)
        if not torch.equal(logits.argmax(-1), targets):
            raise ValueError('greedy and teacher-forced reference paths disagree')
        mask = torch.ones_like(targets, dtype=torch.bool)
        return make_target_margins(logits, targets, mask).cpu()


def replay_deficit(model, model_inputs, response_start, eos_id, tau):
    if int(model_inputs['input_ids'][0, -1]) != eos_id:
        raise ValueError('cached response must include EOS')
    logits, targets = aligned_response_logits(model, model_inputs, response_start)
    mask = torch.ones_like(targets, dtype=torch.bool)
    return decision_deficits(logits, targets, mask, tau.to(logits.device)).mean()

"""Fixed candidate pools and answer-identity scores for the three-task study."""
import random

from .answer_decisions import LABELS, PREFIX, choice_view
from .data import load_split
from .scoring import normalize

TASKS = {'medbook': 'MedicalVQA_MedBook', 'pathvqa': 'MedicalVQA_Pathology', 'slake': 'MedicalVQA_Radiology'}


def memory_pool(root, reference, kind):
    if kind not in ('medbook', 'pathvqa'):
        raise ValueError('only the first two tasks have replay pools')
    rows = list(load_split(root / 'data/prepared' / TASKS[kind], 'train'))
    if kind == 'medbook':
        from .semantic_replay import known_views

        ids = {item['id'] for item, _ in known_views(reference)}
        pool = [item for item in rows if item['id'] in ids]
        if len(pool) != 338:
            raise ValueError('historical MedBook candidate cohort changed')
    else:
        by_label = {label: [] for label in ('yes', 'no')}
        for item in rows:
            label = normalize(item['answer'])
            if label in by_label:
                by_label[label].append(item)
        rows = [item for group in by_label.values() for item in group]
        rng = random.Random(100)
        pool = [item for group in by_label.values() for item in rng.sample(group, 169)]
        ids = {item['id'] for item in pool}
    return pool, [item for item in rows if item['id'] not in ids]


def decision_views(items, kind, rotations=4):
    if kind == 'medbook':
        return [choice_view(item, offset) for item in items for offset in range(rotations)]
    if kind == 'pathvqa':
        return [(item, ('yes', 'no').index(normalize(item['answer']))) for item in items]
    raise ValueError('unknown replay task')


def candidate_tokens(tokenizer, kind):
    labels = LABELS if kind == 'medbook' else ('yes', 'no')
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    seqs = [tokenizer.encode(PREFIX + label, add_special_tokens=False) for label in labels]
    if not prefix or any(seq[:-1] != prefix for seq in seqs):
        raise ValueError('each legal answer must continue the prefix as one token')
    indices = [seq[-1] for seq in seqs]
    if len(set(indices)) != len(labels):
        raise ValueError('semantic labels map to duplicate token IDs')
    return prefix, indices


def fp32_scores(model, processor, item, device, token_spec):
    import torch
    from torch.nn import functional as F

    from .multimodal import make_inputs

    prefix, candidates = token_spec
    inputs = make_inputs(processor, item, device=device, assistant_prefix=PREFIX)
    if inputs['input_ids'][0, -len(prefix):].tolist() != prefix:
        raise ValueError('expected the verified answer prefix at the scoring position')
    module = model.module if hasattr(model, 'module') else model
    head = module.get_output_embeddings()
    ids = torch.tensor(candidates, dtype=torch.long, device=device)

    def score_selected(layer, args, output):
        del output
        weights = layer.weight.index_select(0, ids).float()
        bias = layer.bias.index_select(0, ids).float() if layer.bias is not None else None
        with torch.autocast(device_type='cuda', enabled=False):
            return F.linear(args[0].float(), weights, bias)

    hook = head.register_forward_hook(score_selected, prepend=True)
    try:
        scores = model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1]
    finally:
        hook.remove()
    if scores.dtype != torch.float32 or scores.numel() != len(candidates):
        raise ValueError('the model did not return the selected FP32 scores')
    return scores


def margin(scores, target):
    if not 0 <= target < scores.numel():
        raise ValueError('target outside the legal answer set')
    return scores[target] - scores.masked_fill(scores.new_tensor(range(scores.numel())).eq(target), -float('inf')).max()

"""MedBook answer decisions indexed by option identity, independent of EOS."""
import re

from .scoring import normalize

LABELS = 'ABCD'
PREFIX = r'\boxed{'
_OPTION = re.compile(r'^([A-D])\. (\S.*)$')
_BOX = re.compile(r'\\boxed\s*\{([^{}]*)\}')


def choice_view(item, rotation):
    """Rotate option positions and keep the ground-truth answer identity."""
    gold = normalize(item['answer']).upper()
    lines = item['question'].splitlines()
    found = [(i, match) for i, line in enumerate(lines) if (match := _OPTION.fullmatch(line))]
    if len(found) != 4 or ''.join(match[1] for _, match in found) != LABELS or gold not in set(LABELS):
        raise ValueError('expected four ordered, labeled options and an answer letter')
    values = [match[2] for _, match in found]
    if len(set(map(normalize, values))) != 4:
        raise ValueError('semantic choice identities require distinct option text')
    rotation %= 4
    values = values[rotation:] + values[:rotation]
    for label, (index, _), answer in zip(LABELS, found, values, strict=True):
        lines[index] = f'{label}. {answer}'
    target = (LABELS.index(gold) - rotation) % 4
    return {**item, 'question': '\n'.join(lines), 'answer': LABELS[target]}, target


def choice_token_ids(tokenizer):
    prefix_ids = tokenizer.encode(PREFIX, add_special_tokens=False)
    completions = [tokenizer.encode(PREFIX + label, add_special_tokens=False) for label in LABELS]
    if any(ids[:-1] != prefix_ids for ids in completions):
        raise ValueError('answer letters must extend the forced prefix by exactly one token')
    ids = tuple(tokens[-1] for tokens in completions)
    if len(set(ids)) != 4:
        raise ValueError('answer letters map to distinct token IDs')
    return ids


def choice_logits(model, processor, item, device, candidate_ids):
    from .multimodal import make_inputs

    inputs = make_inputs(processor, item, device=device, assistant_prefix=PREFIX)
    prefix_ids = processor.tokenizer.encode(PREFIX, add_special_tokens=False)
    if inputs['input_ids'][0, -len(prefix_ids):].tolist() != prefix_ids:
        raise ValueError('the input template changed the tokenization of the forced answer prefix')
    return model(**inputs, use_cache=False).logits[0, -1, list(candidate_ids)].float()


def answer_candidate(text):
    matches = _BOX.findall(text)
    return normalize(matches[-1] if matches else text)

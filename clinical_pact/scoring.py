"""Frozen normalized exact-match scorer, equivalent to the clinical scorer."""
import re
import unicodedata


def normalize(value):
    value = unicodedata.normalize('NFKC', str(value)).casefold().strip()
    return ' '.join(re.sub(r'[^\w\s]', ' ', value).split())


def score_response(text, answer):
    boxes = re.findall(r'\\boxed\s*\{([^{}]*)\}', text)
    candidate = boxes[-1] if boxes else text
    gold = normalize(answer)
    return float(bool(gold) and normalize(candidate) == gold)

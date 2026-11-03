"""Closed-answer rewards: the only accepted final answer is in \\boxed{}."""
import re


def extract_boxed(text):
    matches = re.findall(r"\\boxed\s*\{([^{}]*)\}", text)
    return matches[-1].strip() if matches else ""


def canonical_answer(value):
    value = str(value).strip().casefold()
    return value if re.fullmatch(r"[a-z]|yes|no", value) else ""


def score_response(text, answer):
    boxed = extract_boxed(text)
    if boxed:
        prediction = canonical_answer(boxed)
    else:
        # Accept a bare option or a leading option followed by punctuation and short option text.
        match = re.fullmatch(r"\s*([A-Da-d])(?:\s*[.):：](?:\s+[^\n]{1,150})?)?\s*", text)
        prediction = canonical_answer(match.group(1)) if match else ""
    target = canonical_answer(answer)
    return float(bool(target and prediction and target == prediction))


def accuracy_reward(completions, assistant, **kwargs):
    if len(completions) != len(assistant):
        raise ValueError("Completion and answer counts differ")
    return [score_response(c[0]["content"], a) for c, a in zip(completions, assistant)]

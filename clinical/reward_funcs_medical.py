import re, unicodedata

def normalize(value):
    value=unicodedata.normalize("NFKC",str(value)).casefold().strip()
    value=re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())

def score_response(text, answer):
    boxes=re.findall(r"\\boxed\s*\{([^{}]*)\}",text)
    candidate=boxes[-1] if boxes else text
    return float(bool(normalize(answer)) and normalize(candidate)==normalize(answer))

def accuracy_reward(completions,assistant,**kwargs):
    if len(completions)!=len(assistant):raise ValueError("length mismatch")
    return [score_response(c[0]["content"],a) for c,a in zip(completions,assistant)]

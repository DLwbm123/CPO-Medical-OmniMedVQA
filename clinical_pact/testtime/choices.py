"""Variable-option rotations, token identity, and answer extraction."""
from dataclasses import replace
import re

from ..scoring import normalize
from .tt_data import OPTION


BOX = re.compile(r'\\boxed\s*\{([^{}]*)\}')


def choice_view_k(item, rotation):
    k = len(item.letters)
    offset = rotation % k
    options = item.options[offset:]+item.options[:offset]
    prefix = item.question[:OPTION.search(item.question).start()]
    next_question = prefix+'\n'.join(f'{letter}. {choice}' for letter,choice in zip(item.letters,options,strict=True))
    mapping = tuple(item.letters[(i-offset)%k] for i in range(k))
    return replace(item,question=next_question,options=options),mapping


def letter_tokens(tokenizer, letters, prefix=r'\boxed{'):
    tokens = tokenizer.encode(prefix,add_special_tokens=False)
    replies = [tokenizer.encode(prefix+letter,add_special_tokens=False) for letter in letters]
    if not tokens or any(r[:-1]!=tokens for r in replies):
        raise ValueError('each legal option must add exactly one token to the prefix')
    ids = [r[-1] for r in replies]
    if len(set(ids))!=len(letters):
        raise ValueError('option token IDs must be unique')
    return tokens,ids


def bare_letter_tokens(tokenizer, letters):
    encoded = [tokenizer.encode(letter,add_special_tokens=False) for letter in letters]
    if any(len(ids)!=1 or normalize(tokenizer.decode(ids,skip_special_tokens=True))!=letter.lower()
           for ids,letter in zip(encoded,letters,strict=True)):
        raise ValueError('bare options must round-trip as single tokens')
    ids = [x[0] for x in encoded]
    if len(set(ids))!=len(letters):
        raise ValueError('bare option token IDs must be unique')
    return ids


def extract_letter(text, letters):
    matches = BOX.findall(text)
    answer = normalize(matches[-1] if matches else text).upper()
    return answer if len(answer)==1 and answer in letters else None

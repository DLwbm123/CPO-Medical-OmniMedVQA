"""Stage B vote aggregation and training-trajectory selection without labels."""
from collections import Counter
from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class VoteDecision:
    counts: dict[str, int]
    pseudo_label: str | None
    peak: int
    entropy: float
    in_band: bool

    def rewards(self, answers):
        return [float(self.pseudo_label is not None and letter == self.pseudo_label)
                for letter in answers]


def gather_votes(local):
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tuple(local)
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered,tuple(local))
    return tuple(letter for group in gathered for letter in group)


def vote_decision(answers, letters, band=(10,14)):
    if len(answers) != 16 or any(letter is not None and letter not in letters for letter in answers):
        raise ValueError('expected sixteen legal letters or invalid votes')
    counts = Counter(letter for letter in answers if letter is not None)
    peak = max(counts.values(),default=0)
    leading = [letter for letter,count in counts.items() if count == peak]
    pseudo = leading[0] if len(leading) == 1 else None
    all_counts = (*counts.values(), answers.count(None))
    entropy = -sum(c/16*math.log(c/16) for c in all_counts if c)
    return VoteDecision({letter:counts[letter] for letter in letters},
                        pseudo,peak,entropy,pseudo is not None and band[0] <= peak <= band[1])


def training_indices(seed, step, question_id):
    return tuple(sorted(random.Random(f'{seed}:{step}:{question_id}').sample(range(16),8)))

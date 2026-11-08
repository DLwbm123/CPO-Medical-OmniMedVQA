"""Training-only strategy refresh with a bounded procedural proposal grammar.

This is a deliberate safety-constrained variant of free-text proposal: the
local proposer chooses procedure IDs and cannot add medical facts or answers.
The bounded development driver integrates refresh into distributed training;
complete mid-stage Trainer/checkpoint recovery remains unimplemented.
"""
from dataclasses import dataclass, field
import copy
from .prompt_population import Candidate, choose_cards

PROCEDURES = (
    'Identify exactly what the question asks.',
    'Check whether the question contains negation.',
    'Check the requested location and level of detail.',
    'Compare the relevant differences between explicit answer options.',
    'Compare the requested finding with the image before choosing an answer.',
    'Preserve the requested answer type and return only the final answer.',
)


def proposal_texts(proposals):
    if not isinstance(proposals, (list, tuple)) or len(proposals) != 2:
        raise ValueError('proposer must return exactly two procedure-ID sequences')
    texts = []
    for sequence in proposals:
        if not isinstance(sequence, (list, tuple)) or not 1 <= len(sequence) <= 4:
            raise ValueError('each proposal needs one to four procedure IDs')
        if any(type(index) is not int or index not in range(len(PROCEDURES)) for index in sequence):
            raise ValueError('unknown procedure or free text in proposal')
        if len(set(sequence)) != len(sequence):
            raise ValueError('duplicate procedure within a proposal')
        texts.append(' '.join(PROCEDURES[index] for index in sequence))
    return texts


@dataclass
class StrategyCycle:
    cards: tuple[str, str]
    period: int = 100
    last_refresh: int = 0
    cycle: int = 0
    rollouts_in_flight: bool = False
    history: list = field(default_factory=list)

    def __post_init__(self):
        if self.period <= 0 or len(self.cards) != 2 or not all(isinstance(card, str) for card in self.cards):
            raise ValueError('invalid card cycle')

    def begin_rollouts(self):
        if self.rollouts_in_flight:
            raise ValueError('cannot start another rollout batch before consuming it')
        self.rollouts_in_flight = True
        return self.cycle, self.cards

    def finish_rollouts(self, cycle):
        if not self.rollouts_in_flight or cycle != self.cycle:
            raise ValueError('stale or missing rollout cycle')
        self.rollouts_in_flight = False

    def refresh(self, update, *, split, audit_ids, evaluate, propose, count_tokens, invalid_proposal='raise'):
        """Callbacks use current frozen policy and train-only fixed audit items.

        evaluate(text, audit_ids) returns one binary verifier result per audit
        item. propose receives aggregate scores only and returns procedure IDs.
        count_tokens uses the actual model tokenizer. No sample text is passed
        to the proposer. All GPU/teacher calls must be charged by the caller.
        """
        if split != 'train':
            raise ValueError('strategy selection is training-only')
        if self.rollouts_in_flight:
            raise ValueError('cannot refresh while old rollout/logprob data are live')
        if type(update) is not int or update < self.last_refresh:
            raise ValueError('optimizer update moved backwards')
        if update == self.last_refresh or update % self.period:
            return False
        if update != self.last_refresh + self.period:
            raise ValueError('a scheduled refresh was skipped')
        if not audit_ids or len(set(audit_ids)) != len(audit_ids):
            raise ValueError('audit IDs must be nonempty and unique')
        audit_ids = tuple(audit_ids)

        def scored(text):
            scores = tuple(evaluate(text, audit_ids))
            if len(scores) != len(audit_ids):
                raise ValueError('incomplete audit scores')
            return Candidate(text, count_tokens(text), scores)

        canonical = scored('')
        current = [scored(text) for text in self.cards]
        feedback = {'n': len(audit_ids), 'canonical_correct': sum(canonical.scores),
                    'current_correct': [sum(item.scores) for item in current]}
        if invalid_proposal not in ('raise','retain'):
            raise ValueError('unknown invalid-proposal policy')
        proposal_error = None
        try:
            proposed = proposal_texts(propose(feedback, PROCEDURES))
        except ValueError as error:
            if invalid_proposal == 'raise':
                raise
            proposed = []
            proposal_error = type(error).__name__
        candidates = current + [scored(text) for text in proposed]
        selected = choose_cards(canonical, candidates)
        # Publish a new cycle only after every callback and validation succeeds.
        self.cards = selected
        self.last_refresh = update
        self.cycle += 1
        self.history.append({'update': update, 'cycle': self.cycle, 'audit_n': len(audit_ids),
                             'canonical_correct': sum(canonical.scores),
                             'candidate_correct': [sum(item.scores) for item in candidates],
                             'distinct_nonempty_cards': len({card for card in selected if card}), 'proposal_error': proposal_error})
        return True

    def checkpoint(self):
        if self.rollouts_in_flight:
            raise ValueError('checkpoint only at a drained optimizer boundary')
        return {'cards': tuple(self.cards), 'period': self.period, 'last_refresh': self.last_refresh,
                'cycle': self.cycle, 'history': copy.deepcopy(self.history)}

    @classmethod
    def restore(cls, state):
        restored = cls(tuple(state['cards']), state['period'], state['last_refresh'], state['cycle'])
        if restored.last_refresh != restored.cycle * restored.period:
            raise ValueError('inconsistent cycle and optimizer cursor')
        restored.history = copy.deepcopy(state['history'])
        return restored

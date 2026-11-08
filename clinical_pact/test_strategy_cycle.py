import unittest
from clinical_pact.strategy_cycle import StrategyCycle, proposal_texts


class StrategyCycleTests(unittest.TestCase):
    def callbacks(self):
        calls = []
        def evaluate(text, ids):
            calls.append((text, ids))
            return (1, 0, 0, 0) if not text else (1, 1, 0, 0)
        def propose(feedback, procedures):
            self.assertEqual(set(feedback), {'n', 'canonical_correct', 'current_correct'})
            self.assertEqual(feedback['n'], 4)
            return [[0, 5], [1, 2]]
        return dict(split='train', audit_ids=('a', 'b', 'c', 'd'), evaluate=evaluate,
                    propose=propose, count_tokens=lambda text: len(text.split())), calls

    def test_refresh_is_atomic_and_keeps_rollout_condition(self):
        cycle = StrategyCycle(('oldA', 'oldB'))
        kwargs, calls = self.callbacks()
        identity, cards = cycle.begin_rollouts()
        with self.assertRaises(ValueError): cycle.refresh(100, **kwargs)
        self.assertEqual(cycle.cards, cards)
        cycle.finish_rollouts(identity)
        self.assertTrue(cycle.refresh(100, **kwargs))
        self.assertEqual(len(calls), 5)
        self.assertFalse(cycle.refresh(100, **kwargs))
        restored = StrategyCycle.restore(cycle.checkpoint())
        self.assertEqual(restored.checkpoint(), cycle.checkpoint())
        with self.assertRaises(ValueError): restored.refresh(300, **kwargs)
        with self.assertRaises(ValueError): restored.finish_rollouts(identity)

    def test_proposer_cannot_introduce_free_text(self):
        for invalid in ([[0], ['answer']], [[True], [1]], [[0, 0], [1]], [[99], [1]]):
            with self.assertRaises(ValueError): proposal_texts(invalid)

    def test_test_split_rejected_and_callback_failure_does_not_update(self):
        cycle = StrategyCycle(('oldA', 'oldB'))
        kwargs, _ = self.callbacks()
        kwargs['split'] = 'test'
        with self.assertRaises(ValueError): cycle.refresh(100, **kwargs)
        kwargs['split'] = 'train'
        kwargs['propose'] = lambda *args: [[0], ['freeform']]
        before = cycle.checkpoint()
        with self.assertRaises(ValueError): cycle.refresh(100, **kwargs)
        self.assertEqual(before, cycle.checkpoint())

    def test_local_synchronized_refresh_does_not_mutate_previous_cycle(self):
        from clinical_pact.strategy_sync import synchronized_refresh
        kwargs, _ = self.callbacks()
        cycle = StrategyCycle(('oldA', 'oldB'))
        refreshed = synchronized_refresh(cycle, 100, **kwargs)
        self.assertEqual(cycle.cycle, 0)
        self.assertEqual(refreshed.cycle, 1)

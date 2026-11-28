import unittest

from .guard import Guard, total_variation
from .votes import training_indices, vote_decision


class PilotRewardTest(unittest.TestCase):
    def test_majority_gate_rewards_and_subset(self):
        answers = ['A']*10 + ['B']*4 + [None]*2
        result = vote_decision(answers,'AB')
        self.assertEqual(result.pseudo_label,'A')
        self.assertTrue(result.in_band)
        chosen = training_indices(100,1,'case-1')
        self.assertEqual(chosen,training_indices(100,1,'case-1'))
        self.assertEqual(len(set(chosen)),8)
        self.assertEqual(result.rewards([answers[i] for i in chosen]),
                         [float(i < 10) for i in chosen])
        tied = vote_decision(['A']*8+['B']*8,'AB')
        self.assertIsNone(tied.pseudo_label)
        self.assertEqual(tied.rewards(['A','B',None]),[0.,0.,0.])
        self.assertFalse(vote_decision(['B']*15+[None],'AB').in_band)

    def test_guard_window_and_brake_stop(self):
        guard = Guard({'A':50,'B':50},window=2,consecutive_limit=2)
        self.assertEqual(guard.observe_update([{'A':8,'B':8}]),(0.,False,False))
        self.assertEqual(guard.observe_update([{'A':16},{'A':16}]),(.5,True,False))
        self.assertEqual(guard.observe_update([{'A':16}]),(.5,True,False))
        self.assertEqual(guard.observe_update([{'A':16}]),(.5,True,True))
        self.assertEqual(guard.observe_update([{'A':8,'B':8}]*2),(0.,False,False))
        self.assertIsNone(total_variation({}, {'A':1}))

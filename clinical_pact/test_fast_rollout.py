import unittest
from clinical_pact.fast_rollout import assign_rank_card


class FastRolloutTests(unittest.TestCase):
    def test_rank_allocation_keeps_original_context(self):
        row = {'pact_question_id': 'q', 'prompt': [{'role': 'user', 'content': 'canonical'}]}
        inputs = [row, row]
        for rank, expected in enumerate(('', '', 'alpha', 'beta')):
            prepared, qid = assign_rank_card(inputs, rank, 'alpha', 'beta')
            self.assertEqual(qid, 'q')
            self.assertTrue(all(p['prompt'][-1]['content'] == ('canonical\nAnswering strategy:\n' + expected if expected else 'canonical') for p in prepared))
        self.assertEqual(row['prompt'][-1]['content'], 'canonical')
        with self.assertRaises(ValueError):
            assign_rank_card(inputs[:1], 0, 'alpha', 'beta')

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

    def test_two_rank_allocation_has_four_canonical_two_each_card(self):
        row = {'pact_question_id':'q','prompt':[{'role':'user','content':'canonical'}]}
        left,_ = assign_rank_card([row]*4,0,'alpha','beta',world_size=2)
        right,_ = assign_rank_card([row]*4,1,'alpha','beta',world_size=2)
        texts = [item['prompt'][-1]['content'] for item in left+right]
        self.assertEqual(texts.count('canonical'),4)
        self.assertEqual(texts.count('canonical\nAnswering strategy:\nalpha'),2)
        self.assertEqual(texts.count('canonical\nAnswering strategy:\nbeta'),2)
        self.assertEqual(row['prompt'][-1]['content'],'canonical')

import unittest
from clinical_pact.scoring import score_response
from clinical_pact.memory import DecisionMemory, DecisionRecord
from clinical_pact.grouping import Rollout, validate_groups


def record(task, i, **changes):
    fields = dict(task=task, record_id=str(i), image_ref='private-image', prompt_fingerprint='v1', response_ids=(3, 2), margins=(0.4, 0.3), eos_id=2, priority=i, stratum=str(i % 2), correctly_scored=True, complete=True, path_aligned=True)
    fields.update(changes)
    return DecisionRecord(**fields)


class Components(unittest.TestCase):
    def test_scoring(self):
        self.assertEqual(score_response('explanation \\boxed{ A }', 'a'), 1)
        self.assertEqual(score_response('yes', 'yes'), 1)
        self.assertEqual(score_response('yes indeed', 'yes'), 0)
        self.assertEqual(score_response('', ''), 0)
        self.assertEqual(score_response('\\boxed{A} then \\boxed{B}', 'B'), 1)

    def test_reference_validation_and_rebalance(self):
        with self.assertRaises(ValueError): record('a', 0, response_ids=(3,))
        with self.assertRaises(ValueError): record('a', 0, margins=(0.01, 0.3))
        memory = DecisionMemory(4)
        memory.add_task('a', [record('a', i) for i in range(4)])
        memory.add_task('b', [record('b', i) for i in range(4)])
        self.assertEqual([len(memory.tasks[x]) for x in ('a', 'b')], [2, 2])
        with self.assertRaises(ValueError): memory.add_task('a', [record('a', 10)])

    def test_group_context_and_counts(self):
        data = [Rollout(('stage', 1, 'question'), strategy, strategy, strategy, float(i % 2)) for strategy, n in [('canonical', 4), ('card_a', 2), ('card_b', 2)] for i in range(n)]
        self.assertEqual(len(validate_groups(data)), 1)
        with self.assertRaises(ValueError): validate_groups(data[:-1])
        data[0] = Rollout(('stage', 1, 'question'), 'canonical', 'one', 'other', 1.0)
        with self.assertRaises(ValueError): validate_groups(data)


if __name__ == '__main__':
    unittest.main()

class DataTests(unittest.TestCase):
    def test_missing_or_escaping_images_fail_closed(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from clinical_pact.data import load_split
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'jsons/train').mkdir(parents=True)
            (root / 'images').mkdir()
            row = {'id': '1', 'image': '../outside', 'conversations': [{'value': 'q'}, {'value': 'a'}]}
            (root / 'jsons/train/data.json').write_text(json.dumps([row]))
            with self.assertRaises(ValueError): list(load_split(root, 'train'))

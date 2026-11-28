"""Verify vote accounting and the boundary between evaluation and training."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from .choices import choice_view_k, extract_letter
from .probe import exact_votes, sampled_votes, summarize, evaluate_gate
from .tt_data import load_unlabeled, load_labeled_eval, reserve_reference_ids
from .tt_memory import audit_items


class VoteTest(unittest.TestCase):
    def test_multinomial_matches_monte_carlo(self):
        p = np.array([.42,.15,.12,.08,.23])
        exact = exact_votes(p,0,(10,14))
        rng = np.random.default_rng(100)
        for n in (8,16):
            counts = rng.multinomial(n,p,size=150_000)[:, :4]
            correct = (counts[:,0] > counts[:,1:].max(axis=1))
            self.assertAlmostEqual(exact[f'maj{n}'],correct.mean(),delta=.005)
            if n == 16:
                peaks = counts.max(axis=1)
                histogram = np.bincount(peaks,minlength=17)/len(peaks)
                np.testing.assert_allclose(exact['agreement'],histogram,atol=.005)
                self.assertAlmostEqual(exact['band_correct'],
                                       (correct & (peaks >= 10) & (peaks <= 14)).mean(),delta=.005)

    def test_invalid_ties_and_gate(self):
        all_invalid = exact_votes(np.array([0.,0.,1.]),0,(10,14))
        self.assertEqual(all_invalid['maj16'],0)
        self.assertEqual(all_invalid['agreement'][0],1)
        tied = sampled_votes([0,1]*4+[2]*8,2,0,(10,14))
        self.assertEqual(tied['maj16'],0.)
        self.assertEqual(tied['maj8'],0.)
        self.assertEqual(tied['pass16'],1.)
        accepted = sampled_votes([0]*10+[2]*6,2,0,(10,14))
        row = summarize([dict(accepted,gold='A')],(10,14))
        self.assertEqual(row['band_coverage'],1)
        self.assertEqual(row['pseudo_accuracy'],1)
        self.assertEqual(row['predicted_marginal']['invalid'],.375)
        row['passes'] = evaluate_gate(row,dict(min_gain=.02,min_coverage=.3,min_pseudo_accuracy=.7))
        self.assertTrue(json.loads(json.dumps(row))['passes'])


class DataTest(unittest.TestCase):
    def test_redacted_loader_audit_and_rotation(self):
        with TemporaryDirectory() as base:
            root = Path(base)
            (root/'images').mkdir()
            (root/'images'/'x.jpg').touch()
            folder = root/'unlabeled'/'train'
            folder.mkdir(parents=True)
            path = folder/'data.json'
            rows = [dict(id=str(i),image='x.jpg',source_dataset='synthetic',
                         question='<image>\nQuestion\nA. first\nB. second\nC. third')
                    for i in range(5)]
            path.write_text(json.dumps(rows))
            items = load_unlabeled(root,'train',100)
            reference = reserve_reference_ids(items,count=2)
            stream = load_unlabeled(root,'train',100,exclude_ids=reference)
            self.assertFalse(set(reference) & {item.id for item in stream})
            item = stream[0]
            self.assertFalse(hasattr(item,'answer'))
            self.assertEqual(audit_items(stream,[item.id],2,100),
                             audit_items(reversed(stream),[item.id],2,100))
            self.assertNotIn(item.id,{a.id for a in audit_items(stream,[item.id],2,100)})
            rotated, mapping = choice_view_k(item,4)
            self.assertEqual(rotated.options,('second','third','first'))
            self.assertEqual(mapping,('C','A','B'))
            path.write_text(json.dumps([dict(rows[0],answer='A')]))
            with self.assertRaises(ValueError):
                load_unlabeled(root,'train',100)
            with self.assertRaises(ValueError):
                load_labeled_eval(root,'train')

    def test_extraction_matches_frozen_boxed_rule(self):
        self.assertEqual(extract_letter(r'text \boxed{A} then \boxed{c}', 'ABC'),'C')
        self.assertEqual(extract_letter('b.', 'AB'),'B')
        self.assertIsNone(extract_letter(r'\boxed{C}', 'AB'))
        self.assertIsNone(extract_letter('A or B', 'AB'))


if __name__ == '__main__':
    unittest.main()

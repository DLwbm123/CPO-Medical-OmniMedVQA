from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .guard import mask_image_gate
from .replication import image_paired_interval, replication_gate, reserve_groups


class ReplicationTest(unittest.TestCase):
    def test_reserves_all_questions_for_the_same_image(self):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            (root/'a').touch()
            (root/'b').touch()
            (root/'alias').symlink_to(root/'a')
            rows = [{'id':'a','image':'a'},{'id':'alias','image':'alias'},{'id':'b','image':'b'}]
            holdout, training, held_images, train_images = reserve_groups(rows,root,1,100)
            self.assertEqual('a' in holdout,'alias' in holdout)
            self.assertFalse(holdout & {r['id'] for r in training})
            self.assertEqual((held_images,train_images),(1,1))

    def test_image_gate_removes_both_strategy_and_kl_masks(self):
        import torch

        batch = {'advantages':torch.ones(4), 'completion_mask':torch.ones((4,3))}
        mask_image_gate(batch,['a','b'],{'a'},2)
        self.assertEqual(batch['advantages'].tolist(),[0,0,1,1])
        self.assertEqual(batch['completion_mask'].sum(dim=1).tolist(),[0,0,3,3])

    def test_gate_requires_each_retention_threshold_and_uses_calibration_or(self):
        paired = image_paired_interval([1,1,1],['a','a','b'],{'bootstrap_samples':50,'bootstrap_seed':100})
        self.assertEqual((paired['ci95'],paired['bootstrap_images']),([1.,1.],2))
        def metrics(real,black,wrong,top):
            return {'Dermoscopy':{'real':{'accuracy':real,'high_confidence_wrong_rate':wrong,
                                         'selective_accuracy':{'0.25':{'accuracy':top}}},
                                  'black':{'accuracy':black}}}
        gate = {'min_decision_gain':.05,'max_image_dependence_drop':.02,
                'max_high_confidence_wrong_rate':.03,'max_top_quartile_accuracy_drop':.03,
                'max_retained_macro_drop':.02,'max_retained_task_drop':.05}
        frozen, adapted = metrics(.7,.4,.02,.94),metrics(.76,.47,.05,.92)
        self.assertTrue(replication_gate(frozen,adapted,paired,{'CT':-.02},'Dermoscopy',gate)['passed'])
        retained = {'CT':.05,'OCT':-.06}
        outcome = replication_gate(frozen,adapted,paired,retained,'Dermoscopy',gate)
        self.assertFalse(outcome['passed'])
        self.assertFalse(outcome['conditions']['retention'])

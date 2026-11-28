import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from .decision_diagnostic import choose, paired_interval, route, summarize, training_prior


class DecisionDiagnosticTest(unittest.TestCase):
    def test_ignores_illegal_tokens_and_scores_legal_confidence(self):
        import torch

        item = SimpleNamespace(letters='AB')
        model = Mock(return_value=SimpleNamespace(logits=torch.tensor([[[100.,4.,5.,0.]]])))
        with patch('clinical_pact.testtime.decision_diagnostic.input_for',
                   return_value={'input_ids':torch.tensor([[0,3]])}) as make:
            letter, confidence = choose(model,None,item,'reason',r'\boxed{',[3],[1,2],True)
        self.assertTrue(make.call_args.kwargs['black_image'])
        self.assertEqual(letter,'B')
        self.assertAlmostEqual(confidence,0.7310586)

    def test_prior_tie_breaks_without_validation_and_routes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/'MedicalVQA_Fundus'/'jsons'/'train'
            path.mkdir(parents=True)
            (path/'data.json').write_text(json.dumps([
                {'id':i,'conversations':[{'value':'Q\nA. x\nB. y'},{'value':letter}]}
                for i,letter in enumerate(('B','A'))]))
            self.assertEqual(training_prior(root,2),('A',{'A':1,'B':1,'C':0,'D':0}))
        cfg = {'bootstrap_seed':100,'bootstrap_samples':1000,
               'format_gain_cutoff':.02,'decision_gain_cutoff':.05}
        advantage = paired_interval([1]*15,cfg)
        self.assertEqual(advantage['ci95'],[1.,1.])
        self.assertEqual(route(.01,advantage,advantage,cfg),'return_to_labeled_v3_1')
        self.assertEqual(route(.05,advantage,advantage,cfg),'prospective_calibration_pilot')
        trivial = paired_interval([0]*15,cfg)
        self.assertEqual(route(.06,advantage,trivial,cfg),'inconclusive_new_modality_pilot')

    def test_confident_errors_and_selective_accuracy_have_distinct_denominators(self):
        rows = [{'letter':'A','correct':True,'confidence':.99},
                {'letter':'B','correct':False,'confidence':.95},
                {'letter':'A','correct':False,'confidence':.52}]
        result = summarize(rows,{'confidence_cutoff':.9,'selective_coverages':[.25,1.]})
        self.assertEqual(result['high_confidence_wrong_rate'],1/3)
        self.assertEqual(result['error_rate_if_high_confidence'],1/2)
        self.assertEqual(result['selective_accuracy']['0.25']['accuracy'],1)

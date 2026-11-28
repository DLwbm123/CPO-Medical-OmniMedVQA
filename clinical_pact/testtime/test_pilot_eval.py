import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from .pilot_eval import evaluation_cohorts, greedy_letter
from .probe import sample_answers


class PilotCohortTest(unittest.TestCase):
    def test_deterministic_subsets_and_full_target_validation(self):
        spec = {'seed':100, 'task':'Fundus', 'validation_counts':{'CT':5, 'Fundus':5},
                'other_validation_n':3}
        rows = [(SimpleNamespace(id=str(i)), 'A') for i in range(5)]
        with patch('clinical_pact.testtime.pilot_eval.load_labeled_eval', return_value=rows) as loader:
            first = dict(evaluation_cohorts(Path('data'), spec))
            loader.return_value = rows[::-1]
            second = dict(evaluation_cohorts(Path('data'), spec))
        self.assertEqual(first, second)
        self.assertEqual(set(first), {'CT','Fundus'})
        self.assertEqual(len(first['CT']),3)
        self.assertEqual(len(first['Fundus']),5)
        self.assertTrue(all(call.args[1]=='validation' for call in loader.call_args_list))

    def test_decode_uses_protocol_defaults_after_checkpoint_resave(self):
        import torch

        tokenizer = Mock()
        tokenizer.decode.return_value = 'A'
        tokenizer.batch_decode.return_value = ['A']*16
        tokenizer.all_special_ids = [99]
        inputs = {'input_ids': torch.tensor([[10]])}
        saved = SimpleNamespace(do_sample=True, temperature=1e-6, repetition_penalty=1.05, eos_token_id=99)
        model = Mock(generation_config=saved)
        model.generate.return_value = torch.tensor([[10,1,99]])
        self.assertEqual(greedy_letter(model,tokenizer,inputs,'AB',SimpleNamespace(do_sample=False)),'A')
        self.assertFalse(model.generate.call_args.kwargs['use_model_defaults'])
        self.assertFalse(model.generate.call_args.kwargs['generation_config'].do_sample)
        model.generate.return_value = torch.tensor([[10,1,99]]*16)
        spec = {'sampling':{'do_sample':True,'temperature':1.,'repetition_penalty':1.},
                'agreement_band_votes':[10,14]}
        sample_answers(model,tokenizer,inputs,'AB','A',spec,100)
        call = model.generate.call_args.kwargs
        self.assertFalse(call['use_model_defaults'])
        self.assertEqual(call['generation_config'].temperature,1.)
        self.assertEqual(call['generation_config'].repetition_penalty,1.)

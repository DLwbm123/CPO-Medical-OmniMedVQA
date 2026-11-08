import unittest
import torch
from unittest.mock import patch
from clinical_pact.strategy_model import LocalProcedureProposer, PolicyStrategyEvaluator
from clinical_pact.strategy_cycle import PROCEDURES


class Encoding(dict):
    def to(self, device): return self


class Tokenizer:
    eos_token_id = 2
    def __init__(self, decoded): self.decoded = decoded; self.messages = None
    def apply_chat_template(self, messages, **kwargs): self.messages = messages; return 'aggregate-only'
    def __call__(self, prompt, **kwargs): return Encoding(input_ids=torch.tensor([[0, 1, 2]]))
    def decode(self, tokens, **kwargs): return self.decoded


class Model:
    training = False
    def __init__(self):
        self.calls = 0
        self.kwargs = None
        self.parameter = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    def parameters(self): return iter([self.parameter])
    def generate(self, **kwargs):
        self.calls += 1
        self.kwargs = kwargs
        return torch.cat([kwargs['input_ids'],torch.tensor([[3, 2]])],dim=1)


class StrategyModelTests(unittest.TestCase):
    def test_local_proposer_keeps_feedback_aggregate_and_counts_failed_calls(self):
        model = Model()
        tokenizer = Tokenizer('[[0,5],[1,2]]')
        proposer = LocalProcedureProposer(model, tokenizer, 'cpu')
        feedback = {'n':4,'canonical_correct':1,'current_correct':[2,2]}
        self.assertEqual(proposer(feedback, PROCEDURES), [[0,5],[1,2]])
        self.assertEqual(model.calls, 1)
        self.assertEqual(len(proposer.cost), 1)
        self.assertFalse(model.kwargs['synced_gpus'])
        import json
        payload = json.loads(tokenizer.messages[-1]['content'])
        self.assertEqual(set(payload), {'aggregate_training_feedback', 'allowed_procedures'})
        self.assertEqual(payload['aggregate_training_feedback'], feedback)
        tokenizer.decoded = '[[0],["medical free text"]]'
        with self.assertRaises(ValueError): proposer(feedback, PROCEDURES)
        self.assertEqual(len(proposer.cost), 2)
        with self.assertRaises(ValueError): proposer(dict(feedback, sample='private'), PROCEDURES)
        self.assertEqual(model.calls, 2)
        model.parameter.requires_grad_(True)
        with self.assertRaises(ValueError): proposer(feedback, PROCEDURES)

    def test_policy_evaluator_training_boundary_and_context(self):
        from types import SimpleNamespace
        model = Model()
        model.generation_config = SimpleNamespace(forced_bos_token_id=None,forced_eos_token_id=None,suppress_tokens=None,begin_suppress_tokens=None)
        processor = SimpleNamespace(tokenizer=Tokenizer('yes'))
        item = {'id':'a','question':'q','answer':'yes','image':'private'}
        with self.assertRaises(ValueError): PolicyStrategyEvaluator(model,processor,[item],'cpu',split='test')
        evaluator = PolicyStrategyEvaluator(model,processor,[item],'cpu',split='train')
        with patch('clinical_pact.multimodal.make_inputs',return_value={'input_ids':torch.tensor([[0,1,2]])}) as build:
            self.assertEqual(evaluator('approved card',('a',)), (1,))
            self.assertTrue(build.call_args.kwargs['append_strategy'])
        self.assertEqual(evaluator.cost[0]['generated_tokens'], 2)
        with self.assertRaises(ValueError): evaluator('',('test-id',))
        self.assertEqual(model.calls, 1)

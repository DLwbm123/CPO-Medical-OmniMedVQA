import unittest
from types import SimpleNamespace
from contextlib import contextmanager
from unittest.mock import patch
import torch
from clinical_pact.dynamic_trainer import DynamicStrategyTrainer
from clinical_pact.fast_rollout import FirstStageRolloutTrainer
from clinical_pact.strategy_cycle import StrategyCycle


@contextmanager
def gathered(policy, accelerator, **kwargs):
    yield policy


class DynamicTrainerTests(unittest.TestCase):
    def make_trainer(self):
        trainer = object.__new__(DynamicStrategyTrainer)
        trainer.strategy_cycle = StrategyCycle(('oldA','oldB'))
        trainer.pact_cards = trainer.strategy_cycle.cards
        trainer.state = SimpleNamespace(global_step=100)
        trainer.model_wrapped = torch.nn.Linear(1, 1).train()
        trainer.accelerator = None
        trainer._buffered_inputs = 'old-rollouts'
        def factory(policy, update):
            self.assertFalse(policy.training)
            return dict(split='train',audit_ids=('a','b'),
                        evaluate=lambda text,ids:(1,0) if not text else (1,1),
                        propose=lambda feedback,procedures:[[0],[1]],count_tokens=lambda text:len(text.split()))
        trainer.refresh_factory = factory
        return trainer

    def test_refresh_restores_train_mode_and_clears_old_rollouts_once(self):
        trainer = self.make_trainer()
        with patch('clinical_pact.dynamic_trainer.unwrap_model_for_generation',gathered), patch.object(FirstStageRolloutTrainer,'_generate_and_score_completions',return_value={'ok':True}):
            self.assertEqual(trainer._generate_and_score_completions([]), {'ok':True})
            self.assertEqual(trainer.strategy_cycle.cycle, 1)
            self.assertTrue(trainer.model_wrapped.training)
            self.assertIsNone(trainer._buffered_inputs)
            trainer.refresh_factory = lambda *args: (_ for _ in ()).throw(AssertionError('must not refresh twice'))
            trainer._generate_and_score_completions([])
            self.assertEqual(trainer.strategy_cycle.cycle, 1)

    def test_failed_refresh_keeps_cycle_and_restores_model(self):
        trainer = self.make_trainer()
        trainer.refresh_factory = lambda *args: (_ for _ in ()).throw(ValueError('invalid proposer'))
        with patch('clinical_pact.dynamic_trainer.unwrap_model_for_generation',gathered):
            with self.assertRaises(ValueError): trainer._generate_and_score_completions([])
        self.assertEqual(trainer.strategy_cycle.cycle, 0)
        self.assertEqual(trainer._buffered_inputs, 'old-rollouts')
        self.assertTrue(trainer.model_wrapped.training)

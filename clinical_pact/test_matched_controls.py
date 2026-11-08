import unittest
from unittest.mock import patch
import torch

from clinical_pact.matched_controls import UnitWeightController, FixedPromptControlTrainer
from clinical_pact.fast_rollout import FirstStageRolloutTrainer


class MatchedControlTests(unittest.TestCase):
    def test_fixed_weight_does_not_adapt_to_low_or_high_audit(self):
        controller = UnitWeightController()
        controller.add_task('old')
        self.assertEqual(controller.states['old'].weight, 1.0)
        for value in (0., 4., 0.):
            controller.observe_uniform_estimates({'old': value})
            deficit = torch.tensor(0.3, requires_grad=True)
            controller.corrected_replay_loss('old', deficit, 1.).backward()
            self.assertEqual(controller.states['old'].weight, 1.0)
            self.assertEqual(deficit.grad.item(), 1.0)
        self.assertEqual(controller.states['old'].observations, 3)

    def test_controls_bypass_dynamic_strategy_refresh(self):
        trainer = object.__new__(FixedPromptControlTrainer)
        trainer.refresh_factory = lambda *args: self.fail('control must not call proposer/audit selection')
        with patch.object(FirstStageRolloutTrainer, '_generate_and_score_completions', return_value='canonical') as generate:
            self.assertEqual(trainer._generate_and_score_completions([]), 'canonical')
            generate.assert_called_once_with(trainer, [])

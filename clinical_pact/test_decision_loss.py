import unittest
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch unavailable')
class DecisionLossTests(unittest.TestCase):
    def test_causal_alignment_and_eos(self):
        from types import SimpleNamespace
        from clinical_pact.decision_loss import reference_targets, replay_deficit

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.tensor([[[0., 0., 3., 0.], [0., 0., 0., 3.], [0., 3., 0., 0.], [0., 0., 0., 0.]]]))
            def forward(self, input_ids, use_cache=False):
                return SimpleNamespace(logits=self.logits)

        model = Toy()
        inputs = {'input_ids': torch.tensor([[0, 2, 3, 1]])}
        tau = reference_targets(model, inputs, 1, eos_id=1)
        self.assertEqual(tau.shape, (1, 3))
        self.assertEqual(replay_deficit(model, inputs, 1, 1, tau).item(), 0)
        with torch.no_grad():
            model.logits[0, 2, 1] = -3
        deficit = replay_deficit(model, inputs, 1, 1, tau)
        self.assertGreaterEqual(deficit.item(), 1)
        deficit.backward()
        self.assertLess(model.logits.grad[0, 2, 1].item(), 0)
        with self.assertRaises(ValueError):
            reference_targets(model, inputs, 1, eos_id=2)

    def test_advantages_match_trl_sample_std(self):
        from clinical_pact.core import question_group_advantages
        rewards = torch.tensor([1., 0., 1., 0., 1., 0., 1., 0.])
        actual = question_group_advantages(rewards, [('q', 1)] * 8)
        expected = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        self.assertTrue(torch.allclose(actual, expected))

    def test_trl_adapter_adds_auxiliary_gradient(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from trl import GRPOTrainer
        from clinical_pact.trainer import SingleProcessReplayTrainer
        from clinical_pact.core import TaskDualController

        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.tensor([[[0., 0., 3.], [3., 0., 0.], [0., 0., 0.]]]))
            def forward(self, input_ids, use_cache=False):
                return SimpleNamespace(logits=self.logits)

        model = Toy()
        controller = TaskDualController()
        controller.add_task('old')
        controller.states['old'].weight = 2.0
        provider = SimpleNamespace(draw=lambda: ('old', 1.0, {'input_ids': torch.tensor([[1, 2, 0]])}, 1, 0, torch.tensor([[1., 1.]])))
        trainer = object.__new__(SingleProcessReplayTrainer)
        trainer.pact_controller = controller
        trainer.pact_replay_provider = provider
        with patch.object(GRPOTrainer, 'compute_loss', lambda *args, **kwargs: model.logits.sum() * 0):
            with torch.no_grad(): model.logits[0, 1, 0] = -2
            loss = trainer.compute_loss(model, {})
            loss.backward()
        self.assertGreaterEqual(loss.item(), 2)
        self.assertLess(model.logits.grad[0, 1, 0].item(), 0)

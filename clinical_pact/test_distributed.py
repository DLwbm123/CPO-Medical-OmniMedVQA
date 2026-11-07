import unittest
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch unavailable')
class TestDistributedPrimitives(unittest.TestCase):
    def test_local_group_and_sampling(self):
        from clinical_pact.core import TaskDualController
        from clinical_pact.distributed import distributed_group_advantages, synchronized_task
        rewards = torch.tensor([1., 0., 0., 1., 0., 1., 0., 1.])
        advantages = distributed_group_advantages(rewards, [('q', 1)] * 8)
        self.assertAlmostEqual(advantages.mean().item(), 0., places=6)
        controller = TaskDualController()
        controller.add_task('old')
        task, probability = synchronized_task(controller, torch.Generator().manual_seed(1))
        self.assertEqual((task, probability), ('old', 1.))

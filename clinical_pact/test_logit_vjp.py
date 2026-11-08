"""Compare local loss VJPs before attributing a model-gradient discrepancy.

CPU primitive tests do not establish CUDA determinism or real-model equivalence.
"""
import unittest
import torch
from clinical_pact.core import decision_deficits, token_margins


class LogitVJPTests(unittest.TestCase):
    def test_gap_vjp_matches_explicit_branch_and_repeated_backward(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                generator = torch.Generator().manual_seed(11)
                target_ids = torch.tensor([[0, 1, 2]])
                base = torch.randn((1, 3, 7), generator=generator) * 0.1
                base.scatter_add_(-1, target_ids.unsqueeze(-1), torch.full((1, 3, 1), 4.0))
                leaf = base.to(dtype).requires_grad_()
                z = leaf.float()
                mask = torch.ones_like(target_ids, dtype=torch.bool)
                tau = torch.full((1, 3), .625)
                values, indices = z[0, -1].topk(2)
                target = int(target_ids[0, -1])
                competitor = int(indices[1] if int(indices[0]) == target else indices[0])
                offset = (token_margins(z, target_ids)[0, -1].detach() + tau[0, -1]).detach()
                modified = z.clone()
                modified[0, -1, competitor] = z[0, -1, competitor] + offset
                actual = decision_deficits(modified, target_ids, mask, tau).sum()
                explicit = 1 - (z[0, -1, target] - z[0, -1, competitor] - offset) / tau[0, -1]
                first = torch.autograd.grad(actual, leaf, retain_graph=True)[0]
                repeated = torch.autograd.grad(actual, leaf, retain_graph=True)[0]
                reference = torch.autograd.grad(explicit, leaf)[0]
                self.assertTrue(torch.equal(first, repeated))
                self.assertTrue(torch.equal(first, reference))
                self.assertEqual(torch.count_nonzero(first).item(), 2)
                self.assertLess(first[0, -1, target].item(), 0)
                self.assertGreater(first[0, -1, competitor].item(), 0)

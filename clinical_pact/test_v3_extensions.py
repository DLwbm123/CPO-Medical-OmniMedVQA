import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from clinical_pact.decision_fisher import FisherPenalty, global_top_indices, merge_fisher
from clinical_pact.dynamic_sampling import uniform_rewards
from clinical_pact.fisher_boundary import calibrate
from clinical_pact.retention_reward import sampled_log_ratio, shape_rewards, unpadded_equal_prefixes
from clinical_pact.v3_eval import categorical_kl


def check_shards(rank, store, state_path):
    dist.init_process_group('gloo', init_method=Path(store).as_uri(), rank=rank, world_size=2)
    try:
        scores = torch.tensor([5., 2., 2., 1.] if rank == 0 else [4., 2., 2., 0.])
        selected, mean = global_top_indices(scores, .5)
        assert selected.tolist() == ([0, 1, 2] if rank == 0 else [0])
        assert float(mean) == 3.25
        assert uniform_rewards([1, 1], 'cpu')
        assert uniform_rewards([0, 0], 'cpu')
        assert not uniform_rewards([rank, rank], 'cpu')

        # Two module names refer to the same physical parameter, sharded into two equal parts.
        model = torch.nn.Module()
        model.embed = torch.nn.Embedding(2, 2)
        model.head = torch.nn.Linear(2, 2, bias=False)
        model.head.weight = model.embed.weight
        parameter = model.embed.weight
        assert tuple(dict(model.named_parameters())) == ('embed.weight',)
        parameter.ds_tensor = torch.tensor([2., 3.]) + rank
        gradient = torch.ones(2)
        reference = torch.tensor([0., 1.])
        f = torch.tensor([2., 3.])
        indices = torch.tensor([0, 1], dtype=torch.int32) + rank*2
        path = Path(state_path)/f'{rank}.pt'
        torch.save({'rank':rank, 'world_size':2, 'entries':{'embed.weight':(indices, f, reference)}}, path)
        optimizer = SimpleNamespace(_get_fp32_grad_state_partition=lambda p, **_: (gradient, 0))
        engine = SimpleNamespace(module=model, optimizer=optimizer, device='cpu')
        penalty = FisherPenalty(path, gamma=.25)
        actual_energy = penalty.apply(engine, .5, audit=True)
        current = parameter.ds_tensor.clone().requires_grad_()
        energy = (f * (current - reference).square()).sum()
        expected_force, = torch.autograd.grad(.25*.5*energy, current)
        torch.testing.assert_close(gradient, 1 + expected_force)
        assert actual_energy == 65.
    finally:
        dist.destroy_process_group()


class ExtensionTests(unittest.TestCase):
    def test_retention_is_detached_and_uses_upper_cap(self):
        raw = torch.tensor([0., 1., 1., 1.])
        actor = torch.tensor([[4., 1.], [5., 1.], [1., 999.], [-2., 1.]], requires_grad=True)
        ref = torch.zeros_like(actor)
        mask = torch.tensor([[1, 1], [1, 1], [1, 0], [1, 0]], dtype=torch.bool)
        drift = sampled_log_ratio(actor, ref, mask)
        torch.testing.assert_close(drift, torch.tensor([5., 6., 1., -2.]))
        self.assertIs(shape_rewards(raw, drift, 0), raw)
        actual = shape_rewards(raw, drift, .05)
        torch.testing.assert_close(actual, torch.tensor([0., .9, .95, 1.1]))
        self.assertFalse(actual.requires_grad)

    def test_prefix_batch_guard_and_token_kl(self):
        ids = torch.tensor([[1, 2], [1, 2]])
        features = {'input_ids':ids, 'attention_mask':torch.ones_like(ids)}
        self.assertTrue(unpadded_equal_prefixes(features))
        features['attention_mask'][0,0] = 0
        self.assertFalse(unpadded_equal_prefixes(features))
        features['attention_mask'][0,0] = 1
        ids[0,0] = 3
        self.assertFalse(unpadded_equal_prefixes(features))
        p = torch.tensor([.75,.25])
        q = torch.tensor([.5,.5])
        torch.testing.assert_close(categorical_kl(p.log(), q.log()), (p*(p.log()-q.log())).sum())
        self.assertEqual(float(categorical_kl(p.log(), p.log())), 0.)

    def test_two_rank_mask_and_synchronized_reward(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(check_shards, (str(Path(directory)/'store'), directory), nprocs=2)

    def test_single_rank_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            store = (Path(directory)/'store').as_uri()
            dist.init_process_group('gloo', init_method=store, rank=0, world_size=1)
            try:
                indices, mean = global_top_indices(torch.tensor([1., 2., 2.]), .5)
                self.assertEqual(indices.tolist(), [1, 2])
                self.assertEqual(float(mean), 2.)
            finally:
                dist.destroy_process_group()

    def test_merge_reanchors_old_and_new_coordinates_and_calibrates(self):
        from safetensors.torch import save_file

        old = (torch.tensor([0,2], dtype=torch.int32), torch.tensor([1.,2.]), torch.tensor([0.,0.]))
        current = torch.tensor([1.,2.,3.,4.])
        selected = torch.tensor([1,2])
        result = merge_fisher(old, current, torch.tensor([3.,4.]), selected, 0)
        self.assertEqual(result[0].tolist(), [0,1,2])
        torch.testing.assert_close(result[1], torch.tensor([1.,3.,6.]))
        torch.testing.assert_close(result[2], current[:3])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/'private').mkdir()
            torch.save({'entries':{'model.language_model.embed_tokens.weight':result}}, out/'private/f2_r0.pt')
            save_file({'model.embed_tokens.weight':current + 1}, out/'weights.safetensors')
            self.assertEqual(calibrate(out, out, 1), .1)


if __name__ == '__main__':
    unittest.main()

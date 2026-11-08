import random
from pathlib import Path
import tempfile
import unittest
import torch
from clinical_pact.compressed_checkpoint import save_compressed_state, load_compressed_state


class CompressedCheckpointTests(unittest.TestCase):
    def test_optimizer_and_views_roundtrip_exactly(self):
        torch.manual_seed(77)
        parameter = torch.nn.Parameter(torch.randn(5, 4))
        optimizer = torch.optim.AdamW([parameter], lr=.01)
        parameter.square().sum().backward()
        optimizer.step()
        shared = torch.arange(100, dtype=torch.float32)
        source = {'model':parameter.detach(), 'optimizer':optimizer.state_dict(), 'rng':random.Random(7).getstate(),
                  'view_a':shared[2:30], 'view_b':shared[20:50]}
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'state.pt'
            save_compressed_state(source, path, byte_limit=1000000)
            restored = load_compressed_state(path)
            self.assertTrue(torch.equal(source['model'], restored['model']))
            for key, value in source['optimizer']['state'][0].items():
                self.assertTrue(torch.equal(value, restored['optimizer']['state'][0][key]))
            self.assertEqual(source['rng'], restored['rng'])
            self.assertEqual(restored['view_a'].untyped_storage().data_ptr(), restored['view_b'].untyped_storage().data_ptr())
            restored['view_a'][18] = 999
            self.assertEqual(restored['view_b'][0].item(), 999)

    def test_file_limit_failure_preserves_prior_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'state.pt'
            save_compressed_state({'value':torch.tensor([3])}, path, byte_limit=10000)
            with self.assertRaises(OSError):
                save_compressed_state({'value':torch.randn(1000)}, path, byte_limit=16)
            self.assertEqual(load_compressed_state(path)['value'].item(), 3)
            self.assertFalse(list(Path(root).glob('.compressed-*')))

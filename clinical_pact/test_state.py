import random
import tempfile
import unittest
from pathlib import Path
from clinical_pact.core import TaskDualController
from clinical_pact.memory import DecisionMemory
from clinical_pact.state import pack_state, unpack_state, atomic_save
from clinical_pact.test_components import record
import torch


class StateTests(unittest.TestCase):
    def test_recovery_keeps_immutable_memory_and_rng(self):
        memory = DecisionMemory(4)
        memory.add_task('a', [record('a', 1)])
        controller = TaskDualController()
        controller.add_task('a')
        controller.observe_uniform_estimates({'a': 0.5})
        rng = random.Random(15)
        state = pack_state(memory, controller, {'cursor': 5}, ('a', 'b'), 3, rng.getstate())
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'state.pt'
            atomic_save(state, path)
            loaded = torch.load(path, map_location='cpu', weights_only=True)
        m2, c2, sampler, cards, cycle, replay_rng = unpack_state(loaded)
        self.assertEqual(m2.tasks['a'], memory.tasks['a'])
        self.assertEqual(c2.states['a'], controller.states['a'])
        self.assertEqual((sampler, cards, cycle), ({'cursor': 5}, ('a', 'b'), 3))
        second = random.Random()
        second.setstate(replay_rng)
        self.assertEqual(second.random(), rng.random())

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

    def test_optimizer_rng_and_scheduler_resume_matches_two_updates(self):
        from clinical_pact.state import pack_runtime_state, restore_runtime_state
        torch.manual_seed(40)
        random.seed(40)
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        memory = DecisionMemory(2)
        controller = TaskDualController()
        pact = pack_state(memory, controller, {'cursor': 1}, ('a', 'b'), 0, random.Random(5).getstate())

        def update(model, optimizer, scheduler):
            x = torch.randn(3, 2) * random.random()
            optimizer.zero_grad()
            model(x).square().mean().backward()
            optimizer.step()
            scheduler.step()

        update(model, optimizer, scheduler)
        weights = {key: value.detach().clone() for key, value in model.state_dict().items()}
        state = pack_runtime_state(optimizer, scheduler, pact)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'runtime.pt'
            atomic_save(state, path)
            state = torch.load(path, map_location='cpu', weights_only=True)
        update(model, optimizer, scheduler)
        expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
        resumed = torch.nn.Linear(2, 1)
        resumed.load_state_dict(weights)
        opt2 = torch.optim.AdamW(resumed.parameters(), lr=0.01)
        sched2 = torch.optim.lr_scheduler.StepLR(opt2, step_size=1, gamma=0.9)
        restore_runtime_state(state, opt2, sched2)
        update(resumed, opt2, sched2)
        for key, value in resumed.state_dict().items():
            self.assertTrue(torch.equal(value, expected[key]))
        self.assertEqual(sched2.state_dict(), scheduler.state_dict())

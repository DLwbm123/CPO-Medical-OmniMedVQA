"""Single-sample, task-balanced risk replay for the gated TRL adapter."""
import random
import torch


class ReplayProvider:
    def __init__(self, memory, controller, load_inputs, seed):
        self.memory = memory
        self.controller = controller
        self.load_inputs = load_inputs
        self.rng = random.Random(seed)

    def draw(self):
        task_ids, probabilities = self.controller.sampling_distribution()
        if not task_ids or set(task_ids) != set(self.memory.tasks):
            raise ValueError('controller and nonempty memory tasks must match')
        probs = probabilities.tolist()
        task = self.rng.choices(task_ids, weights=probs, k=1)[0]
        record = self.rng.choice(self.memory.tasks[task])
        inputs, response_start = self.load_inputs(record)
        if tuple(inputs['input_ids'][0, response_start:].tolist()) != record.response_ids:
            raise ValueError('replay input differs from immutable reference response')
        tau = torch.tensor([[0.5 * min(m, 2.0) for m in record.margins]], dtype=torch.float32)
        return task, probs[task_ids.index(task)], inputs, response_start, record.eos_id, tau

    def get_rng_state(self):
        return self.rng.getstate()

    def set_rng_state(self, state):
        self.rng.setstate(state)

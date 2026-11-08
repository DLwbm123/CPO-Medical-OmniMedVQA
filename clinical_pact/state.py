"""Private PACT extra state for exact recovery alongside trainer checkpoints."""
from dataclasses import asdict
import os
from pathlib import Path
import tempfile
import torch
from .core import TaskDualController, TaskState
from .memory import DecisionMemory, DecisionRecord


def pack_state(memory, controller, sampler_state, cards, cycle, replay_rng_state):
    if set(memory.tasks) != set(controller.states):
        raise ValueError('controller and memory tasks must match')
    return {'version': 1, 'memory_capacity': memory.capacity,
            'memory': {task: [asdict(record) for record in records] for task, records in memory.tasks.items()},
            'controller_config': {k: getattr(controller, k) for k in ('budget', 'learning_rate', 'weight_cap', 'ema_decay', 'exploration', 'risk_prior')},
            'controller_states': {k: asdict(v) for k, v in controller.states.items()},
            'sampler_state': sampler_state, 'cards': tuple(cards), 'cycle': int(cycle), 'replay_rng_state': replay_rng_state}


def unpack_state(value):
    if value.get('version') != 1:
        raise ValueError('unsupported PACT state version')
    memory = DecisionMemory(value['memory_capacity'])
    memory.tasks = {task: [DecisionRecord(**record) for record in records] for task, records in value['memory'].items()}
    controller = TaskDualController(**value['controller_config'])
    controller.states = {task: TaskState(**state) for task, state in value['controller_states'].items()}
    if set(memory.tasks) != set(controller.states) or sum(map(len, memory.tasks.values())) > memory.capacity:
        raise ValueError('invalid checkpoint memory/controller state')
    return memory, controller, value['sampler_state'], tuple(value['cards']), value['cycle'], value['replay_rng_state']


def atomic_save(value, path):
    """Call from rank zero inside the private checkpoint directory only."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.pact-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)

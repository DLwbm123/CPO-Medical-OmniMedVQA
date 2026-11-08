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


def pack_runtime_state(optimizer, scheduler, pact_state):
    """Capture rank-local runtime state alongside the trainer's model checkpoint.

    This helper does not gather ZeRO optimizer shards or install a Trainer
    callback. The caller must save one rank-local RNG state per rank and use
    the backend's own optimizer checkpoint API for partitioned optimizers.
    """
    import copy
    import random
    return {'version': 1, 'optimizer': copy.deepcopy(optimizer.state_dict()),
            'scheduler': copy.deepcopy(scheduler.state_dict()) if scheduler is not None else None,
            'torch_rng': torch.get_rng_state(), 'python_rng': random.getstate(),
            'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
            'pact': copy.deepcopy(pact_state)}


def restore_runtime_state(value, optimizer, scheduler):
    """Restore local state only; model weights must already have been restored."""
    import random
    if value.get('version') != 1:
        raise ValueError('unsupported runtime checkpoint version')
    if (scheduler is None) != (value['scheduler'] is None):
        raise ValueError('scheduler presence differs from checkpoint')
    if value['cuda_rng'] is not None and (not torch.cuda.is_available() or len(value['cuda_rng']) != torch.cuda.device_count()):
        raise ValueError('CUDA RNG topology differs from checkpoint')
    # Validate the PACT payload before mutating optimizer or random state.
    restored = unpack_state(value['pact'])
    optimizer.load_state_dict(value['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(value['scheduler'])
    torch.set_rng_state(value['torch_rng'].cpu())
    random.setstate(value['python_rng'])
    if value['cuda_rng'] is not None:
        torch.cuda.set_rng_state_all(value['cuda_rng'])
    return restored

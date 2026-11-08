"""Lossless streaming checkpoint codec; ZeRO integration remains unvalidated.

No uncompressed temporary file is created. A failed write preserves the old
checkpoint. Compression ratio is data dependent and never an admission proof.
Only load trusted local artifacts: DeepSpeed states can require pickle objects.
"""
import gzip
import os
from pathlib import Path
import tempfile
import torch


class _BoundedWriter:
    def __init__(self, stream, limit):
        self.stream, self.limit, self.written = stream, limit, 0
    def write(self, value):
        if self.written + len(value) > self.limit:
            raise OSError('compressed checkpoint file budget exceeded')
        result = self.stream.write(value)
        self.written += result
        return result
    def flush(self):
        return self.stream.flush()


def save_compressed_state(state, path, *, byte_limit):
    if type(byte_limit) is not int or byte_limit <= 0:
        raise ValueError('positive explicit compressed-file byte limit required')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.compressed-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as raw:
            writer = _BoundedWriter(raw, byte_limit)
            with gzip.GzipFile(fileobj=writer, mode='wb', compresslevel=1, mtime=0) as encoded:
                torch.save(state, encoded, _use_new_zipfile_serialization=False)
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_compressed_state(path, map_location='cpu'):
    with gzip.open(path, 'rb') as decoded:
        return torch.load(decoded, map_location=map_location, weights_only=False)


def make_compressed_checkpoint_engine(*, byte_limit, config_params=None):
    """Explicit opt-in only after a real ZeRO save/load validation.

    Callers must confirm that every consumer uses this engine rather than raw
    torch.load and that total checkpoint/output budgets are independently guarded.
    """
    from deepspeed.runtime.checkpoint_engine.torch_checkpoint_engine import TorchCheckpointEngine
    class CompressedEngine(TorchCheckpointEngine):
        def save(self, state_dict, path):
            save_compressed_state(state_dict, path, byte_limit=byte_limit)
        def load(self, path, map_location=None):
            return load_compressed_state(path, map_location=map_location)
    engine = CompressedEngine(config_params=config_params)
    engine.name = 'ExperimentalLosslessCompressedCheckpoint'
    return engine

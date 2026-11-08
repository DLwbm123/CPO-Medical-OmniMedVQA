"""Frozen single-example stage comparison; aggregate-only, no efficacy control."""
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, set_seed

from .data import load_split
from .dev_stream import save_json
from .multimodal import make_inputs
from .scoring import score_response


def main():
    root = Path(os.environ['OUT'])
    destination = root / 'validation'
    protocol = json.loads((destination / 'protocol.json').read_text())
    rank = int(os.environ['RANK'])
    stage = rank + 1
    if rank not in (0, 1):
        raise ValueError('exactly two checkpoint workers are supported')
    torch.cuda.set_device(rank)
    set_seed(protocol['seed'])
    torch.use_deterministic_algorithms(True)
    model_path = root / f'stage{stage}' / 'model'
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True,
                                             min_pixels=128*28*28, max_pixels=512*28*28)
    processor.tokenizer.padding_side = 'left'
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, dtype=torch.bfloat16, attn_implementation='sdpa',
        local_files_only=True).to(rank).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    config = model.generation_config
    if config.forced_bos_token_id is not None or config.forced_eos_token_id is not None or config.suppress_tokens or config.begin_suppress_tokens:
        raise ValueError('implicit generation constraints are outside the protocol')
    result = {'stage': stage, 'scope': protocol['scope'], 'datasets': []}
    started = time.monotonic()
    for task in protocol['tasks']:
        items = list(load_split(Path(os.environ['ROOT']) / 'data/prepared' / task['name'], 'validation'))
        if len(items) != task['n']:
            raise ValueError('validation denominator differs from the frozen protocol')
        metric = {'task': task['name'], 'expected_n': task['n'], 'evaluated': 0,
                  'correct': 0, 'eos_terminated': 0, 'generated_tokens': 0}
        result['datasets'].append(metric)
        for item in items:
            inputs = make_inputs(processor, item, device=rank)
            with torch.inference_mode():
                tokens = model.generate(**inputs, do_sample=False, num_beams=1,
                                        max_new_tokens=protocol['max_new_tokens'],
                                        repetition_penalty=1.0, use_cache=True,
                                        synced_gpus=False)
            response = tokens[:, inputs['input_ids'].shape[1]:]
            text = processor.tokenizer.decode(response[0], skip_special_tokens=True)
            metric['correct'] += int(score_response(text, item['answer']))
            metric['evaluated'] += 1
            metric['generated_tokens'] += response.numel()
            metric['eos_terminated'] += int(bool(response.numel()) and int(response[0, -1]) == processor.tokenizer.eos_token_id)
            if metric['evaluated'] % 100 == 0 or metric['evaluated'] == task['n']:
                save_json(destination / f'stage{stage}_progress.json', result)
        metric['accuracy'] = metric['correct'] / metric['evaluated']
    result['elapsed_seconds'] = time.monotonic() - started
    result['peak_allocated_gpu_bytes'] = torch.cuda.max_memory_allocated(rank)
    save_json(destination / f'stage{stage}_result.json', result)


if __name__ == '__main__':
    main()

"""Build a task-boundary Fisher shard and calibrate its scale on a retained control."""
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist

from .answer_decisions import PREFIX
from .data import load_split
from .decision_fisher import global_top_indices, merge_fisher
from .score_retained_checkpoints import write_json
from .scoring import normalize
from .v3_memory import TASKS, candidate_tokens


def gold_score(model, processor, item, kind, device):
    from .multimodal import make_inputs

    answer = item['answer'].strip()
    if kind == 'medbook':
        _, options = candidate_tokens(processor.tokenizer, kind)
        target = [options['ABCD'.index(normalize(answer).upper())]]
    elif normalize(answer) in ('yes', 'no'):
        _, options = candidate_tokens(processor.tokenizer, kind)
        target = [options[('yes', 'no').index(normalize(answer))]]
    else:
        target = processor.tokenizer.encode(answer, add_special_tokens=False)
    if not target:
        raise ValueError('Fisher needs a nonempty gold answer')
    features = make_inputs(processor, item, device=device, assistant_prefix=PREFIX)
    gold = torch.tensor([target], device=device)
    features['input_ids'] = torch.cat((features['input_ids'], gold), dim=1)
    features['attention_mask'] = torch.ones_like(features['input_ids'])
    logits = model(**features, use_cache=False, logits_to_keep=len(target)+1).logits[:, :-1]
    return logits.float().log_softmax(-1).gather(-1, gold[..., None]).sum()


def build(root, out, checkpoint, stage, protocol):
    import deepspeed
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from transformers.integrations import HfDeepSpeedConfig

    start = time.monotonic()
    device = torch.device('cuda', int(os.environ['LOCAL_RANK']))
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision('highest')
    dist.init_process_group('nccl')
    rank, world = dist.get_rank(), dist.get_world_size()
    ds = {'train_micro_batch_size_per_gpu': 1, 'gradient_accumulation_steps': 1,
          'zero_allow_untested_optimizer': True, 'bf16': {'enabled': True},
          'zero_optimization': {'stage': 3, 'overlap_comm': False, 'contiguous_gradients': True,
                                'sub_group_size': 100_000_000, 'reduce_bucket_size': 10_000_000}}
    ds_config = HfDeepSpeedConfig(ds)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint, dtype=torch.bfloat16, local_files_only=True, attn_implementation='sdpa',
    ).eval()
    processor = AutoProcessor.from_pretrained(root/'models/base', local_files_only=True,
                                             min_pixels=128*28*28, max_pixels=512*28*28)
    optimizer = torch.optim.SGD(model.parameters(), lr=0)
    engine, *_ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config.config)
    kind = 'medbook' if stage == 2 else 'pathvqa'
    rows = list(load_split(root/'data/prepared'/TASKS[kind], 'train'))
    n = protocol['fisher_samples']
    if not 1 <= n <= 512 or not rows:
        raise ValueError('Fisher requires between 1 and 512 training questions')
    sampled = random.Random(protocol['seed'] + stage).sample(rows, min(n, len(rows)))
    params = {n:p for n,p in engine.module.named_parameters() if p.requires_grad}
    sizes = {n:max(0, min(p.ds_tensor.numel(), p.ds_numel-rank*p.ds_tensor.numel())) for n,p in params.items()}
    diagonal = torch.zeros(sum(sizes.values()), dtype=torch.float32, device=device)
    parts = dict(zip(sizes, diagonal.split(list(sizes.values())), strict=True))
    for item in sampled:
        # Every rank sees the same observation, so reduce-scatter preserves its gradient before squaring.
        engine.optimizer.zero_grad(set_to_none=True)
        engine.optimizer.grad_partitions_flat_buffer.zero_()
        score = gold_score(engine, processor, item, kind, device)
        if not bool(torch.isfinite(score)):
            raise FloatingPointError('nonfinite gold answer score during Fisher measurement')
        engine.backward(score, scale_wrt_gas=False)
        for name, p in params.items():
            g, _ = engine.optimizer._get_fp32_grad_state_partition(p, release_swap_buffers=False)
            g = g[:sizes[name]].float()
            parts[name].addcmul_(g, g, value=1/len(sampled))

    chosen, mean = global_top_indices(diagonal, protocol['fisher_q'])
    prior = {}
    if stage == 3:
        previous = torch.load(out/'private'/f'f2_r{rank}.pt', map_location='cpu', mmap=True, weights_only=True)
        if previous['world_size'] != world or previous['rank'] != rank:
            raise ValueError('prior Fisher does not match current ZeRO rank layout')
        prior = previous['entries']
    cutoff = chosen.new_tensor(list(sizes.values())).cumsum(dim=0)
    ends = torch.searchsorted(chosen, cutoff).tolist()
    entries = {}
    start_coord = cursor = 0
    for (name, p), end in zip(params.items(), ends, strict=True):
        selected = chosen[cursor:end] - start_coord
        if selected.numel() or name in prior:
            weights = parts[name][selected] / mean.float()
            offsets = rank*p.ds_tensor.numel()
            entries[name] = merge_fisher(prior.get(name), p.ds_tensor.flatten()[:sizes[name]], weights, selected, offsets)
        cursor, start_coord = end, start_coord + sizes[name]
    if set(prior) - set(params):
        raise ValueError('canonical Fisher parameter names changed between stages')
    target = out/'private'/f'f{stage}_r{rank}.pt'
    temp = target.with_suffix('.tmp')
    torch.save({'rank':rank, 'world_size':world, 'entries':entries}, temp)
    temp.replace(target)
    count = torch.tensor(sum(state[0].numel() for state in entries.values()), device=device)
    dist.all_reduce(count)
    if rank == 0:
        write_json(out/f'fisher_s{stage}.json', {'task':kind, 'samples':len(sampled),
                   'q':protocol['fisher_q'], 'coordinates':int(count), 'seconds':time.monotonic()-start})
    dist.destroy_process_group()


def checkpoint_key(name, keys):
    candidates = [name]
    if name.startswith('model.language_model.'):
        candidates.append(name.replace('model.language_model.', 'model.', 1))
    if name.startswith('model.visual.'):
        candidates.append(name.removeprefix('model.'))
    if name in ('model.language_model.embed_tokens.weight', 'model.embed_tokens.weight', 'lm_head.weight'):
        candidates.extend(('model.language_model.embed_tokens.weight', 'model.embed_tokens.weight', 'lm_head.weight'))
    return next(candidate for candidate in candidates if candidate in keys)


def calibrate(out, checkpoint, ranks):
    from safetensors import safe_open

    with ExitStack() as stack:
        shards = {p.name:stack.enter_context(safe_open(str(p), framework='pt', device='cpu'))
                  for p in checkpoint.glob('*.safetensors')}
        files = {key:f for f in shards.values() for key in f.keys()}
        if not files:
            raise ValueError('calibration requires a retained safetensors checkpoint')
        states = [torch.load(out/'private'/f'f2_r{i}.pt', map_location='cpu', mmap=True, weights_only=True)
                  for i in range(ranks)]
        names = sorted({name for state in states for name in state['entries']})
        energy = 0.
        for name in names:
            key = checkpoint_key(name, files)
            current = files[key].get_tensor(key).view(-1)
            for state in states:
                entry = state['entries'].get(name)
                if entry is not None:
                    positions, f, anchor = entry
                    for i, w, a in zip(positions.split(1_048_576), f.split(1_048_576), anchor.split(1_048_576), strict=True):
                        delta = current[i.long()].double() - a.double()
                        energy += float(torch.dot(w.double(), delta.square()))
        if not math.isfinite(energy) or energy <= 0:
            raise ValueError('GSPO-prime calibration energy must be finite and positive')
        gamma = 1/energy
        write_json(out/'fisher_gamma.json', {'calibration_energy':energy, 'gamma':gamma})
        return gamma


def main():
    out, root, reference = (Path(os.environ[k]) for k in ('OUT','ROOT','REFERENCE'))
    protocol = json.loads((out/'protocol.json').read_text())
    if os.environ.get('FISHER_PHASE') == 'calibrate':
        gamma = calibrate(out, root/protocol['fisher_calibration_checkpoint'], 1 if protocol.get('single_gpu') else 2)
        write_json(out/'protocol.json', {**protocol, 'fisher_gamma':gamma})
        return
    stage = int(os.environ['STAGE'])
    endpoint = reference/'stage1/model' if stage == 2 else out/'s2/model'
    build(root, out, endpoint, stage, protocol)


if __name__ == '__main__':
    main()

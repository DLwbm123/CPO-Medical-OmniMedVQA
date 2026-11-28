"""Stage A evaluation of label-free majority voting on held-out validation."""
from collections import Counter
from copy import deepcopy
from functools import lru_cache
from math import comb, factorial
import json
import os
from pathlib import Path
import random
import re
import time

import numpy as np

from .choices import bare_letter_tokens, extract_letter
from .tt_data import load_labeled_eval
from ..score_retained_checkpoints import write_json

METRICS = ('pass1', 'maj8', 'maj16')
VOTES = 16


def compositions(total, categories):
    if categories == 1:
        yield (total,)
        return
    for count in range(total + 1):
        for tail in compositions(total-count, categories-1):
            yield (count, *tail)


@lru_cache(maxsize=None)
def multinomial_table(n, k):
    counts = np.array(list(compositions(n, k+1)))
    weights = np.array([factorial(n)/np.prod([factorial(int(x)) for x in row], dtype=np.float64)
                        for row in counts])
    legal = counts[:, :k]
    peaks = legal.max(axis=1)
    winners = np.where((legal == peaks[:, None]).sum(axis=1) == 1, legal.argmax(axis=1), -1)
    return counts, weights, peaks, winners


def exact_votes(probabilities, gold, band):
    k = len(probabilities)-1
    result = {'pass1': float(probabilities[gold])}
    for n in (8, VOTES):
        counts, weights, peaks, winners = multinomial_table(n, k)
        mass = weights * np.power(probabilities, counts).prod(axis=1)
        correct = winners == gold
        result[f'maj{n}'] = float(mass[correct].sum())
        result[f'pass{n}'] = float(1-(1-probabilities[gold])**n)
        if n == VOTES:
            result['agreement'] = np.bincount(peaks, weights=mass, minlength=VOTES+1)
            result['band_correct'] = float(mass[correct & (peaks >= band[0]) & (peaks <= band[1])].sum())
    result['marginal'] = probabilities
    return result


def sampled_votes(votes, k, gold, band):
    if len(votes) != VOTES:
        raise ValueError('expected exactly sixteen samples')
    counts = np.bincount(votes, minlength=k+1)
    hits = int(counts[gold])
    result = {'pass1': hits/VOTES, 'marginal': counts/VOTES}
    for n in (8, VOTES):
        legal = np.bincount(votes[:n], minlength=k+1)[:k]
        peak = legal.max()
        result[f'maj{n}'] = float(legal[gold] == peak and (legal == peak).sum() == 1)
        result[f'pass{n}'] = 1-(comb(VOTES-hits,n)/comb(VOTES,n))
    result['agreement'] = np.eye(VOTES+1)[peak]
    result['band_correct'] = result['maj16'] * (band[0] <= peak <= band[1])
    return result


def summarize(rows, band):
    n = len(rows)
    out = {key: sum(row[key] for row in rows)/n for key in (*METRICS, 'pass8', 'pass16')}
    histogram = sum(row['agreement'] for row in rows)/n
    coverage = float(histogram[band[0]:band[1]+1].sum())
    out.update(n=n, agreement=histogram.tolist(), band_coverage=coverage,
               pseudo_accuracy=float(sum(row['band_correct'] for row in rows)/(n*coverage)) if coverage else None,
               gain=out['maj16']-out['pass1'])
    marginal = Counter(dict.fromkeys('ABCD', 0.0))
    for row in rows:
        marginal.update(dict(zip('ABCD',row['marginal'][:-1])))
        marginal['invalid'] += row['marginal'][-1]
    gold = Counter(row['gold'] for row in rows)
    out['predicted_marginal'] = {key: float(marginal[key]/n) for key in (*'ABCD','invalid')}
    out['gold_marginal'] = {key: gold[key]/n for key in 'ABCD'}
    if 'greedy' in rows[0]:
        out['greedy'] = sum(row['greedy'] for row in rows)/n
    if 'multi_token' in rows[0]:
        out['multi_token_fraction'] = sum(row['multi_token'] for row in rows)/n
    return out


def compare(exact, sampled, z_threshold):
    result = {}
    for key in METRICS:
        deltas = np.array([full[key]-first[key] for first, full in zip(exact,sampled,strict=True)])
        mean, se = float(deltas.mean()), float(deltas.std(ddof=1)/len(deltas)**0.5)
        result[key] = {'a2_minus_matched_a1': mean, 'paired_se': se,
                       'beyond_noise': bool(abs(mean) > z_threshold*se)}
    return result


def evaluate_gate(summary, cfg):
    return (summary['gain'] >= cfg['min_gain']
            and summary['band_coverage'] >= cfg['min_coverage']
            and summary['pseudo_accuracy'] is not None
            and summary['pseudo_accuracy'] >= cfg['min_pseudo_accuracy'])


@lru_cache(maxsize=1)
def reasoning_prompt():
    import yaml

    prompt_dir = Path(__file__).parent
    system = yaml.safe_load((prompt_dir.parent/'prompts.yaml').read_text())['system']
    prompt = yaml.safe_load((prompt_dir/'prompts.yaml').read_text())['tt_reason_user']
    return system, prompt


def input_for(processor, item, mode, *, assistant_prefix='', black_image=False, replacement_image=None):
    from ..multimodal import make_inputs

    if mode == 'canonical':
        if black_image or replacement_image is not None:
            raise ValueError('replacement-image diagnostic requires the reasoning prompt')
        return make_inputs(processor, {'image': item.image, 'question': item.question},
                           device=0,assistant_prefix=assistant_prefix)
    from PIL import Image

    system, prompt = reasoning_prompt()
    question = re.sub(r'<image(?:[_ ]?\d+)?>', '', item.question).strip()
    if black_image and replacement_image is not None:
        raise ValueError('black and replacement images are mutually exclusive')
    with Image.open(item.image if replacement_image is None else replacement_image) as source:
        image = Image.new('RGB',source.size) if black_image else source.convert('RGB')
    content = [{'type': 'image', 'image': image}, {'type': 'text', 'text': prompt + '\n' + question}]
    messages = [{'role':'system','content':system}, {'role':'user','content':content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + assistant_prefix
    return processor(text=[text], images=[image], padding=False, return_tensors='pt').to(0)


def forward_votes(model, tokenizer, inputs, letters, gold, band):
    logits = model(**inputs, logits_to_keep=1, use_cache=False).logits[0, -1].float()
    ids = bare_letter_tokens(tokenizer, letters)
    legal = logits.softmax(0)[ids].cpu().numpy().astype(np.float64)
    ps = np.append(legal, max(0., 1.-legal.sum()))
    ps /= ps.sum()
    first = int(logits.argmax())
    row = exact_votes(ps, letters.index(gold), band)
    row.update(gold=gold, greedy=int(first == ids[letters.index(gold)]))
    return row, first, ids


def sample_answers(model, tokenizer, inputs, letters, gold, cfg, seed):
    import torch

    torch.manual_seed(seed)
    generation = deepcopy(model.generation_config)
    for key, value in cfg['sampling'].items():
        setattr(generation, key, value)
    start = inputs['input_ids'].shape[1]
    tokens = model.generate(**inputs, generation_config=generation,
                            use_model_defaults=False, num_return_sequences=VOTES,
                            synced_gpus=False)[:, start:].tolist()
    answers = [extract_letter(text, letters) for text in tokenizer.batch_decode(tokens, skip_special_tokens=True)]
    votes = [letters.index(letter) if letter is not None else len(letters) for letter in answers]
    row = sampled_votes(votes, len(letters), letters.index(gold), cfg['agreement_band_votes'])
    eos = generation.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos or ())
    def multi_token(response):
        stop = next((i for i, token in enumerate(response) if token in eos), len(response))
        return sum(token not in tokenizer.all_special_ids for token in response[:stop]) > 1

    row.update(gold=gold, multi_token=sum(map(multi_token,tokens))/VOTES)
    return row


def sample_modalities(cfg, root):
    for name, count in cfg['validation_counts'].items():
        data = load_labeled_eval(root/f'MedicalVQA_{name}', 'validation')
        if len(data) != count:
            raise ValueError(f'{name} validation denominator changed')
        rng = random.Random(cfg['seed'])
        cohort = sorted(data, key=lambda pair: pair[0].id)
        if len(cohort) > cfg['a1_limit']:
            cohort = rng.sample(cohort, cfg['a1_limit'])
        yield name, cohort, set(rng.sample(range(len(cohort)), cfg['a2_n']))


def fmt(value):
    return '—' if value is None else f'{100*value:.2f}%'


def report(result):
    lines = ['# 测试时持续强化学习阶段 A', '',
             f"门槛{'通过' if result['passed'] else '未通过'}。判定口径为 {result['gate_source']}，"
             f"{len(result['passing_modalities'])} / {len(result['modalities'])} 个模态通过。",
             '', '所有评分使用未微调基座和 OmniMedVQA 验证集。', '']
    for source in ('A1', 'A2', 'A2_reason'):
        if source not in next(iter(result['modalities'].values())):
            continue
        lines += [f'## {source}', '',
                  '| 模态 | 题数 | 首 token 贪心 | Pass@1 | Maj@8 | Maj@16 | 10–14 票覆盖 | 伪标签准确率 | 无效票 | 通过 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
        for name, rows in result['modalities'].items():
            row = rows[source]
            metrics = [row.get('greedy'), *(row[k] for k in METRICS),
                       row['band_coverage'], row['pseudo_accuracy'], row['predicted_marginal']['invalid']]
            lines.append(f"| {name} | {row['n']} | " + ' | '.join(map(fmt,metrics))
                         + f" | {'是' if row['passes'] else '否'} |")
        lines.append('')
    lines += ['## 首 token 近似核对', '',
              '| 模态 | 裸字母命中 | 解码与前向一致 | 多 token 回答 | A2 − 匹配 A1 Maj@16 | 配对标准误 |',
              '|---|---:|---:|---:|---:|---:|']
    for name, rows in result['modalities'].items():
        check, paired = rows['first_token_check'], rows['matched_comparison']['maj16']
        lines.append(f"| {name} | {check['bare']}/{check['n']} | {check['match']}/{check['n']} | "
                     f"{fmt(rows['A2']['multi_token_fraction'])} | {fmt(paired['a2_minus_matched_a1'])} | "
                     f"{fmt(paired['paired_se'])} |")
    lines += ['', f"阶段 B 候选模态为 {result['candidate_modality'] or '无'}。",
              'JSON 同时记录完整票数分布、覆盖率、字母边际分布和配对差异。', '']
    return '\n'.join(lines)


def run(out, cfg):
    import torch
    from ..validate_development import load_checkpoint

    clock = json.loads((out/'clock.json').read_text())
    started = time.monotonic() - (time.time()-clock['started'])
    resuming = os.environ.get('TT_RESUME') == '1'
    model, processor = load_checkpoint(os.environ['TT_CHECKPOINT'], 0, cfg['seed'])
    tokenizer = processor.tokenizer
    root = Path(os.environ['TT_DATA'])
    cohort_list = list(sample_modalities(cfg,root))
    modes = json.loads((out/'probe_partial.json').read_text()) if resuming else {}
    if resuming:
        if list(modes) != list(cfg['validation_counts']) or any(
            modes[name]['A1']['n'] != len(cohort) or modes[name]['A2']['n'] != len(sample)
            for name, cohort, sample in cohort_list
        ):
            raise ValueError('resume requires completed canonical results for all modalities')
    total = sum(len(cohort) for _, cohort, _ in cohort_list)
    done = 0
    inference_started = time.monotonic()

    def progress(phase, name, processed, size, required, total_required=None):
        if processed == 20 or processed % 25 == 0 or processed == size:
            elapsed = time.monotonic()-inference_started
            rate = elapsed/done
            projection = time.monotonic()-started + rate * max(0, required-done)
            info = dict(phase=phase, modality=name, processed=processed, size=size,
                        elapsed_seconds=time.monotonic()-started, projected_wall_seconds=projection,
                        optional_projected_wall_seconds=(
                            time.monotonic()-started+rate*max(0,total_required-done)
                            if total_required is not None else None),
                        peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated())
            write_json(out/'progress.json',info)
            if done >= 20 and projection > cfg['projected_wall_limit_seconds']:
                raise RuntimeError('stage A forecast exceeds its frozen time limit')

    band = cfg['agreement_band_votes']
    with torch.inference_mode():
        for mode_index, (name, cohort, sample_indices) in enumerate([] if resuming else cohort_list):
            all_exact, matched, samples = [], [], []
            diagnostic = dict(n=0,bare=0,match=0)
            for i, (item, gold) in enumerate(cohort):
                inputs = input_for(processor,item,'canonical')
                exact, first, ids = forward_votes(model,tokenizer,inputs,item.letters,gold,band)
                all_exact.append(exact)
                if i in sample_indices:
                    if diagnostic['n'] < cfg['greedy_check_n']:
                        greedy = model.generate(**inputs,max_new_tokens=1,do_sample=False,num_beams=1,
                                                top_p=1.0,top_k=0,temperature=1.0,
                                                repetition_penalty=1.0,synced_gpus=False)
                        generated = int(greedy[0,-1])
                        diagnostic['n'] += 1
                        diagnostic['bare'] += int(generated in ids)
                        diagnostic['match'] += int(generated == first)
                    seed = cfg['seed'] + mode_index*cfg['a1_limit'] + i
                    samples.append(sample_answers(model,tokenizer,inputs,item.letters,gold,cfg,seed))
                    matched.append(exact)
                done += 1
                progress('canonical',name,i+1,len(cohort),total,total+len(cohort_list)*cfg['a2_n'])
            modes[name] = {'A1':summarize(all_exact,band),'A1_matched':summarize(matched,band),
                           'A2':summarize(samples,band), 'first_token_check':diagnostic,
                           'matched_comparison':compare(matched,samples,cfg['max_paired_z'])}
            write_json(out/'probe_partial.json',modes)

        def needs_sampling(row):
            check = row['first_token_check']
            return (row['A2']['multi_token_fraction'] >= cfg['multi_token_threshold']
                    or check['bare'] != check['n'] or check['match'] != check['n']
                    or any(score['beyond_noise'] for score in row['matched_comparison'].values()))

        source = 'A2' if any(map(needs_sampling,modes.values())) else 'A1'

        def passing(metric_source):
            return [name for name, metrics in modes.items() if metrics[metric_source]['passes']]

        for metrics in modes.values():
            for key in ('A1','A1_matched','A2'):
                metrics[key]['passes'] = evaluate_gate(metrics[key],cfg['gate'])
        fallback = len(passing(source)) < cfg['gate']['min_modalities']
        canonical_source = source
        if fallback:
            source = 'A2_reason'
            done = 0
            inference_started = time.monotonic()
            required = cfg['a2_n'] * sum(source not in metrics for metrics in modes.values())
            for mode_index, (name, cohort, sample_indices) in enumerate(cohort_list):
                if source in modes[name]:
                    continue
                samples = []
                for i in sorted(sample_indices):
                    item,gold = cohort[i]
                    inputs = input_for(processor,item,'reason')
                    seed = cfg['seed'] + mode_index*cfg['a1_limit'] + i
                    samples.append(sample_answers(model,tokenizer,inputs,item.letters,gold,cfg,seed))
                    done += 1
                    progress('reason',name,len(samples),cfg['a2_n'],required)
                metrics = summarize(samples,band)
                metrics['passes'] = evaluate_gate(metrics,cfg['gate'])
                modes[name][source] = metrics
                write_json(out/'probe_partial.json',modes)

    passed_modes = passing(source)
    passed = len(passed_modes) >= cfg['gate']['min_modalities']
    candidate = max(passed_modes,key=lambda name: modes[name][source]['gain']) if passed else None
    result = dict(gate_source=source, canonical_source=canonical_source,
                  reasoning_fallback=fallback, passed=passed, passing_modalities=passed_modes,
                  candidate_modality=candidate, modalities=modes, elapsed_seconds=time.monotonic()-started,
                  peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated())
    write_json(out/'probe_result.json',result)
    (out/'PROBE_RESULT_ZH.md').write_text(report(result))


def main():
    out = Path(os.environ['OUT'])
    run(out,json.loads((out/'protocol.json').read_text()))


if __name__ == '__main__':
    main()

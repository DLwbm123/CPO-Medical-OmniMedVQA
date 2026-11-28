"""Checkpoint-only diagnosis of constrained choice, prior, and image dependence."""
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np

from .choices import extract_letter, letter_tokens
from .pilot_eval import evaluation_cohorts
from .probe import input_for
from .tt_data import parse_options
from ..score_retained_checkpoints import write_json
from ..supervise_semantic_replay import run_workers


def training_prior(root, expected):
    rows = json.loads((root/'MedicalVQA_Fundus'/'jsons'/'train'/'data.json').read_text())
    if len(rows) != expected or len({r['id'] for r in rows}) != expected:
        raise ValueError('Fundus training denominator differs from the frozen diagnostic')
    counts = dict.fromkeys('ABCD',0)
    for row in rows:
        question, answer = (turn['value'] for turn in row['conversations'])
        _, letters = parse_options(question)
        label = extract_letter(answer,letters)
        if label is None:
            raise ValueError('invalid Fundus training label')
        counts[label] += 1
    return max(counts,key=counts.get), counts


def choose(model, processor, item, prompt, prefix, prefix_ids, candidates, black):
    inputs = input_for(processor,item,prompt,assistant_prefix=prefix,black_image=black)
    if inputs['input_ids'][0,-len(prefix_ids):].tolist() != prefix_ids:
        raise ValueError('forced-choice prefix changed after joining it to the prompt')
    scores = model(**inputs,logits_to_keep=1,use_cache=False).logits[0,-1,candidates].float()
    if not bool(scores.isfinite().all()):
        raise ValueError('nonfinite forced-choice scores')
    probabilities = scores.softmax(dim=0)
    best = int(probabilities.argmax())
    return item.letters[best], float(probabilities[best])


def summarize(rows, cfg):
    n = len(rows)
    high = [row for row in rows if row['confidence'] >= cfg['confidence_cutoff']]
    high_wrong = sum(not row['correct'] for row in high)
    ordered = sorted(rows,key=lambda row: -row['confidence'])
    selective = {}
    for coverage in cfg['selective_coverages']:
        k = max(1,round(n*coverage))
        selective[str(coverage)] = {'selected':k,'correct':sum(r['correct'] for r in ordered[:k]),
                                    'accuracy':sum(r['correct'] for r in ordered[:k])/k}
    correct = sum(row['correct'] for row in rows)
    counts = Counter(r['letter'] for r in rows)
    return {'n':n,'correct':correct,'accuracy':correct/n,'predictions':{k:counts[k] for k in 'ABCD'},
            'high_confidence':len(high),'high_confidence_wrong':high_wrong,
            'high_confidence_wrong_rate':high_wrong/n,
            'error_rate_if_high_confidence':high_wrong/len(high) if high else None,
            'selective_accuracy':selective}


def paired_interval(delta, cfg):
    delta = np.asarray(delta,dtype=float)
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    scores = []
    repetitions = cfg['bootstrap_samples']
    for offset in range(0,repetitions,500):
        indices = rng.integers(0,len(delta),size=(min(500,repetitions-offset),len(delta)))
        scores.extend(delta[indices].mean(axis=1))
    bounds = np.quantile(scores,[0.025,0.975]).tolist()
    return {'estimate':float(delta.mean()),'ci95':bounds,'n':len(delta)}


def route(visual, prior_margin, image_gain, cfg):
    if visual < cfg['format_gain_cutoff']:
        return 'return_to_labeled_v3_1'
    if (visual >= cfg['decision_gain_cutoff'] and prior_margin['ci95'][0] > 0
            and image_gain['ci95'][0] > 0):
        return 'prospective_calibration_pilot'
    return 'inconclusive_new_modality_pilot'


def evaluate(out, cfg, pilot, arm):
    import torch
    from ..validate_development import load_checkpoint

    started = time.monotonic()
    model, processor = load_checkpoint(os.environ['TT_CHECKPOINT'],0,pilot['seed'])
    prefix = cfg['assistant_prefix']
    tokens = {letters:letter_tokens(processor.tokenizer,letters,prefix)
              for letters in ('AB','ABC','ABCD')}
    root = Path(os.environ['TT_DATA'])
    variants = ('real','black') if arm in cfg['black_arms'] else ('real',)
    cohorts = list(evaluation_cohorts(root,pilot))
    total = sum(len(rows) for _, rows in cohorts)*len(variants)
    done = 0
    with torch.inference_mode():
        for name, rows in cohorts:
            predictions = {variant:[] for variant in variants}
            for item, label in rows:
                prefix_ids, candidate_ids = tokens[item.letters]
                for variant in variants:
                    result, confidence = choose(model,processor,item,pilot['prompt'],
                                                prefix,prefix_ids,candidate_ids,variant=='black')
                    predictions[variant].append({'id':item.id,'letter':result,
                                                 'correct':result==label,'confidence':confidence})
                    done += 1
                if done % 100 == 0 or done == total:
                    write_json(out/f'{arm}_progress.json',{'modality':name,'processed':done,'total':total,
                                                          'seconds':time.monotonic()-started})
            write_json(out/'private'/f'{arm}_{name}.json',predictions)


def aggregate(out,cfg,pilot):
    root = Path(os.environ['TT_DATA'])
    letter, counts = training_prior(root,cfg['expected_fundus_training_rows'])
    tables, differences = {}, {}
    for name, rows in evaluation_cohorts(root,pilot):
        ids = [item.id for item,_ in rows]
        scores = {}
        tables[name] = {}
        for arm in cfg['real_arms']:
            predictions = json.loads((out/'private'/f'{arm}_{name}.json').read_text())
            scores[arm] = {}
            for variant, records in predictions.items():
                if [row['id'] for row in records] != ids:
                    raise ValueError('checkpoint predictions do not match the frozen cohort order')
                tables[name].setdefault(arm,{})[variant] = summarize(records,cfg)
                scores[arm][variant] = np.array([row['correct'] for row in records],dtype=int)
        before, after = scores['frozen'], scores['tt_v0']
        real = after['real']-before['real']
        black = after['black']-before['black']
        if name == cfg['task']:
            prior = np.array([label==letter for _,label in rows],dtype=int)
            prior_result = {'letter':letter,'train_counts':counts,
                            'correct':int(prior.sum()),'n':len(rows),'accuracy':float(prior.mean())}
            differences = {
                'real_gain':paired_interval(real,cfg),'black_gain':paired_interval(black,cfg),
                'image_specific_gain':paired_interval(real-black,cfg),
                'tt_v0_real_minus_black':paired_interval(after['real']-after['black'],cfg),
                'tt_v0_minus_prior':paired_interval(after['real']-prior,cfg)}
    decision = route(differences['real_gain']['estimate'],differences['tt_v0_minus_prior'],
                     differences['image_specific_gain'],cfg)
    write_json(out/'diagnostic_result.json',{'stage_b_original_decision':'failed',
        'split':'validation','modalities':tables,'fundus_prior':prior_result,
        'fundus_paired_differences':differences,'direction':decision})


def main():
    out = Path(os.environ['OUT'])
    cfg = json.loads((out/'protocol.json').read_text())
    pilot = json.loads((out/cfg['parent_pilot']).read_text())
    if cfg['split'] != 'validation' or pilot['split'] != 'validation' or cfg['task'] != pilot['task']:
        raise ValueError('diagnostic must match the Fundus validation protocol')
    arm = os.environ.get('TT_DIAG_ARM')
    if arm is not None:
        if arm not in cfg['real_arms']:
            raise ValueError('unknown diagnostic checkpoint')
        evaluate(out,cfg,pilot,arm)
        return
    gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if gpus != ['3','5']:
        raise ValueError('diagnostic uses the two authorized GPUs')
    (out/'private').mkdir(mode=0o700)
    started = time.time()
    deadline = started+min(cfg['max_wall_seconds'],23*3600)
    with (out/'clock.json').open('x') as stream:
        json.dump({'started':started,'deadline':deadline},stream)
    root = Path(os.environ['ROOT'])
    def worker(name,gpu,relative):
        pointer = (root/relative).resolve(strict=True)
        if not (pointer/'config.json').is_file():
            raise ValueError('diagnostic checkpoint is incomplete')
        return {'MODULE':'clinical_pact.testtime.decision_diagnostic','TT_DIAG_ARM':name,
                'CUDA_VISIBLE_DEVICES':gpu,'TT_CHECKPOINT':str(pointer)}
    phases = [run_workers(out,cfg,0,[worker('frozen','3','models/base'),
                                   worker('tt_v0','5','series_20261127/run20/model')],deadline)]
    if phases[-1]['reason']=='completed':
        phases.append(run_workers(out,cfg,1,[worker('ttrl','3','series_20261127/run22/model')],deadline))
    if phases[-1]['reason']=='completed':
        aggregate(out,cfg,pilot)
    write_json(out/'status.json',{'reason':phases[-1]['reason'],'phases':phases})
    if phases[-1]['reason']!='completed':
        raise SystemExit(1)


if __name__=='__main__':
    main()

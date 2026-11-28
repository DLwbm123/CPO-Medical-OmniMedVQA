"""P0: score label-free image evidence against held-out answers."""
from copy import deepcopy
import json
import os
from pathlib import Path
import time

import numpy as np

from .choices import extract_letter, letter_tokens
from .counterfactual import CounterfactualPool, image_record
from .grounding import evidence_weight
from .pilot_eval import evaluation_cohorts
from .probe import input_for
from .votes import vote_decision
from ..score_retained_checkpoints import write_json
from ..supervise_semantic_replay import run_workers


def split_images(root, task, split, include_ids=None):
    folder = root/f'MedicalVQA_{task}'
    image_root = (folder/'images').resolve(strict=True)
    rows = json.loads((folder/'jsons'/split/'data.json').read_text())
    selected = [row for row in rows if include_ids is None or str(row['id']) in include_ids]
    if include_ids is not None and {str(row['id']) for row in selected} != include_ids:
        raise ValueError('reserved questions do not align with the source split')
    return [image_record(row,image_root) for row in selected]


def protected_images(root,pilot,replication,additional_holdouts):
    protected = {r.image for task in pilot['validation_counts'] for split in ('validation','test')
                 for r in split_images(root,task,split)}
    manifests = {'Dermoscopy':replication/'private'/'holdout_ids.json', **additional_holdouts}
    for task, path in manifests.items():
        ids = set(json.loads(path.read_text()))
        protected.update(r.image for r in split_images(root,task,'train',ids))
    return protected


def legal_log_probs(model, processor, item, prompt, prefix, choice_tokens, *, black=False,replacement=None):
    inputs = input_for(processor,item,prompt,assistant_prefix=prefix,
                       black_image=black,replacement_image=replacement)
    prefix_ids, choices = choice_tokens
    if inputs['input_ids'][0,-len(prefix_ids):].tolist() != prefix_ids:
        raise ValueError('forced-choice assistant prefix retokenized')
    logits = model(**inputs,logits_to_keep=1,use_cache=False).logits[0,-1,choices].float()
    return logits.log_softmax(dim=0)


def summarize_evidence(records,cfg):
    high = [r for r in records if r['weight'] >= cfg['high_weight_threshold']]
    low = [r for r in records if r['weight'] < cfg['high_weight_threshold']]
    a = sum(r['correct'] for r in high)/len(high) if high else None
    b = sum(r['correct'] for r in low)/len(low) if low else None
    gap = a-b if a is not None and b is not None else None
    pos = np.array([r['evidence'] for r in records if r['correct']])
    neg = np.array([r['evidence'] for r in records if not r['correct']])
    auc = float(((pos[:,None] > neg)+0.5*(pos[:,None] == neg)).mean()) if pos.size and neg.size else None
    band = [r for r in records if r['in_band']]
    eligible = sum(r['weight'] >= cfg['grounding']['min_weight'] for r in band)
    fraction = eligible/len(band) if band else 0.
    effective = ((gap is not None and gap >= cfg['min_accuracy_gap'])
                 or (auc is not None and auc >= cfg['min_auroc']))
    return {'high_n':len(high),'high_accuracy':a,'low_n':len(low),'low_accuracy':b,
            'accuracy_gap':gap,'auroc':auc,'vote_band_n':len(band),'eligible_n':eligible,
            'eligible_fraction':fraction,'passed':effective and fraction >= cfg['min_eligible_fraction']}


def choose_mechanism(metrics,cfg):
    for variant in cfg['choice_priority']:
        if all(row[variant]['passed'] for row in metrics.values()):
            return variant
    return None


def evaluate(out,cfg,pilot):
    import torch
    from ..validate_development import load_checkpoint

    started = time.monotonic()
    root = Path(os.environ['TT_DATA'])
    extra = {'XRay':Path(os.environ['TT_XRAY_HOLDOUT'])} if os.environ.get('TT_XRAY_HOLDOUT') else {}
    protected = protected_images(root,pilot,Path(os.environ['TT_REPLICATION_OUT']),extra)
    private = out/'private'
    private.mkdir(mode=0o700)
    write_json(private/'forbidden_images.json',sorted(map(str,protected)))
    model, processor = load_checkpoint(os.environ['TT_CHECKPOINT'],0,cfg['seed'])
    tokenizer = processor.tokenizer
    options = {s:letter_tokens(tokenizer,s,cfg['assistant_prefix']) for s in ('AB','ABC','ABCD')}
    generation = deepcopy(model.generation_config)
    for k,v in pilot['sampling'].items():
        setattr(generation,k,v)
    metrics, done = {}, 0
    total = sum(cfg['evaluation_questions'].values())
    spec = {**pilot,'task':'Fundus','other_validation_n':cfg['evaluation_questions']['Dermoscopy']}
    with torch.inference_mode():
        for mi,(name,rows) in enumerate(evaluation_cohorts(root,spec)):
            if name not in cfg['evaluation_questions']:
                continue
            if len(rows) != cfg['evaluation_questions'][name] or pilot['validation_counts'][name] != cfg['validation_counts'][name]:
                raise ValueError('P0 cohort does not match the frozen split')
            pool = CounterfactualPool(split_images(root,name,'train'),protected,cfg['seed'])
            variants = {key:[] for key in cfg['choice_priority']}
            fallbacks = ties = 0
            for i,(item,gold) in enumerate(rows):
                replacement,fallback = pool.select(item)
                fallbacks += int(fallback)
                torch.manual_seed(cfg['seed']+mi*10_000+i)
                inputs = input_for(processor,item,cfg['prompt'])
                start = inputs['input_ids'].shape[1]
                output = model.generate(**inputs,generation_config=generation,use_model_defaults=False,
                                        num_return_sequences=cfg['rollouts'],synced_gpus=False)[:,start:]
                letters = [extract_letter(text,item.letters) for text in tokenizer.batch_decode(output,skip_special_tokens=True)]
                vote = vote_decision(letters,item.letters,cfg['agreement_band_votes'])
                ties += int(vote.pseudo_label is None)
                prefix = cfg['assistant_prefix']
                common = (model,processor,item,cfg['prompt'],prefix,options[item.letters])
                real = legal_log_probs(*common)
                mismatched = {'counterfactual':legal_log_probs(*common,replacement=replacement.image),
                              'black':legal_log_probs(*common,black=True)}
                if vote.pseudo_label is not None:
                    j = item.letters.index(vote.pseudo_label)
                    for key, logp in mismatched.items():
                        evidence,weight = evidence_weight(float(real[j]),float(logp[j]),
                            cfg['grounding']['evidence_ratio'],cfg['grounding']['temperature'])
                        variants[key].append({'evidence':evidence,'weight':weight,
                                              'correct':vote.pseudo_label==gold,'in_band':vote.in_band})
                done += 1
                if done%20 == 0 or i+1 == len(rows):
                    elapsed = time.monotonic()-started
                    write_json(out/'progress.json',{'modality':name,'processed':done,'total':total,
                                                    'seconds':elapsed,'projected_seconds':elapsed*total/done})
                    if elapsed*total/done > cfg['projected_wall_limit_seconds']:
                        raise RuntimeError('P0 exceeds its frozen time forecast')
            metrics[name] = {key:summarize_evidence(r,cfg) for key,r in variants.items()}
            metrics[name].update(n=len(rows),fallbacks=fallbacks,ties_or_invalid_plurality=ties)
            write_json(out/'partial_result.json',metrics)
    choice = choose_mechanism(metrics,cfg)
    result = {'selected_negative':choice,'passed':choice is not None,'modalities':metrics,
              'elapsed_seconds':time.monotonic()-started}
    write_json(out/'precheck_result.json',result)
    lines = ['# TT-GC 图像证据前置检查','',
             f'判定用替换图像类型为 {choice or "无"}。','',
             '| 模态 | 替换 | 高权重伪标签准确率 | 低权重伪标签准确率 | AUROC | 可训练比例 | 达标 |',
             '|---|---|---:|---:|---:|---:|---|']
    for task, values in metrics.items():
        for key in cfg['choice_priority']:
            r = values[key]
            data = ['—' if r[k] is None else f'{r[k]:.2%}'
                    for k in ('high_accuracy','low_accuracy','auroc','eligible_fraction')]
            lines.append('| '+' | '.join([task,key,*data,'是' if r['passed'] else '否'])+' |')
    (out/'PRECHECK_RESULT_ZH.md').write_text('\n'.join(lines)+'\n')


def main():
    out = Path(os.environ['OUT'])
    cfg = json.loads((out/'protocol.json').read_text())
    pilot = json.loads((out/cfg['parent_protocol']).read_text())
    if cfg['stage'] != 'P0' or cfg['split'] != 'validation':
        raise ValueError('P0 is fixed to the existing validation cohorts')
    if os.environ.get('TT_P0_WORKER')=='1':
        evaluate(out,cfg,pilot)
        return
    started = time.time()
    deadline = started+min(cfg['max_wall_seconds'],23*3600)
    with (out/'clock.json').open('x') as stream:
        json.dump({'started':started,'deadline':deadline},stream)
    phase = run_workers(out,cfg,0,[{'MODULE':'clinical_pact.testtime.precheck_grounding',
                                   'TT_P0_WORKER':'1'}],deadline)
    write_json(out/'status.json',phase)
    if phase['reason'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()

"""Evaluate frozen and LoRA checkpoints with the preregistered pilot gates."""
import json
import os
from pathlib import Path
import time

import numpy as np

from .choices import letter_tokens
from .counterfactual import CounterfactualPool, ImageRecord
from .decision_diagnostic import summarize
from .pilot_eval import evaluation_cohorts
from .precheck_grounding import legal_log_probs
from .replication import cohorts as heldout_cohorts, image_paired_interval, replication_gate
from ..score_retained_checkpoints import write_json


def development_gate(base,control,adapted,task,cfg):
    f,old,gc = base[task],control[task],adapted[task]
    real = gc['real']['accuracy']-f['real']['accuracy']
    baseline_black = old['black']['accuracy']-f['black']['accuracy']
    adapted_black = gc['black']['accuracy']-f['black']['accuracy']
    old_errors = old['real']['high_confidence_wrong']
    errors = gc['real']['high_confidence_wrong']
    other = {name:adapted[name]['real']['accuracy']-base[name]['real']['accuracy']
             for name in adapted if name!=task}
    if len(other)!=7:
        raise ValueError('development gate requires all seven retained modalities')
    checks = {
        'real_gain':real+1e-12 >= cfg['min_real_gain_over_frozen'],
        'black_gain':adapted_black <= baseline_black*cfg['max_black_gain_fraction_of_full_parameter_tt_v0']+1e-12,
        'high_confidence_errors':errors <= old_errors*cfg['max_high_confidence_errors_fraction_of_full_parameter_tt_v0'],
        'retained_modalities':min(other.values())+1e-12 >= -cfg['max_other_modality_drop']}
    return {'passed':all(checks.values()),'checks':checks,'real_gain':real,
            'baseline_black_gain':baseline_black,'adapted_black_gain':adapted_black,
            'baseline_high_confidence_wrong':old_errors,'adapted_high_confidence_wrong':errors,
            'other_modality_deltas':other}


def evaluate(out,cfg,dev,pilot,arm,root,replication):
    import torch
    from ..validate_development import load_checkpoint

    started = time.monotonic()
    task = cfg['task']
    model,processor = load_checkpoint(os.environ['TT_CHECKPOINT'],0,cfg['seed'])
    if os.environ.get('TT_ADAPTER'):
        from peft import PeftModel

        model = PeftModel.from_pretrained(model,os.environ['TT_ADAPTER'],is_trainable=False).eval()
    manifest = json.loads(Path(os.environ['TT_COUNTERFACTUAL_MANIFEST']).read_text())
    items = [ImageRecord(r['id'],Path(r['image']),r['source_dataset']) for r in manifest]
    forbidden = json.loads(Path(os.environ['TT_FORBIDDEN_IMAGES']).read_text())
    pool = CounterfactualPool(items,forbidden,cfg['seed'])
    if cfg['stage']=='D':
        cohorts = [(name,rows) for name,rows in evaluation_cohorts(root,pilot)]
    else:
        cohorts = [(name,rows) for name,rows,_ in heldout_cohorts(replication,cfg,pilot,root,arm)]
    borrowed = arm in ('frozen','full_tt_v0')
    if borrowed:
        cohorts = [(name,rows) for name,rows in cohorts if name==task]
    decoder = dev['frozen_decoding']
    prefix = decoder['assistant_prefix']
    token_ids = {s:letter_tokens(processor.tokenizer,s,prefix) for s in ('AB','ABC','ABCD')}
    settings = {'confidence_cutoff':decoder['confidence_cutoff'],'selective_coverages':[.25,.5,.75,1.]}
    summaries,done,fallbacks = {},0,0
    total = sum(len(rows) for _,rows in cohorts)
    with torch.inference_mode():
        for name,rows in cohorts:
            variants = ['counterfactual'] if borrowed else (cfg['evaluate_target_views'] if name==task else ['real'])
            data = {k:[] for k in variants}
            for item,gold in rows:
                replacement = None
                if 'counterfactual' in variants:
                    replacement,fallback = pool.select(item)
                    fallbacks += int(fallback)
                for variant in variants:
                    args = {'black':variant=='black','replacement':replacement.image if variant=='counterfactual' else None}
                    logps = legal_log_probs(model,processor,item,pilot['prompt'],prefix,token_ids[item.letters],**args)
                    index = int(logps.argmax())
                    answer = item.letters[index]
                    data[variant].append({'id':item.id,'letter':answer,'correct':answer==gold,
                                          'confidence':float(logps[index].exp())})
                done += 1
                if done%50 == 0 or done == total:
                    write_json(out/'progress.json',{'processed':done,'total':total,'modality':name,
                                                    'seconds':time.monotonic()-started})
            summaries[name] = {view:summarize(records,settings) for view,records in data.items()}
            if name==task and 'real' in data:
                private = out/'private'
                private.mkdir(mode=0o700,exist_ok=True)
                write_json(private/'target_predictions.json',data)
    write_json(out/'evaluation.json',{'arm':arm,'modalities':summaries,'counterfactual_fallbacks':fallbacks,
                                      'seconds':time.monotonic()-started})


def confirmation_gate(out,cfg,pilot,root,replication,adapted):
    task = cfg['task']
    real = json.loads((out/'private'/'target_predictions.json').read_text())['real']
    frozen = json.loads((replication/'private'/f'frozen_{task}.json').read_text())['real']
    cohort = next(heldout_cohorts(replication,cfg,pilot,root,'frozen'))[1]
    ids = [item.id for item,_ in cohort]
    if [r['id'] for r in real] != ids or [r['id'] for r in frozen] != ids:
        raise ValueError('confirmation predictions differ from the replicated holdout order')
    original = json.loads((replication/'amended_replication_result.json').read_text())
    base = original['arms']['tt_v0']['frozen']
    other = {name:adapted[name]['real']['accuracy']-base[name]['real']['accuracy']
             for name in adapted if name!=task}
    if len(other)!=7:
        raise ValueError('confirmation must cover all seven retained modalities')
    delta = np.array([int(a['correct'])-int(b['correct']) for a,b in zip(real,frozen,strict=True)])
    paired = image_paired_interval(delta,[str(item.image) for item,_ in cohort],cfg)
    return replication_gate(base,adapted,paired,other,task,cfg['gate'])


def main():
    out = Path(os.environ['OUT'])
    cfg = json.loads((out/'protocol.json').read_text())
    dev = cfg if cfg['stage']=='D' else json.loads((out/cfg['development_protocol']).read_text())
    pilot = json.loads((out/dev['parent_protocol']).read_text())
    root = Path(os.environ['TT_DATA'])
    replication = Path(os.environ['TT_REPLICATION_OUT'])
    arm = os.environ['TT_ARM']
    evaluate(out,cfg,dev,pilot,arm,root,replication)
    if arm != 'tt_gc':
        return
    adapted = json.loads((out/'evaluation.json').read_text())['modalities']
    if cfg['stage']=='D':
        archive = json.loads((out/dev['development_reference_aggregates'][0]).read_text())['modalities']
        base = {name:r['frozen'] for name,r in archive.items()}
        control = {name:r['tt_v0'] for name,r in archive.items()}
        gate = development_gate(base,control,adapted,cfg['task'],cfg['gate'])
    else:
        gate = confirmation_gate(out,cfg,pilot,root,replication,adapted)
    write_json(out/'gate_result.json',gate)


if __name__ == '__main__':
    main()

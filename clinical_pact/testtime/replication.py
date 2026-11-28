"""Frozen Dermoscopy replication and its aggregate-only scientific decisions."""
import json
import os
from pathlib import Path
import random
import socket
import time

import numpy as np

from .decision_diagnostic import score_cohorts, summarize
from .pilot_eval import evaluation_cohorts
from .redact_train import redact_records
from .tt_data import load_labeled_eval
from ..score_retained_checkpoints import write_json
from ..supervise_semantic_replay import run_workers


def select_modality(probe, previous):
    candidates = set(probe['passing_modalities']) - {'Fundus'}
    dependence = {}
    for name in sorted(candidates):
        frozen = previous['modalities'][name]['frozen']
        dependence[name] = frozen['real']['accuracy'] - frozen['black']['accuracy']
    return max(dependence,key=dependence.get), dependence


def reserve_groups(rows, root, target, seed):
    groups = {}
    for row in rows:
        path = (root/row['image']).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError('source image escaped its root')
        groups.setdefault(path,[]).append(row)
    paths = sorted(groups)
    random.Random(seed).shuffle(paths)
    reserved = []
    for path in paths:
        reserved.extend(groups[path])
        if len(reserved) >= target:
            break
    if len(reserved) < target:
        raise ValueError('holdout is smaller than its frozen minimum')
    reserved_ids = {str(row['id']) for row in reserved}
    if len({str(row['id']) for row in rows}) != len(rows):
        raise ValueError('duplicate source question IDs')
    training = [r for r in rows if str(r['id']) not in reserved_ids]
    holdout_images = {(root/r['image']).resolve() for r in reserved}
    train_images = {(root/r['image']).resolve() for r in training}
    if holdout_images & train_images:
        raise ValueError('holdout and adaptation images overlap')
    return reserved_ids, training, len(holdout_images), len(train_images)


def prepare(out,cfg,root,pilot):
    source = root/f"MedicalVQA_{cfg['task']}"
    records = json.loads((source/'jsons'/'train'/'data.json').read_text())
    if len(records) != cfg['expected_source_train']:
        raise ValueError('source training denominator changed')
    images = (source/'images').resolve(strict=True)
    ids, training, held_images, train_images = reserve_groups(
        records,images,cfg['holdout_target_questions'],cfg['split_seed'])
    private = out/'private'
    private.mkdir(mode=0o700)
    n = redact_records(source,private/'stream',training,pilot['seed'])
    if n < cfg['updates']*pilot['features']['prompts_per_update']:
        raise ValueError('adaptation stream is too short after excluding the holdout')
    write_json(private/'holdout_ids.json',sorted(ids))
    write_json(out/'split_summary.json',{'holdout_questions':len(ids),'holdout_images':held_images,
                                       'available_unlabeled_questions':n,'available_training_images':train_images,
                                       'overlap_images':0})


def cohorts(out,cfg,pilot,root,arm):
    task = cfg['task']
    ids = set(json.loads((out/'private'/'holdout_ids.json').read_text()))
    holdout = load_labeled_eval(root/f'MedicalVQA_{task}','train',include_ids=ids)
    yield task, sorted(holdout,key=lambda pair: pair[0].id), ('real','black')
    if arm != 'frozen':
        for name, rows in evaluation_cohorts(root,pilot):
            if name != task:
                yield name,rows,('real',)


def image_paired_interval(delta, images, cfg):
    _, groups = np.unique(images,return_inverse=True)
    n = len(set(groups))
    sums = np.bincount(groups,weights=delta)
    sizes = np.bincount(groups)
    rng = np.random.default_rng(cfg['bootstrap_seed'])
    means = []
    for offset in range(0,cfg['bootstrap_samples'],500):
        draw = rng.integers(0,n,(min(500,cfg['bootstrap_samples']-offset),n))
        means.extend(sums[draw].sum(axis=1)/sizes[draw].sum(axis=1))
    return {'estimate':float(np.mean(delta)),'ci95':np.quantile(means,[.025,.975]).tolist(),
            'bootstrap_images':n,'n':len(delta)}


def replication_gate(base, adapted, real_gain, retained, task, gate):
    a, b = base[task],adapted[task]
    dependence_drop = (b['real']['accuracy'] - b['black']['accuracy']
                       - a['real']['accuracy'] + a['black']['accuracy'])
    quartile_drop = (b['real']['selective_accuracy']['0.25']['accuracy']
                     - a['real']['selective_accuracy']['0.25']['accuracy'])
    other_mean = sum(retained.values())/len(retained)
    check = {
        'decision_gain':real_gain['estimate']+1e-12 >= gate['min_decision_gain'] and real_gain['ci95'][0] > 0,
        'image_dependence':dependence_drop+1e-12 >= -gate['max_image_dependence_drop'],
        'calibration':b['real']['high_confidence_wrong_rate'] <= gate['max_high_confidence_wrong_rate']
                      or quartile_drop+1e-12 >= -gate['max_top_quartile_accuracy_drop'],
        'retention':other_mean+1e-12 >= -gate['max_retained_macro_drop']
                    and min(retained.values())+1e-12 >= -gate['max_retained_task_drop']}
    return {'passed':all(check.values()),'conditions':check,'real_gain':real_gain,
            'image_dependence_delta':dependence_drop,'top_quartile_delta':quartile_drop,
            'retention_macro_delta':other_mean,'retention_by_modality':retained}


def compare(out,cfg,pilot,root,reused,arm):
    task = cfg['task']
    summaries = {}
    target_correct = {}
    for model in ('frozen',arm):
        summaries[model] = {}
        for name,rows,variants in cohorts(out,cfg,pilot,root,model):
            data = json.loads((out/'private'/f'{model}_{name}.json').read_text())
            if set(data) != set(variants):
                raise ValueError('evaluation views do not match the frozen protocol')
            ids = [item.id for item,_ in rows]
            for view in data.values():
                if [r['id'] for r in view] != ids:
                    raise ValueError('prediction IDs differ from the frozen evaluation order')
            summaries[model][name] = {k:summarize(v,cfg) for k,v in data.items()}
            if name == task:
                target_correct[model] = np.array([row['correct'] for row in data['real']],dtype=int)
                images = [str(item.image) for item,_ in rows]
    base, adapted = summaries['frozen'],summaries[arm]
    for name in adapted.keys() - {task}:
        reference = reused['modalities'][name]['frozen']['real']
        if reference['n'] != adapted[name]['real']['n']:
            raise ValueError('retained-modality evaluation denominator changed')
        base[name] = {'real':reference}
    retained = {name:adapted[name]['real']['accuracy']-base[name]['real']['accuracy']
                for name in adapted if name != task}
    if len(retained) != 7:
        raise ValueError('the seven other modalities must be scored')
    paired = image_paired_interval(target_correct[arm]-target_correct['frozen'],images,cfg)
    return {'frozen':base,'adapted':adapted,'gate':replication_gate(base,adapted,paired,retained,task,cfg['gate'])}


def training_summary(destination):
    result = json.loads((destination/'training_result.json').read_text())
    timings = json.loads((destination/'timing.json').read_text())
    votes = [json.loads(line) for line in (destination/'votes.jsonl').read_text().splitlines()]
    result.update(image_gated_questions=sum(r['image_gate_skips'] for r in votes),
                  total_questions=sum(len(r['questions']) for r in votes),
                  brake_steps=sum(r['brake'] for r in timings),
                  peak_reserved_bytes=max(r['peak_reserved_bytes'] for r in timings))
    return result


def write_report(out,result):
    split = result['split']
    checks = result['arms']
    n = split['holdout_questions']
    lines = ['# Dermoscopy 独立复核结果','',
             f'验证集共 {n} 题，来自 {split["holdout_images"]} 张图像。训练流与验证图像路径重叠为零。',
             '', '| 组别 | 真实图 | 黑图 | 视觉差距变化 | 高置信误答 | Top-25% 准确率 | 关口 |',
             '|---|---:|---:|---:|---:|---:|---|']
    frozen = next(iter(checks.values()))['frozen']['Dermoscopy']
    def metrics(row):
        return [f'{row["real"]["accuracy"]:.2%}',f'{row["black"]["accuracy"]:.2%}',
                f'{row["real"]["high_confidence_wrong"]}/{n}',
                f'{row["real"]["selective_accuracy"]["0.25"]["accuracy"]:.2%}']
    a,b,c,d = metrics(frozen)
    lines.append(f'| Frozen + 约束解码 | {a} | {b} | — | {c} | {d} | — |')
    for arm, r in checks.items():
        a,b,c,d = metrics(r['adapted']['Dermoscopy'])
        check = r['gate']
        lines.append(f'| {arm} | {a} | {b} | {100*check["image_dependence_delta"]:+.2f} pp | '
                     f'{c} | {d} | {"通过" if check["passed"] else "未通过"} |')
    for arm, r in checks.items():
        g = r['gate']
        x, (lo,hi) = g['real_gain']['estimate'],g['real_gain']['ci95']
        other = ', '.join(f'{name} {100*d:+.2f} pp' for name,d in g['retention_by_modality'].items())
        lines.extend(['',f'{arm} 的真实图配对增益为 {100*x:+.2f} pp，95% 区间 '
                      f'[{100*lo:+.2f}, {100*hi:+.2f}]。其余模态宏平均 '
                      f'{100*g["retention_macro_delta"]:+.2f} pp。各模态变化为 {other}。'])
    directions = {'design_stage_c':'进入阶段 C 设计。','image_gate_core':'图像门控纳入主方法。',
                  'return_to_labeled_v3_1':'结束测试时路线，回到有标签的 v3.1。'}
    lines.extend(['',directions[result['direction']],'','Fundus 原阶段 B 判定为未通过。',''])
    (out/'DERMOSCOPY_PILOT_ZH.md').write_text('\n'.join(lines))


def main():
    out = Path(os.environ['OUT'])
    cfg = json.loads((out/'protocol.json').read_text())
    pilot = json.loads((out/'pilot_protocol.json').read_text())
    root = Path(os.environ['TT_DATA'])
    arm = os.environ.get('TT_REPLICATION_EVAL')
    if arm:
        score_cohorts(out,cfg,pilot,arm,list(cohorts(out,cfg,pilot,root,arm)))
        return
    previous = json.loads((out/'decision_diagnostic_result.json').read_text())
    probe = json.loads((out/'probe_result.json').read_text())
    selected, dependence = select_modality(probe,previous)
    gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    if selected != cfg['task'] or gpus != ['3','5']:
        raise ValueError('selected modality or assigned GPUs differ from the frozen replication')
    started = time.time()
    deadline = started + min(cfg['max_wall_seconds'],23*3600)
    clock = {'started':started,'deadline':deadline}
    with (out/'clock.json').open('x') as stream:
        json.dump(clock,stream)
    prepare(out,cfg,root,pilot)
    model = str(Path(os.environ['TT_CHECKPOINT']).resolve(strict=True))
    phases = []
    def phase(name, workers):
        write_json(out/'progress.json',{'phase':name,'state':'running'})
        result = run_workers(out,cfg,len(phases),workers,deadline)
        phases.append({'name':name,**result})
        write_json(out/'status.json',{'reason':'running' if result['reason']=='completed' else result['reason'],
                                     'phases':phases})
        if result['reason']!='completed':
            raise SystemExit(1)
    def evaluate(arm, checkpoint):
        phase(f'eval_{arm}',[{'MODULE':'clinical_pact.testtime.replication','TT_REPLICATION_EVAL':arm,
                             'CUDA_VISIBLE_DEVICES':'5','TT_CHECKPOINT':checkpoint}])
    def train(arm, destination, name, smoke=False):
        destination.mkdir()
        arm_spec = {**pilot,'stage':'B_replication','task':cfg['task']}
        arm_spec['features'] = {**pilot['features'],'updates':cfg['updates']}
        arm_spec['arms'] = {'tt_v0':pilot['arms']['tt_v0'],
                          'image_gate':{**pilot['arms']['tt_v0'],'image_gate_threshold':cfg['image_gate_threshold']}}
        if smoke:
            arm_spec['acceptance_updates'] = 1
        write_json(destination/'protocol.json',arm_spec)
        write_json(destination/'clock.json',clock)
        with socket.socket() as s:
            s.bind(('127.0.0.1',0))
            port = str(s.getsockname()[1])
        phase(name,[{'MODULE':'clinical_pact.testtime.pilot_train','TT_ARM':arm,
                              'OUT':str(destination),'TT_DATA':'','TT_CHECKPOINT':model,
                              'TT_REDACTED_TASK':str(out/'private'/'stream'),'MASTER_ADDR':'127.0.0.1',
                              'MASTER_PORT':port,'RANK':str(rank),'LOCAL_RANK':str(rank),'WORLD_SIZE':'2'}
                             for rank in range(2)])

    smoke = out/'s0'
    train('image_gate',smoke,'verify_image_gate',smoke=True)
    trial = json.loads((smoke/'votes.jsonl').read_text().strip())
    votes = trial['questions']
    count = sum(q['blind_vote_probability'] is not None
                and q['blind_vote_probability'] >= cfg['image_gate_threshold'] for q in votes)
    if (len(votes) != pilot['features']['prompts_per_update'] or trial['image_gate_skips'] != count
            or not any(q['blind_vote_probability'] is not None for q in votes)):
        raise ValueError('image gate verification failed')
    evaluate('frozen',model)
    results = {}
    for index, arm in enumerate(cfg['arms'],1):
        destination = out/f's{index}'
        train(arm,destination,f'train_{arm}')
        evaluate(arm,str(destination/'model'))
        results[arm] = compare(out,cfg,pilot,root,previous,arm)
        results[arm]['training'] = training_summary(destination)
        write_json(out/f'{arm}_aggregate.json',results[arm])
    if results['tt_v0']['gate']['passed']:
        direction = 'design_stage_c'
    elif results['image_gate']['gate']['passed']:
        direction = 'image_gate_core'
    else:
        direction = 'return_to_labeled_v3_1'
    aggregate = {'task':selected,'frozen_selection_gaps':dependence,'arms':results,'direction':direction,
                 'stage_b_original_decision':'failed','split':json.loads((out/'split_summary.json').read_text())}
    write_json(out/'replication_result.json',aggregate)
    write_report(out,aggregate)
    write_json(out/'status.json',{'reason':'completed','phases':phases})
    write_json(out/'progress.json',{'phase':'completed','state':direction})


if __name__ == '__main__':
    main()

"""Fixed-prompt GSPO with v3 answer-margin replay and per-task flip budgets."""
import json
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F
from transformers import AutoProcessor, TrainerCallback, set_seed
import yaml

from .data import load_split
from .dev_stream import Images, reward
from .dynamic_sampling import UniformGroup, uniform_rewards
from .matched_controls import FixedPromptControlTrainer, control_training_args
from .score_retained_checkpoints import write_json
from .shard_projection import combine_replay_gradient
from .strategy_cycle import StrategyCycle
from .v3_memory import TASKS, candidate_tokens, decision_views, fp32_scores, margin, memory_pool


def anchored_margin_loss(current, initial, beta, anchored=True):
    target = max(0.5 * initial, 0.5) if anchored else 0.5
    return beta * F.softplus((target-current)/beta)


class StepTiming(TrainerCallback):
    def __init__(self, destination, clock, protocol, stage):
        self.path = destination/'timing.json'
        self.started = clock['started']
        self.total_steps = protocol['updates'] * (4-stage)
        self.eval_reserve = 3600 if stage == 2 else 1800
        self.limit = protocol.get('projected_wall_limit_seconds', 72_000)
        self.rows = []

    def on_step_begin(self, *args, **kwargs):
        self.begin = time.monotonic()

    def on_step_end(self, args, state, control, **kwargs):
        device = torch.device('cuda', torch.cuda.current_device())
        seconds = torch.tensor(time.monotonic()-self.begin, device=device, dtype=torch.float64)
        peak = torch.tensor([torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)], device=device)
        peaks = [torch.empty_like(peak) for _ in range(dist.get_world_size())]
        dist.all_reduce(seconds, op=dist.ReduceOp.MAX)
        dist.all_gather(peaks, peak)
        row = {'step':state.global_step, 'seconds':float(seconds),
               'peak_allocated_bytes':[int(x[0]) for x in peaks], 'peak_reserved_bytes':[int(x[1]) for x in peaks]}
        self.rows.append(row)
        stop = torch.zeros((), device=device, dtype=torch.int)
        if dist.get_rank() == 0:
            if state.global_step == 20:
                average = sum(r['seconds'] for r in self.rows[10:20])/10
                row['projected_wall_seconds'] = (time.time()-self.started
                    + (self.total_steps-state.global_step)*average + self.eval_reserve)
                stop.fill_(row['projected_wall_seconds'] > self.limit)
            write_json(self.path, self.rows)
        dist.broadcast(stop, src=0)
        if bool(stop):
            raise RuntimeError('forecast exceeds the frozen wall-time limit; see timing.json')


class V3Trainer(FixedPromptControlTrainer):
    def __init__(self, *args, root, reference, out, protocol, stage, **kwargs):
        super().__init__(*args, old_records=[], control_arm='plain_gspo', **kwargs)
        self.stage, self.out, self.spec = stage, out, protocol
        self.dynamic_sampling = bool(protocol.get('dynamic_sampling'))
        self.dynamic_history = []
        self._dynamic_verifier_inputs = None
        if self.dynamic_sampling:
            self._question_indexes = {row['id']:i for i,row in enumerate(self.train_dataset.rows)}
            if len(self._question_indexes) != len(self.train_dataset):
                raise ValueError('deterministic resampling requires unique question IDs')
        self.fisher = None
        if protocol['replay_views'] <= 0 or protocol['replay_views'] % self.accelerator.num_processes:
            raise ValueError('positive replay view count must be divisible by the rank count')
        if protocol.get('fisher_tr'):
            from .decision_fisher import FisherPenalty

            rank = dist.get_rank()
            self.fisher = FisherPenalty(out/'private'/f'f{stage}_r{rank}.pt', protocol['fisher_gamma'])
        self.kinds = ('medbook',) if stage == 2 else ('medbook', 'pathvqa')
        self.new_kind = self.kinds[-1]
        self.tokens = {kind: candidate_tokens(self.pact_processor.tokenizer, kind) for kind in self.kinds}
        self.pools = {}
        rotations = 4 if protocol['rotations'] else 1
        for kind in self.kinds:
            pool, _ = memory_pool(root, reference, kind)
            self.pools[kind] = decision_views(pool, kind, rotations)
        self.records = {}
        self.weights = dict.fromkeys(self.kinds, 0.0)
        if stage == 3:
            old = json.loads((out/'private'/'s2.json').read_text())
            self.records.update(old['records'])
            self.weights.update(old['weights'])
        self.cohort_summary = {}
        self.gradient_history = []
        self.initialized = False
        self.bound_step = None

    def _generate(self, prompts):
        result = super()._generate(prompts)
        if self._dynamic_verifier_inputs is not None:
            completion_ids, completions = result[1], result[3]
            answers = [row['answer'] for row in self._dynamic_verifier_inputs]
            if uniform_rewards(reward(completions, answers), self.accelerator.device):
                tokens = sum(len(ids) for ids in completion_ids)
                counts = torch.tensor(tokens, device=self.accelerator.device)
                dist.all_reduce(counts)
                raise UniformGroup(int(counts))
        return result

    def _generate_and_score_completions(self, inputs):
        if not self.dynamic_sampling or not self.model.training:
            return super()._generate_and_score_completions(inputs)
        origin = inputs[0]['pact_question_id']
        index = self._question_indexes[origin]
        for attempt in range(3):
            if attempt:
                row = self.train_dataset[(index + attempt) % len(self.train_dataset)]
                inputs = [row] * len(inputs)
            self._dynamic_verifier_inputs = inputs if attempt < 2 else None
            try:
                return super()._generate_and_score_completions(inputs)
            except UniformGroup as skip:
                self.dynamic_history.append({'step':self.state.global_step, 'attempt':attempt+1,
                    'skipped_question':inputs[0]['pact_question_id'], 'extra_rollouts':self.num_generations,
                    'extra_completion_tokens':skip.args[0]})
            finally:
                self._dynamic_verifier_inputs = None

    def _scores(self, model, kind, index):
        item, target = self.pools[kind][index]
        return fp32_scores(model, self.pact_processor, item, self.accelerator.device, self.tokens[kind]), target

    def _shard(self, indices):
        rank = self.accelerator.process_index
        size = self.accelerator.num_processes
        for pos in range(rank, ((len(indices)+size-1)//size)*size, size):
            yield indices[pos % len(indices)], pos < len(indices)

    def _calibrate(self, model, kind):
        scores = {}
        for index, count in self._shard(range(len(self.pools[kind]))):
            logits, target = self._scores(model, kind, index)
            if not bool(torch.isfinite(logits).all()):
                raise FloatingPointError('nonfinite reference option scores')
            if count:
                scores[index] = (int(logits.argmax()) == target, float(margin(logits, target)))
        ranks = [None] * self.accelerator.num_processes
        dist.all_gather_object(ranks, scores)
        merged = {index: record for rank in ranks for index, record in rank.items()}
        width = 4 if kind == 'medbook' and self.spec['rotations'] else 1
        self.records[kind] = [[i, merged[i][1]] for start in range(0, len(merged), width)
                              if all(merged[i][0] and merged[i][1] > 0 for i in range(start, start+width))
                              for i in range(start, start+width)]
        if not self.records[kind]:
            raise ValueError('no correct positive-margin reference question in all required views')
        self.cohort_summary[kind] = {'candidates': len(merged)//width,
                                     'questions': len(self.records[kind])//width, 'views': len(self.records[kind])}

    def _audit(self, model, step):
        mode = model.training
        model.eval()
        audit = {}
        try:
            with torch.no_grad():
                if step == 0:
                    self._calibrate(model, self.new_kind)
                for kind, records in self.records.items():
                    flips = torch.zeros((), device=self.accelerator.device, dtype=torch.int64)
                    for index, include in self._shard([i for i, _ in records]):
                        scores, target = self._scores(model, kind, index)
                        if not bool(torch.isfinite(scores).all()):
                            raise FloatingPointError('nonfinite audit answer scores')
                        if include:
                            flips += (scores.argmax() != target).to(torch.int64)
                    dist.all_reduce(flips)
                    wrong = int(flips)
                    rate = wrong / len(records)
                    w = self.weights[kind] + self.spec['dual_rate']*(rate-self.spec['flip_budget'])
                    self.weights[kind] = min(self.spec['dual_cap'], max(0.0, w))
                    audit[kind] = {'flips': wrong, 'n': len(records), 'weight': self.weights[kind]}
        finally:
            model.train(mode)
        record = {'step':step, 'tasks':audit}
        if self.fisher:
            record['omega_f'] = self.fisher.apply(self.boundary_engine, audit=True)
        self.audit_history.append(record)
        if self.accelerator.is_main_process:
            stage_dir = self.out/f's{self.stage}'
            write_json(stage_dir/'audit.json', {'calibration': self.cohort_summary, 'audits': self.audit_history})
            write_json(self.out/'private'/f's{self.stage}.json', {'records': self.records, 'weights': self.weights})
        if step == 0 and audit[self.new_kind]['flips'] != 0:
            raise RuntimeError('new memory must reproduce every step-zero decision before any update')

    def _prepare_step(self, model, step):
        if step % self.spec['audit_interval'] == 0:
            self._audit(model, step)
        # Use the same number of views per update, even after both old tasks exist.
        kind = self.kinds[step % len(self.kinds)]
        records = self.records[kind]
        rng = random.Random(self.spec['seed']*10000+self.stage*1000+step)
        n = self.spec['replay_views']
        chosen = rng.sample(records, n) if len(records) >= n else rng.choices(records, k=n)
        rank = self.accelerator.process_index
        local = chosen[rank::self.accelerator.num_processes]

        def memory_loss(index, initial):
            scores, target = self._scores(model, kind, index)
            return anchored_margin_loss(margin(scores, target), initial, self.spec['beta'], self.spec['anchored'])

        self.bound_step = (step, kind, [lambda index=i, initial=g: memory_loss(index, initial) for i, g in local])

    def _install_boundary(self):
        wrapper = self.accelerator.deepspeed_engine_wrapped
        engine = wrapper.engine
        ranks = 1 if self.spec.get('single_gpu') else 2
        if engine.zero_optimization_stage() != 3 or self.accelerator.num_processes != ranks:
            raise RuntimeError('the actual ZeRO-3 rank count differs from the replay protocol')
        ordinary_backward = wrapper.backward
        self.boundary_engine = engine

        def backward(loss, sync_gradients=True, **kwargs):
            if not sync_gradients:
                return ordinary_backward(loss, sync_gradients=False, **kwargs)
            engine.set_gradient_accumulation_boundary(True)
            engine.backward(loss, **kwargs)
            step, kind, losses = self.bound_step
            coefficient = self.weights[kind]
            fisher_coefficient = max(self.weights.values()) if self.fisher else 0.
            if self.spec.get('memory_smoke'):
                fisher_coefficient = 0.5  # Exercise the penalty buffers even at the zero-flip first step.
            after_projection = (lambda: self.fisher.apply(engine, fisher_coefficient)) if fisher_coefficient else None
            stats = combine_replay_gradient(engine, losses, coefficient, project=self.spec['projection'],
                                            after_projection=after_projection)
            engine.step()
            norm = float(engine.get_global_grad_norm())
            clip = engine.gradient_clipping()
            scale = min(1., clip / (norm + 1e-6)) if clip > 0 else 1.
            retention = stats['preclip_retention_norm']
            stats.update(step=step, task=kind, preclip_total_norm=norm, clip_scale=scale,
                         fisher_coefficient=fisher_coefficient, postclip_retention_norm=retention*scale,
                         retention_to_total_norm=retention/norm if norm else None,
                         postclip_replay_coefficient=coefficient*scale, postclip_fisher_coefficient=fisher_coefficient*scale)
            for component in ('projection','replay','fisher'):
                stats[f'postclip_{component}_norm'] = stats[f'preclip_{component}_norm'] * scale
            self.gradient_history.append(stats)

        wrapper.backward = backward
        self.initialized = True

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        step = self.state.global_step
        if not self.initialized:
            self._install_boundary()
        if self.bound_step is None or self.bound_step[0] != step:
            self._prepare_step(model, step)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)


def main():
    root, out, reference = (Path(os.environ[k]) for k in ('ROOT', 'OUT', 'REFERENCE'))
    stage = int(os.environ['STAGE'])
    protocol = json.loads((out/'protocol.json').read_text())
    kl_beta = float(protocol.get('r1_beta', 0.))
    if protocol.get('r1_alpha') or not 0 <= kl_beta < float('inf'):
        raise ValueError('use finite nonnegative r1_beta for native TRL KL')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    torch.set_float32_matmul_precision('highest')
    set_seed(protocol['seed'])
    initial = reference/'stage1/model' if stage == 2 else out/'s2/model'
    processor = AutoProcessor.from_pretrained(root/'models/base', local_files_only=True,
                                             min_pixels=128*28*28, max_pixels=512*28*28)
    processor.tokenizer.padding_side = 'left'
    task = 'pathvqa' if stage == 2 else 'slake'
    rows = list(load_split(root/'data/prepared'/TASKS[task], 'train'))
    if len(rows) != protocol['training_sizes'][task]:
        raise ValueError('frozen training denominator changed')
    templates = yaml.safe_load((Path(__file__).parent/'prompts.yaml').read_text())
    destination = out/f's{stage}'
    destination.mkdir(parents=True, exist_ok=True)
    args = control_training_args(destination, protocol['updates'], protocol['seed'],
                                 single_gpu=protocol.get('single_gpu', False), kl_beta=kl_beta)
    common = dict(model=str(initial), args=args, processing_class=processor, reward_funcs=reward,
                  train_dataset=Images(rows, templates), strategy_cycle=StrategyCycle(('', '')),
                  refresh_factory=None, processor=processor)
    if protocol.get('mode') == 'no_memory':
        trainer = FixedPromptControlTrainer(**common, old_records=[], control_arm='plain_gspo')
    else:
        trainer = V3Trainer(**common, root=root, reference=reference, out=out,
                            protocol=protocol, stage=stage)
    if kl_beta:
        ref = getattr(trainer.ref_model, 'module', trainer.ref_model)
        if ref is None or Path(ref.config._name_or_path).resolve() != initial.resolve():
            raise RuntimeError('native TRL KL did not use the stage-start checkpoint')
    trainer.batched_same_prompt_generation = bool(protocol.get('batched_same_prompt_generation'))
    clock = json.loads((out/'clock.json').read_text())
    trainer.add_callback(StepTiming(destination, clock, protocol, stage))
    trainer.train()
    omega = trainer.fisher.apply(trainer.boundary_engine, audit=True) if getattr(trainer, 'fisher', None) else None
    if not protocol.get('memory_smoke'):
        trainer.save_model(str(destination/'model'))
    if dist.get_rank() == 0:
        report = {'updates':trainer.state.global_step, 'audits':trainer.audit_history,
                  'gradient_geometry':getattr(trainer,'gradient_history',[]), 'logs':trainer.state.log_history}
        if omega is not None:
            report['fisher_omega_final'] = omega
        if protocol.get('dynamic_sampling'):
            skipped = trainer.dynamic_history
            write_json(out/'private'/f's{stage}_resampling.json', skipped)
            report['resampling'] = {k:sum(x[k] for x in skipped) for k in ('extra_rollouts', 'extra_completion_tokens')}
        write_json(destination/'training.json', report)
    trainer.accelerator.wait_for_everyone()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

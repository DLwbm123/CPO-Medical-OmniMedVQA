"""Matched-start semantic replay on fixed MedBook answer views."""
import json
import os
from pathlib import Path
import random

import torch
import torch.distributed as dist
from torch.nn import functional as F
from transformers import AutoProcessor, set_seed
import yaml

from .answer_decisions import choice_logits, choice_token_ids, choice_view
from .core import TaskDualController
from .data import load_split
from .dev_stream import Images, reward
from .matched_controls import FixedPromptControlTrainer, control_training_args
from .score_retained_checkpoints import write_json
from .strategy_cycle import StrategyCycle
from .validate_development import load_checkpoint


def known_views(reference):
    old = torch.load(reference / 'private_memory.pt', map_location='cpu', weights_only=False)
    if len(old) != 338:
        raise ValueError('expected the retained MedBook reference pool')
    return [choice_view(item, shift) for item, _ in old for shift in (0, 1)]


def calibrate(out, reference, protocol):
    start = reference / 'stage1/model'
    model, processor = load_checkpoint(start, 0, protocol['seed'])
    views = known_views(reference)
    ids = choice_token_ids(processor.tokenizer)
    logits = []
    with torch.inference_mode():
        for item, _ in views:
            scores = choice_logits(model, processor, item, 0, ids)
            if not bool(torch.isfinite(scores).all()):
                raise ValueError('non-finite initial answer scores')
            logits.append(scores.tolist())
    selected = [i for i in range(0, len(views), 2)
                if all(max(range(4), key=logits[j].__getitem__) == views[j][1] for j in (i, i+1))]
    write_json(out / 'calibration.json', {'candidate_questions': len(views)//2,
                                        'eligible_questions': len(selected), 'views_per_question': 2})
    if len(selected) < protocol['minimum_reference_questions']:
        raise ValueError('insufficient initial correct references across both answer views')
    write_json(out / 'private' / 'reference.json', {i: logits[i] for start in selected for i in (start, start+1)})


def decision_gap(logits, target):
    if logits.shape != (4,) or not 0 <= target < 4:
        raise ValueError('requires four logits and a valid target index')
    other = torch.cat((logits[:target], logits[target+1:])).amax()
    return logits[target] - other


class SemanticTrainer(FixedPromptControlTrainer):
    def __init__(self, *args, views, arm, protocol, **kwargs):
        super().__init__(*args, old_records=[], control_arm='plain_gspo', **kwargs)
        self.views, self.arm, self.protocol = views, arm, protocol
        self.candidate_ids = choice_token_ids(self.pact_processor.tokenizer)
        self.controller = TaskDualController(budget=protocol['flip_budget'],
                                             ema_decay=protocol['ema_decay'],
                                             learning_rate=protocol['dual_rate'],
                                             weight_cap=protocol['dual_cap'])
        self.controller.add_task('old')

    def _choice(self, model, record):
        item, target, reference = record
        scores = choice_logits(model, self.pact_processor, item, self.accelerator.device, self.candidate_ids)
        return scores, target, reference

    def _replay(self, model, record):
        scores, target, reference = self._choice(model, record)
        if self.arm == 'semantic_margin':
            return F.relu(self.protocol['margin'] - decision_gap(scores, target))
        if self.arm == 'answer_ce':
            return -scores.log_softmax(0)[target]
        if self.arm == 'answer_kl':
            teacher = scores.new_tensor(reference).softmax(0)
            return F.kl_div(scores.log_softmax(0), teacher, reduction='sum')
        raise ValueError('unknown replay arm')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        step = self.state.global_step
        rank = self.accelerator.process_index
        if self.replay_cache is None or self.replay_cache[0] != step:
            if step % self.protocol['audit_interval'] == 0:
                rng = random.Random(150000 + step * 2 + rank)
                observed = torch.zeros(2, device=self.accelerator.device)
                training = model.training
                model.eval()
                try:
                    with torch.no_grad():
                        for _ in range(self.protocol['audit_per_rank']):
                            scores, target, _ = self._choice(model, rng.choice(self.views))
                            observed[0] += scores.argmax() != target
                            observed[1] += decision_gap(scores, target) < self.protocol['margin']
                        dist.all_reduce(observed)
                finally:
                    model.train(training)
                flip, near = (observed / (dist.get_world_size() * self.protocol['audit_per_rank'])).tolist()
                self.controller.observe_uniform_estimates({'old': flip})
                state = self.controller.states['old']
                self.audit_history.append({'step': step, 'flip_rate': flip, 'below_margin_rate': near,
                                           'smoothed_flip_rate': state.ema_deficit, 'weight': state.weight})
            rng = random.Random(100000 + step * 2 + rank)
            self.replay_cache = step, rng.choices(self.views, k=self.protocol['replay_per_rank'])
        result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        coefficient = (self.controller.states['old'].weight if self.arm == 'semantic_margin'
                       else self.protocol['fixed_control_weight'])
        replay = torch.stack([self._replay(model, record) for record in self.replay_cache[1]]).mean()
        if return_outputs:
            loss, output = result
            return loss + coefficient * replay, output
        return result + coefficient * replay


def train(root, out, reference, protocol, arm):
    if arm not in protocol['arms']:
        raise ValueError('arm is outside the frozen protocol')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    set_seed(protocol['seed'])
    cache = json.loads((out / 'private' / 'reference.json').read_text())
    known = known_views(reference)
    views = [(*known[int(index)], logits) for index, logits in cache.items()]
    processor = AutoProcessor.from_pretrained(root / 'models/base', local_files_only=True,
                                             min_pixels=128*28*28, max_pixels=512*28*28)
    processor.tokenizer.padding_side = 'left'
    rows = list(load_split(root / 'data/prepared/MedicalVQA_Pathology', 'train'))
    templates = yaml.safe_load((Path(__file__).parent / 'prompts.yaml').read_text())
    destination = out / os.environ['STAGE_DIR']
    args = control_training_args(destination, protocol['updates'])
    trainer = SemanticTrainer(model=str(reference / 'stage1/model'), args=args, processing_class=processor,
                              reward_funcs=reward, train_dataset=Images(rows, templates),
                              strategy_cycle=StrategyCycle(('', '')), refresh_factory=None,
                              processor=processor, views=views, arm=arm, protocol=protocol)
    trainer.train()
    trainer.save_model(str(destination / 'model'))
    if dist.get_rank() == 0:
        write_json(destination / 'training.json', {'updates': trainer.state.global_step,
                   'views': len(views), 'logs': trainer.state.log_history, 'audits': trainer.audit_history})
    trainer.accelerator.wait_for_everyone()
    dist.destroy_process_group()


if __name__ == '__main__':
    root, out, reference = (Path(os.environ[key]) for key in ('ROOT', 'OUT', 'REFERENCE'))
    protocol = json.loads((out / 'protocol.json').read_text())
    if os.environ['PHASE'] == 'calibrate':
        calibrate(out, reference, protocol)
    else:
        train(root, out, reference, protocol, os.environ['ARM'])

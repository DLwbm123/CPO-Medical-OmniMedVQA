"""TT-GC and its matched LoRA-only control, using a redacted training flow."""
import json
import os
from pathlib import Path
import re

from peft import LoraConfig
import torch
from transformers import AutoProcessor, TrainerCallback
from trl.models import unwrap_model_for_generation
from trl.models.utils import disable_gradient_checkpointing

from .counterfactual import CounterfactualPool, ImageRecord
from .grounding import blind_kl, evidence_weight, grounded_advantages, update_dual
from .pilot_train import PilotTrainer, TrackPilot, TrainingStream, placeholder_reward, training_config
from .tt_data import load_unlabeled
from ..score_retained_checkpoints import write_json


def adapter_target_pattern(cfg):
    language = '|'.join(map(re.escape,cfg['language_projections']))
    merger = r'|visual\.merger\.mlp\.(?:0|2)' if cfg['visual_merger_linears'] else ''
    return rf'.*(?:language_model\.layers\.\d+\..*\.(?:{language}){merger})'


class GCPilotTrainer(PilotTrainer):
    def __init__(self,*args,grounded_spec,negative,pool,enabled,**kwargs):
        super().__init__(*args,**kwargs)
        self.gc = grounded_spec
        self.use_gc = enabled
        self.negative = negative
        self.pool = pool
        self.dual = grounded_spec['m1']['dual_init']
        self._audit_reference = None
        self._reference_verified = False
        if self.ranks != 1:
            raise ValueError('this pilot implementation requires the verified single-GPU LoRA profile')
        if self.ref_model is not None or not hasattr(self.model,'disable_adapter'):
            raise ValueError('the KL reference must use the same model with its adapter disabled')

    def pair(self,item):
        if self.negative=='black':
            return item,None,False
        replacement, fallback = self.pool.select(item)
        return item,replacement,fallback

    def negative_logps(self,model,pair):
        return self._choice_log_probs(model,pair[0],black=self.negative=='black',replacement=pair[1])

    @torch.no_grad()
    def _verify_initial_adapter(self):
        if self._reference_verified:
            return
        item = self.train_dataset.items[0]
        with (unwrap_model_for_generation(self.model_wrapped,self.accelerator) as model,
              disable_gradient_checkpointing(self.model,self.args.gradient_checkpointing_kwargs)):
            active = self._choice_log_probs(model,item)
            with model.disable_adapter():
                deployment = self._choice_log_probs(model,item)
        if not torch.equal(active,deployment):
            raise ValueError('initial adapter changed legal option probabilities')
        self._reference_verified = True
        if self.accelerator.is_main_process:
            write_json(self.audit.parent/'initial_reference.json',{
                'choice_log_probabilities_bitwise_equal':True,
                'kl_nats':float(blind_kl(active,deployment))})

    @torch.no_grad()
    def _measure_batch(self,pairs):
        with (unwrap_model_for_generation(self.model_wrapped,self.accelerator) as model,
              disable_gradient_checkpointing(self.model,self.args.gradient_checkpointing_kwargs)):
            real = [self._choice_log_probs(model,p[0]) for p in pairs]
            negative = [self.negative_logps(model,p) for p in pairs]
            with model.disable_adapter():
                reference = [self.negative_logps(model,p).detach() for p in pairs]
        return real,negative,reference

    def _generate_and_score_completions(self,inputs):
        self._verify_initial_adapter()
        batch = super()._generate_and_score_completions(inputs)
        if not self.use_gc:
            return batch
        cfg = self.gc['m1']
        self.current_pairs = [self.pair(self.train_dataset.items[self.train_dataset.index[qid]])
                              for qid,_ in self.prompt_set]
        real,negative,self.current_ref = self._measure_batch(self.current_pairs)
        weights, confidences, rows = [], [], []
        for (qid,letters), pair, a, b in zip(self.prompt_set,self.current_pairs,real,negative,strict=True):
            label = self.all_decisions[qid][0].pseudo_label
            weight = confidence = 0.
            evidence = None
            if label is not None:
                j = letters.index(label)
                evidence,weight = evidence_weight(float(a[j]),float(b[j]),
                                                   cfg['evidence_ratio'],cfg['weight_temperature'])
                confidence = float(a[j].exp())
            weights.append(weight)
            confidences.append(confidence)
            rows.append({'id':qid,'evidence':evidence,'weight':weight,'confidence':confidence,
                         'm1_skip':weight<cfg['min_weight'],'m2_skip':confidence>=self.gc['m2']['confidence_cap'],
                         'counterfactual_id':pair[1].id if pair[1] else None,'source_fallback':pair[2]})
        empty = grounded_advantages(batch,weights,confidences,self.num_generations,
                                     cfg['min_weight'],self.gc['m2']['confidence_cap'])
        self.skip_update = self.skip_update or empty
        self.train_kl = []
        self.gc_row = {'step':self.state.global_step+1,'questions':rows,'lambda_b':self.dual,
                       'applied_optimizer_expected':not self.skip_update,
                       'blind_kl_before_update':sum(float(blind_kl(x,y)) for x,y in
                                                    zip(negative,self.current_ref,strict=True))/len(rows)}
        if self.state.global_step==0:
            if any(not torch.equal(x,y) for x,y in zip(negative,self.current_ref,strict=True)):
                raise ValueError('initial blind-input KL reference is not bitwise identical')
        return batch

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        loss = super().compute_loss(model,inputs,return_outputs,num_items_in_batch)
        if not self.use_gc:
            return loss
        # GRPO already divides its loss by accumulation. Four microsteps each add
        # one quarter of the auxiliary objective without re-scaling by accumulation.
        index = self._step % self.args.gradient_accumulation_steps
        if index < len(self.current_pairs):
            current = self.negative_logps(model,self.current_pairs[index])
            term = blind_kl(current,self.current_ref[index])
            self.train_kl.append(float(term.detach()))
            loss = loss + self.dual*term/len(self.current_pairs)
        return loss

    @torch.no_grad()
    def audit_kl(self):
        cfg = self.gc['m1']
        pairs = [self.pair(item) for item in self.train_dataset.items[:cfg['audit_questions']]]
        with (unwrap_model_for_generation(self.model_wrapped,self.accelerator) as model,
              disable_gradient_checkpointing(self.model,self.args.gradient_checkpointing_kwargs)):
            if self._audit_reference is None:
                with model.disable_adapter():
                    self._audit_reference = [self.negative_logps(model,pair).detach() for pair in pairs]
            current = [self.negative_logps(model,pair) for pair in pairs]
        return sum(float(blind_kl(x,y)) for x,y in zip(current,self._audit_reference,strict=True))/len(pairs)


class GroundingAudit(TrainerCallback):
    def __init__(self,trainer,out):
        self.trainer, self.path = trainer,out/'audit.jsonl'

    def on_step_end(self,args,state,control,**kwargs):
        t = self.trainer
        cfg = t.gc['m1']
        row = t.gc_row
        row['training_blind_kl'] = sum(t.train_kl)/len(t.train_kl) if t.train_kl else None
        if state.global_step % cfg['audit_every_steps']==0:
            estimate = t.audit_kl()
            row['audited_blind_kl'] = estimate
            t.dual = update_dual(t.dual,estimate,cfg)
            row['next_lambda_b'] = t.dual
        with self.path.open('a') as stream:
            stream.write(json.dumps(row)+'\n')


def main():
    out = Path(os.environ['OUT'])
    frozen = json.loads((out/'protocol.json').read_text())
    if frozen['stage'] not in ('D','C_gate'):
        raise ValueError('unfrozen TT-GC pilot stage')
    dev = frozen if frozen['stage']=='D' else json.loads((out/frozen['development_protocol']).read_text())
    if os.environ.get('WORLD_SIZE','1') != '1' or ',' in os.environ['CUDA_VISIBLE_DEVICES']:
        raise ValueError('TT-GC pilot uses one assigned GPU per arm')
    arm = os.environ['TT_ARM']
    if arm not in frozen['training_arms']:
        raise ValueError('arm is outside the frozen pilot schedule')
    p0 = json.loads(Path(os.environ['TT_P0_RESULT']).read_text())
    if not p0['passed'] or p0['selected_negative'] not in ('counterfactual','black'):
        raise ValueError('TT-GC pilot requires a passing P0')
    pilot = json.loads((out/dev['parent_protocol']).read_text())
    items = load_unlabeled(Path(os.environ['TT_REDACTED_TASK']),'train',dev['seed'])
    allowed = json.loads(Path(os.environ['TT_COUNTERFACTUAL_MANIFEST']).read_text())
    records = [ImageRecord(x['id'],Path(x['image']),x['source_dataset']) for x in allowed]
    forbidden = json.loads(Path(os.environ['TT_FORBIDDEN_IMAGES']).read_text())
    pool = CounterfactualPool(records,forbidden,dev['seed'])
    n = dev['updates']*dev['prompts_per_update']
    if len(items) < n:
        raise ValueError('redacted flow is shorter than the frozen pilot')
    checkpoint = Path(os.environ['TT_CHECKPOINT']).resolve(strict=True)
    torch.cuda.set_device(0)
    processor = AutoProcessor.from_pretrained(checkpoint,local_files_only=True,
                                              min_pixels=128*28*28,max_pixels=512*28*28)
    processor.tokenizer.padding_side = 'left'
    pilot.update(gpu_count=1,seed=dev['seed'])
    pilot['arms'] = {arm:{'gate':True,'guard':True,'update':True,'kl_beta':dev['kl_beta']}}
    pilot['features'].update({k:dev[k] for k in ('updates','prompts_per_update','training_rollouts',
        'agreement_band_votes','guard_reference_questions','guard_window_questions',
        'guard_max_tv','guard_max_consecutive_steps')})
    smoke = int(os.environ.get('TT_ACCEPTANCE_STEPS',0))
    if smoke not in (0,1,2,3):
        raise ValueError('real-model acceptance is restricted to three steps')
    args = training_config(out,pilot,arm,smoke or dev['updates'])
    cfg = dev['adapter']
    args.learning_rate = cfg['learning_rate']
    adapter = LoraConfig(r=cfg['rank'],lora_alpha=cfg['alpha'],lora_dropout=cfg['dropout'],
                         bias=cfg['bias'],target_modules=adapter_target_pattern(cfg),task_type='CAUSAL_LM')
    trainer = GCPilotTrainer(model=str(checkpoint),args=args,processing_class=processor,
                             reward_funcs=placeholder_reward,train_dataset=TrainingStream(items[:n]),
                             run_spec=pilot,arm=arm,audit=out/'votes.jsonl',peft_config=adapter,
                             grounded_spec=dev,negative=p0['selected_negative'],pool=pool,enabled=arm=='tt_gc')
    names = [name for name,p in trainer.model.named_parameters() if p.requires_grad]
    if (not names or any('lora_' not in name for name in names)
            or not any('visual.merger' in name for name in names)
            or not any('language_model' in name for name in names)):
        raise ValueError('TT-GC must update LoRA weights in language projections and the visual merger only')
    clock = json.loads((out/'clock.json').read_text())
    trainer.add_callback(TrackPilot(trainer,out,clock,pilot))
    if arm=='tt_gc':
        trainer.add_callback(GroundingAudit(trainer,out))
    trainer.train()
    if not smoke:
        trainer.save_model(str(out/'model'))
    write_json(out/'training_result.json',{'steps':trainer.state.global_step,'applied':trainer.updates_applied,
          'brake_stop':trainer.brake_stop,'training_labels':0,'rank':cfg['rank'],
          'first_reference_verified':trainer._reference_verified,'acceptance_steps':smoke})


if __name__ == '__main__':
    main()

"""Stage B GRPO/GSPO using generated votes as its only reward."""
from collections import Counter
import json
import os
from pathlib import Path
import re
import time

from PIL import Image
import torch
from transformers import AutoProcessor, TrainerCallback
from trl import GRPOConfig, GRPOTrainer
from trl.data_utils import prepare_multimodal_messages
from trl.models import unwrap_model_for_generation
from trl.models.utils import disable_gradient_checkpointing
import yaml

from .choices import extract_letter, letter_tokens
from .guard import Guard, mask_image_gate
from .tt_data import load_unlabeled
from .votes import gather_votes, training_indices, vote_decision
from ..score_retained_checkpoints import write_json


def describe(item, user, system):
    question = re.sub(r'<image(?:[_ ]?\d+)?>', '', item.question).strip()
    return [{'role':'system','content':system}, {'role':'user','content':user+'\n'+question}]


def read_image(item):
    with Image.open(item.image) as source:
        return source.convert('RGB')


class TrainingStream(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items
        self.index = {item.id:i for i,item in enumerate(items)}
        directory = Path(__file__).parent
        self.user = yaml.safe_load((directory/'prompts.yaml').read_text())['tt_reason_user']
        self.system = yaml.safe_load((directory.parent/'prompts.yaml').read_text())['system']

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        return {'prompt': describe(item,self.user,self.system), 'image':read_image(item),
                'tt_id':item.id, 'tt_letters':item.letters}


def placeholder_reward(**kwargs):
    raise RuntimeError('stage B uses only the sixteen-vote reward')


class PilotTrainer(GRPOTrainer):
    def __init__(self, *args, run_spec, arm, audit, **kwargs):
        super().__init__(*args, **kwargs)
        self.spec = run_spec['features']
        self.arm = run_spec['arms'][arm]
        self.acceptance = bool(run_spec.get('acceptance_updates'))
        self.audit = audit
        self.guard = None
        self.brake = False
        self.skip_update = False
        self.brake_stop = False
        self.prepared = None
        self.updates_applied = 0
        self.ranks = self.accelerator.num_processes
        if self.ranks != run_spec['gpu_count'] or self.args.steps_per_generation != self.args.gradient_accumulation_steps:
            raise ValueError('pilot requires the frozen GPU count and a complete step per generation batch')

    def _choice_probabilities(self, model, item, *, black=False):
        processor = self.processing_class
        dataset = self.train_dataset
        prefix = r'\boxed{'
        image = read_image(item)
        if black:
            image = Image.new('RGB',image.size)
        messages = describe(item,dataset.user,dataset.system)
        messages[-1]['content'] = [{'type':'image','image':image},
                                   {'type':'text','text':messages[-1]['content']}]
        prompt = processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        ids, choices = letter_tokens(processor.tokenizer,item.letters,prefix)
        tensors = processor(text=[prompt+prefix],images=[image],return_tensors='pt').to(self.accelerator.device)
        if tensors['input_ids'][0,-len(ids):].tolist() != ids:
            raise ValueError('forced-choice prefix tokenization differs at the prompt boundary')
        logits = model(**tensors,logits_to_keep=1,use_cache=False).logits[0,-1,choices].float()
        return logits.softmax(0)

    @torch.no_grad()
    def _guard_reference(self):
        counts = Counter()
        with (
            unwrap_model_for_generation(self.model_wrapped,self.accelerator) as model,
            disable_gradient_checkpointing(self.model,self.args.gradient_checkpointing_kwargs),
        ):
            for item in self.train_dataset.items[:self.spec['guard_reference_questions']]:
                counts[item.letters[int(self._choice_probabilities(model,item).argmax())]] += 1
        return counts

    @torch.no_grad()
    def _blind_predictions(self):
        dataset = self.train_dataset
        local = []
        with (
            unwrap_model_for_generation(self.model_wrapped,self.accelerator) as model,
            disable_gradient_checkpointing(self.model,self.args.gradient_checkpointing_kwargs),
        ):
            for qid,_ in self.prompt_set:
                item = dataset.items[dataset.index[qid]]
                options = self._choice_probabilities(model,item,black=True).tolist()
                if not 0.999 <= sum(options) <= 1.001:
                    raise ValueError('invalid black-image legal-option probabilities')
                local.append((qid,dict(zip(item.letters,options,strict=True))))
        return dict(gather_votes(local))

    def _generate_and_score_completions(self, inputs):
        width = self.num_generations
        prompts = self.spec['prompts_per_update']
        if len(inputs)*self.ranks != width*prompts:
            raise ValueError('the generation batch must contain four groups across ranks')
        groups = [inputs[k:k+width] for k in range(0,len(inputs),width)]
        self.prompt_set = []
        for group in groups:
            keys = {(row['tt_id'],row['tt_letters']) for row in group}
            if len(keys) != 1:
                raise ValueError('mixed questions in one GSPO group')
            self.prompt_set.append(keys.pop())
        self.all_prompts = gather_votes(self.prompt_set)
        if len({qid for qid,_ in self.all_prompts}) != prompts:
            raise ValueError('expected four different questions per attempted update')
        if self.arm['guard'] and self.guard is None:
            reference = self._guard_reference()
            if any(counts != reference for counts in gather_votes([reference])):
                raise ValueError('guard reference differs across ranks')
            self.guard = Guard(reference,window=self.spec['guard_window_questions'],
                               limit=self.spec['guard_max_tv'],
                               consecutive_limit=self.spec['guard_max_consecutive_steps'])
        threshold = self.arm.get('image_gate_threshold')
        blind = self._blind_predictions() if threshold is not None else {}
        result = super()._generate_and_score_completions(inputs)
        before = (result['advantages'] != 0).sum()
        tv, self.brake, self.brake_stop = (
            self.guard.observe_update([d.counts for d,_ in self.all_decisions.values()])
            if self.guard else (None,False,False))
        step = self.state.global_step + 1
        if self.acceptance:
            self.brake, self.brake_stop = step == 3, False
        if self.arm['gate'] and not (self.acceptance and step == 1):
            flags = torch.tensor([not x.in_band for x in self.decisions],device=self.accelerator.device)
            result['advantages'].masked_fill_(flags.repeat_interleave(width),0)
        blind_labels = {qid:blind[qid].get(d.pseudo_label)
                        for qid,(d,_) in self.all_decisions.items()} if blind else {}
        omitted = {qid for qid,p in blind_labels.items() if p is not None and p >= threshold}
        if blind:
            mask_image_gate(result,[qid for qid,_ in self.prompt_set],omitted,width)
        self.skip_update = self.brake or len(omitted) == prompts
        if self.brake or (self.acceptance and step == 2):
            result['advantages'].zero_()
        global_counts = self.accelerator.reduce(torch.stack((before,(result['advantages'] != 0).sum())),
                                                reduction='sum').tolist()
        row = {'step':step,'tv':tv,'brake':self.brake,'brake_stop':self.brake_stop,
               'nonzero_advantages_before':global_counts[0],'nonzero_advantages_after':global_counts[1],
               'image_gate_skips':len(omitted),'skip_update':self.skip_update,
               'questions':[{'id':qid,'counts':d.counts,'pseudo_label':d.pseudo_label,
                             'peak':d.peak,'entropy':d.entropy,'in_band':d.in_band,
                             'blind_vote_probability':blind_labels.get(qid),'image_gated':qid in omitted,
                             'training_indices':list(selection)}
                            for qid,(d,selection) in self.all_decisions.items()]}
        if self.accelerator.is_main_process:
            with self.audit.open('a') as stream:
                stream.write(json.dumps(row)+'\n')
        return result

    def _generate_single_turn(self, prompts):
        prompts_kept, responses_kept = [], []
        self.decisions, self.all_decisions = [], {}
        width = self.num_generations
        tokenizer = self.processing_class.tokenizer
        local_groups = {qid:prompts[i*width:(i+1)*width] for i,(qid,_) in enumerate(self.prompt_set)}
        for qid, letters in self.all_prompts:
            if qid in local_groups:
                group = local_groups[qid]
            else:
                sample = self.train_dataset[self.train_dataset.index[qid]]
                prompt = prepare_multimodal_messages(sample['prompt'],[sample['image']])
                group = [prompt]*width
            prompt_ids, response_ids = [], []
            for _ in range(2//self.ranks):
                p, c, logps, extra = super()._generate_single_turn(group)
                if len(p) != width or len(c) != width or logps is not None or extra:
                    raise ValueError('unexpected generation path for pilot vote group')
                prompt_ids.extend(p)
                response_ids.extend(c)
            answers = [extract_letter(t,letters) for t in tokenizer.batch_decode(response_ids,skip_special_tokens=True)]
            rank_batches = gather_votes([(prompt_ids[0],list(zip(answers,response_ids,strict=True)))])
            if any(ids != prompt_ids[0] for ids in prompt_ids) or any(ids != prompt_ids[0] for ids,_ in rank_batches):
                raise ValueError('vision prompt tokenization differs across rollouts')
            ranked = [rollout for _,batch in rank_batches for rollout in batch]
            decision = vote_decision([a for a,_ in ranked],letters,self.spec['agreement_band_votes'])
            selection = training_indices(self.args.seed,self.state.global_step,qid)
            self.all_decisions[qid] = (decision,selection)
            if qid in local_groups:
                self.decisions.append(decision)
                prompts_kept.extend([prompt_ids[0]]*len(selection))
                responses_kept.extend(ranked[j][1] for j in selection)
        return prompts_kept,responses_kept,None,{}

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        decoded = self.processing_class.tokenizer.batch_decode(completion_ids_list,skip_special_tokens=True)
        scores = []
        for i, ((_,letters), decision) in enumerate(zip(self.prompt_set,self.decisions,strict=True)):
            group = decoded[i*self.num_generations:(i+1)*self.num_generations]
            scores.extend(decision.rewards([extract_letter(t,letters) for t in group]))
        if len(scores) != len(inputs):
            raise ValueError('reward count differs from the sampled training subset')
        local = torch.tensor(scores,dtype=torch.float32,device=self.accelerator.device).unsqueeze(-1)
        return self.accelerator.gather(local)

    def _prepare_inputs(self, inputs):
        return self.prepared if self.prepared is not None else super()._prepare_inputs(inputs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        model.train()
        self.prepared = super()._prepare_inputs(inputs)
        try:
            if self.skip_update:
                # With the DeepSpeed accelerator, skipping backward also skips the optimizer boundary.
                self._step += 1
                return torch.zeros((),device=self.accelerator.device)
            loss = super().training_step(model,inputs,num_items_in_batch)
            if self._step % self.args.gradient_accumulation_steps == 0:
                self.updates_applied += 1
            return loss
        finally:
            self.prepared = None


class TrackPilot(TrainerCallback):
    def __init__(self, trainer, out, clock, spec):
        self.trainer, self.out, self.clock, self.spec = trainer,out,clock,spec
        self.rows = []

    def on_step_begin(self,args,state,control,**kwargs):
        self.started = time.monotonic()
        self.previous_engine_steps = self.trainer.model_wrapped.global_steps

    def on_step_end(self,args,state,control,**kwargs):
        acc = self.trainer.accelerator
        local = torch.tensor([[torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved()]],
                             device=acc.device)
        allocated, reserved = acc.gather(local).T.tolist()
        row = {'step':state.global_step,'seconds':time.monotonic()-self.started,
               'peak_allocated_bytes':max(allocated),'peak_reserved_bytes':max(reserved),
               'per_rank_peak_allocated_bytes':allocated,'per_rank_peak_reserved_bytes':reserved,
               'updates_applied':self.trainer.updates_applied,'brake':self.trainer.brake,
               'skip_update':self.trainer.skip_update,
               'optimizer_steps':self.trainer.model_wrapped.global_steps-self.previous_engine_steps}
        self.rows.append(row)
        if state.global_step == 20:
            seconds = sum(x['seconds'] for x in self.rows[10:])/10
            row['projected_wall_seconds'] = time.time()-self.clock['started'] + (args.max_steps-20)*seconds
        if acc.is_main_process:
            write_json(self.out/'timing.json',self.rows)
        if row['optimizer_steps'] != int(not row['skip_update']):
            raise RuntimeError('optimizer step count contradicts the exclusion signal')
        if row.get('projected_wall_seconds',0) > self.spec['projected_wall_limit_seconds']:
            raise RuntimeError('pilot forecast exceeds its frozen time limit')
        if self.trainer.brake_stop:
            control.should_training_stop = True
        return control


def training_config(out,spec,arm,updates):
    ds = {'bf16':{'enabled':True},'train_micro_batch_size_per_gpu':'auto',
          'gradient_accumulation_steps':'auto','train_batch_size':'auto','gradient_clipping':'auto',
          'zero_optimization':{'stage':3,'overlap_comm':False,'contiguous_gradients':True,
                               'reduce_bucket_size':10_000_000,'stage3_prefetch_bucket_size':10_000_000,
                               'stage3_param_persistence_threshold':100_000,'sub_group_size':100_000_000,
                               'stage3_gather_16bit_weights_on_model_save':True}}
    feat, sampling = spec['features'],spec['sampling']
    prompts, subset = feat['prompts_per_update'],feat['training_rollouts']
    microbatch = 2
    accumulation = prompts*subset//(microbatch*spec['gpu_count'])
    return GRPOConfig(output_dir=str(out), max_steps=updates,
                      per_device_train_batch_size=microbatch, gradient_accumulation_steps=accumulation,
                      steps_per_generation=accumulation, num_generations=subset, num_iterations=1,
                      learning_rate=5e-6, bf16=True, full_determinism=True, gradient_checkpointing=True,
                      gradient_checkpointing_kwargs={'use_reentrant':False},
                      max_completion_length=sampling['max_new_tokens'], temperature=sampling['temperature'],
                      top_k=sampling['top_k'], top_p=sampling['top_p'], min_p=sampling['min_p'],
                      generation_kwargs={'use_cache':sampling['use_cache'],'typical_p':sampling['typical_p']},
                      repetition_penalty=sampling['repetition_penalty'], importance_sampling_level='sequence',
                      loss_type='grpo', scale_rewards='group', beta=spec['arms'][arm]['kl_beta'],
                      seed=spec['seed'], data_seed=spec['seed'], shuffle_dataset=False,
                      deepspeed=ds, report_to='none', logging_steps=1, save_strategy='no',
                      remove_unused_columns=False, dataloader_num_workers=0,
                      model_init_kwargs={'torch_dtype':'bfloat16','attn_implementation':'sdpa','local_files_only':True})


def main():
    out = Path(os.environ['OUT'])
    spec = json.loads((out/'protocol.json').read_text())
    arm = os.environ['TT_ARM']
    if spec['stage'] not in ('B','B_replication') or spec['prompt'] != 'reason' or not spec['arms'].get(arm,{}).get('update'):
        raise ValueError('unknown pilot training setup')
    train_dir = Path(os.environ['TT_REDACTED_TASK'])
    items = load_unlabeled(train_dir,'train',spec['seed'])
    features = spec['features']
    n = features['updates'] * features['prompts_per_update']
    if len(items) < n:
        raise ValueError('the redacted training stream is shorter than the frozen pilot')
    smoke = int(spec.get('acceptance_updates',0))
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    checkpoint = Path(os.environ['TT_CHECKPOINT']).resolve(strict=True)
    processor = AutoProcessor.from_pretrained(checkpoint,local_files_only=True,
                                              min_pixels=128*28*28,max_pixels=512*28*28)
    processor.tokenizer.padding_side = 'left'
    args = training_config(out,spec,arm,smoke or features['updates'])
    trainer = PilotTrainer(model=str(checkpoint),args=args,processing_class=processor,
                           reward_funcs=placeholder_reward,train_dataset=TrainingStream(items[:n]),
                           run_spec=spec,arm=arm,audit=out/'votes.jsonl')
    if spec['trainable_scope'] == 'language':
        trainer.model.visual.requires_grad_(False)
    elif spec['trainable_scope'] != 'full':
        raise ValueError('unsupported pilot trainable scope')
    if any(p.requires_grad for p in trainer.model.visual.parameters()) != (spec['trainable_scope']=='full'):
        raise RuntimeError('vision encoder trainability disagrees with the pilot protocol')
    if trainer.beta:
        ref = trainer.accelerator.unwrap_model(trainer.ref_model)
        if Path(ref.config._name_or_path).resolve() != checkpoint:
            raise ValueError('KL reference must match the pilot boundary model')
    clock = json.loads((out/'clock.json').read_text())
    trainer.add_callback(TrackPilot(trainer,out,clock,spec))
    trainer.train()
    if not smoke:
        trainer.save_model(str(out/'model'))
    if trainer.accelerator.is_main_process:
        write_json(out/'training_result.json',{'steps':trainer.state.global_step,'applied':trainer.updates_applied,
                   'brake_stop':trainer.brake_stop,'training_labels':0,'smoke':bool(smoke),
                   'reference_matches':bool(trainer.beta),'trainable_scope':spec['trainable_scope'],
                   'gpu_count':spec['gpu_count']})
    trainer.accelerator.wait_for_everyone()


if __name__ == '__main__':
    main()

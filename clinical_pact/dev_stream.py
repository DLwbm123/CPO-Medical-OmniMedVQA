"""Two-stage real-model engineering run. Fixed seed cards; not PACT-full efficacy."""
import os, json, random, re, gc, time
from pathlib import Path
from collections import Counter
import torch
import torch.distributed as dist
import yaml
from PIL import Image
from jinja2 import Environment, StrictUndefined
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from trl import GRPOConfig
from trl.models.utils import unwrap_model_for_generation
from .fast_rollout import FirstStageRolloutTrainer
from .data import load_split
from .core import TaskDualController
from .decision_loss import replay_deficit
from .multimodal import make_inputs, greedy_reference
from .scoring import score_response


class Images(torch.utils.data.Dataset):
    def __init__(self, rows, templates): self.rows, self.templates = rows, templates
    def __len__(self): return len(self.rows)
    def __getitem__(self, index):
        row = self.rows[index]
        question = re.sub(r'<image(?:[_ ]?\d+)?>', '', row['question']).strip()
        prompt = Environment(undefined=StrictUndefined).from_string(self.templates['canonical_user']).render(question=question, strategy='')
        with Image.open(row['image']) as image: pixels = image.convert('RGB')
        return {'prompt': [{'role':'system','content':self.templates['system']},{'role':'user','content':prompt}], 'images':[pixels], 'answer':row['answer'], 'pact_question_id':row['id']}


def reward(completions, answer, **kwargs):
    return [score_response(c[0]['content'] if isinstance(c, list) else c, a) for c,a in zip(completions,answer)]


class EngineeringTrainer(FirstStageRolloutTrainer):
    def __init__(self,*args,old_records,processor,**kwargs):
        self.old_records=old_records; self.pact_processor=processor; self.controller=TaskDualController(); self.replay_cache=None
        if old_records: self.controller.add_task('old')
        super().__init__(*args,**kwargs)

    def create_optimizer(self):
        optimizer=super().create_optimizer()
        for group in self.optimizer.param_groups:
            group['foreach']=False
        self.optimizer.defaults['foreach']=False
        return optimizer

    def replay_inputs(self, record):
        item, reference=record
        x=make_inputs(self.pact_processor,item,device=self.accelerator.device)
        start=x['input_ids'].shape[1]
        response=torch.tensor([reference['response_ids']],dtype=torch.long,device=self.accelerator.device)
        x['input_ids']=torch.cat([x['input_ids'],response],dim=1)
        x['attention_mask']=torch.ones_like(x['input_ids'])
        tau=torch.tensor([[0.5*min(m,2.) for m in reference['margins']]],device=self.accelerator.device)
        return x,start,reference['eos_id'],tau

    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        step=self.state.global_step
        if self.old_records and (self.replay_cache is None or self.replay_cache[0]!=step):
            rng=random.Random(100000+step*4+self.accelerator.process_index)
            # Engineering audit every optimizer step, 8 uniform samples/rank.
            audit=torch.zeros((),device=self.accelerator.device)
            model.eval()
            with torch.no_grad():
                for _ in range(8):
                    x,start,eos,tau=self.replay_inputs(self.old_records[rng.randrange(len(self.old_records))])
                    audit+=replay_deficit(model,x,start,eos,tau)/8
            dist.all_reduce(audit);audit/=dist.get_world_size()
            self.controller.observe_uniform_estimates({'old':float(audit)})
            model.train()
            self.replay_cache=(step,self.old_records[rng.randrange(len(self.old_records))])
        loss=super().compute_loss(model,inputs,return_outputs,num_items_in_batch)
        if not self.old_records:return loss
        x,start,eos,tau=self.replay_inputs(self.replay_cache[1])
        return loss+self.controller.corrected_replay_loss('old',replay_deficit(model,x,start,eos,tau),1.)


def save_json(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));os.replace(temp,path)


def main():
    root=Path(os.environ['ROOT']); out=Path(os.environ['OUT']); rank=int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    templates=yaml.safe_load((Path(__file__).parent/'prompts.yaml').read_text())
    processor=AutoProcessor.from_pretrained(root/'models/base',local_files_only=True,min_pixels=128*28*28,max_pixels=512*28*28)
    processor.tokenizer.padding_side='left'
    records=[]; source=root/'models/base'
    ds={'bf16':{'enabled':True},'train_micro_batch_size_per_gpu':'auto','gradient_accumulation_steps':'auto','train_batch_size':'auto','zero_optimization':{'stage':3,'overlap_comm':False,'contiguous_gradients':True,'reduce_bucket_size':10000000,'stage3_prefetch_bucket_size':10000000,'stage3_param_persistence_threshold':100000,'stage3_gather_16bit_weights_on_model_save':True},'gradient_clipping':'auto'}
    for stage,task in enumerate(('MedicalVQA_MedBook','MedicalVQA_Pathology')):
        rows=list(load_split(root/'data/prepared'/task,'train'))
        args=GRPOConfig(output_dir=str(out/f'stage{stage+1}'),max_steps=2,per_device_train_batch_size=2,gradient_accumulation_steps=4,steps_per_generation=1,num_generations=8,num_iterations=1,learning_rate=5e-6,bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},max_completion_length=128,importance_sampling_level='sequence',loss_type='grpo',beta=0.,temperature=1.,repetition_penalty=1.,seed=100,data_seed=100,deepspeed=ds,report_to='none',logging_steps=1,save_strategy='no',remove_unused_columns=False,model_init_kwargs={'torch_dtype':'bfloat16','attn_implementation':'sdpa','local_files_only':True},dataloader_num_workers=0)
        trainer=EngineeringTrainer(model=str(source),args=args,processing_class=processor,reward_funcs=reward,train_dataset=Images(rows,templates),card_a=templates['seed_strategy_1'],card_b=templates['seed_strategy_2'],old_records=records,processor=processor)
        if rank==0:save_json(out/'progress.json',{'stage':stage+1,'state':'training','max_updates':2})
        trainer.train()
        trainer.save_model(str(out/f'stage{stage+1}'/'model'))
        if rank==0:
            save_json(out/f'stage{stage+1}_training.json',{'updates':trainer.state.global_step,'logs':trainer.state.log_history,'scope':'engineering only; fixed seed cards; no performance comparison','dual':{k:vars(v) for k,v in trainer.controller.states.items()}})
        trainer.accelerator.wait_for_everyone()
        if stage==0:
            rejected=Counter(); candidates=random.Random(100).sample(rows,min(32,len(rows)))
            with unwrap_model_for_generation(trainer.model_wrapped,trainer.accelerator,gather_deepspeed3_params=True) as model:
                model.eval()
                for item in candidates:
                    reference,reason=greedy_reference(model,processor,item,trainer.accelerator.device)
                    if reference is not None:records.append((item,reference))
                    else:rejected[reason]+=1
            if rank==0:
                save_json(out/'memory_aggregate.json',{'candidates':len(candidates),'eligible':len(records),'rejected':dict(rejected)})
                torch.save(records,out/'private_memory.pt')
            trainer.accelerator.wait_for_everyone()
            if not records:raise RuntimeError('zero eligible memories; stop without substituting gold answers')
        source=out/f'stage{stage+1}'/'model'
        del trainer;gc.collect();torch.cuda.empty_cache()
    if rank==0:save_json(out/'completed.json',{'stages':2,'updates_per_stage':2,'scope':'real-model engineering; no efficacy result; fixed cards, dual audit every step; no efficacy comparison or resume validation'})


if __name__=='__main__':main()

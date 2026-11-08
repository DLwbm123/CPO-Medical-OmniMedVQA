"""Frozen two-GPU two-task development training; stage-boundary recovery only.

Bounded-grammar card variant; no test evaluation or confirmatory efficacy claim.
"""
import gc,json,os,random,time
from pathlib import Path
import torch
import torch.distributed as dist
import yaml
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration,set_seed
from trl import GRPOConfig
from trl.models.utils import unwrap_model_for_generation
from .dev_stream import Images,reward,save_json
from .fast_rollout import FirstStageRolloutTrainer
from .dynamic_trainer import DynamicStrategyTrainer
from .strategy_cycle import StrategyCycle
from .strategy_model import LocalProcedureProposer,PolicyStrategyEvaluator
from .data import load_split
from .core import TaskDualController
from .decision_loss import replay_deficit
from .multimodal import make_inputs,greedy_reference
from .scoring import score_response


class DevelopmentTrainer(DynamicStrategyTrainer):
    def __init__(self,*args,old_records,processor,**kwargs):
        self.old_records=old_records;self.pact_processor=processor;self.controller=TaskDualController();self.replay_cache=None;self.audit_history=[]
        if old_records:self.controller.add_task('old')
        super().__init__(*args,**kwargs)
    def create_optimizer(self):
        result=super().create_optimizer()
        for group in self.optimizer.param_groups:group['foreach']=False
        self.optimizer.defaults['foreach']=False
        return result
    def replay_inputs(self,record):
        item,reference=record;x=make_inputs(self.pact_processor,item,device=self.accelerator.device)
        start=x['input_ids'].shape[1];response=torch.tensor([reference['response_ids']],dtype=torch.long,device=self.accelerator.device)
        x['input_ids']=torch.cat((x['input_ids'],response),dim=1);x['attention_mask']=torch.ones_like(x['input_ids'])
        tau=torch.tensor([[.5*min(m,2.) for m in reference['margins']]],device=self.accelerator.device)
        return x,start,reference['eos_id'],tau
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        step=self.state.global_step
        if self.old_records and (self.replay_cache is None or self.replay_cache[0]!=step):
            rng=random.Random(100000+step*2+self.accelerator.process_index)
            if step%10==0:
                audit=torch.zeros((),device=self.accelerator.device);model.eval()
                with torch.no_grad():
                    for _ in range(16):
                        x,start,eos,tau=self.replay_inputs(self.old_records[rng.randrange(len(self.old_records))])
                        audit+=replay_deficit(model,x,start,eos,tau)/16
                dist.all_reduce(audit);audit/=dist.get_world_size()
                self.controller.observe_uniform_estimates({'old':float(audit)})
                self.audit_history.append({'step':step,'deficit':float(audit),'weight':self.controller.states['old'].weight})
                model.train()
            self.replay_cache=(step,[self.old_records[rng.randrange(len(self.old_records))] for _ in range(2)])
        loss=super().compute_loss(model,inputs,return_outputs,num_items_in_batch)
        if self.old_records:
            for record in self.replay_cache[1]:
                x,start,eos,tau=self.replay_inputs(record)
                loss=loss+self.controller.corrected_replay_loss('old',replay_deficit(model,x,start,eos,tau)/2,1.)
        return loss


def main():
    rank=int(os.environ['RANK']);torch.cuda.set_device(int(os.environ['LOCAL_RANK']));dist.init_process_group('nccl')
    root=Path(os.environ['ROOT']);out=Path(os.environ['OUT']);set_seed(100)
    templates=yaml.safe_load((Path(__file__).parent/'prompts.yaml').read_text())
    processor=AutoProcessor.from_pretrained(root/'models/base',local_files_only=True,min_pixels=128*28*28,max_pixels=512*28*28)
    processor.tokenizer.padding_side='left'
    proposer=None
    if rank==0:
        base=Qwen2_5_VLForConditionalGeneration.from_pretrained(root/'models/base',dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
        for p in base.parameters():p.requires_grad_(False)
        proposer=LocalProcedureProposer(base,processor.tokenizer,'cuda')
    dist.barrier()
    cards=(templates['seed_strategy_1'],templates['seed_strategy_2']);records=[];source=root/'models/base'
    start_stage=int(os.environ.get('START_STAGE','1'))
    if start_stage not in (1,2):raise ValueError('only initial or completed stage1 boundary start is supported')
    if start_stage==2:
        boundary=json.loads((out/'stage1_boundary.json').read_text())
        if boundary['updates']!=200 or boundary['next_stage']!=2 or (out/'stage2_training.json').exists() or (out/'completed.json').exists():
            raise ValueError('expected unfinished stage2 and a completed 200-update stage1 boundary')
        source=out/'stage1/model';cards=tuple(boundary['cards'])
        records=torch.load(out/'private_memory.pt',map_location='cpu',weights_only=False)
        if len(cards)!=2 or not records or not (source/'config.json').is_file():raise ValueError('incomplete stage1 boundary')
    ds={'bf16':{'enabled':True},'train_micro_batch_size_per_gpu':'auto','gradient_accumulation_steps':'auto','train_batch_size':'auto','zero_optimization':{'stage':3,'overlap_comm':False,'contiguous_gradients':True,'reduce_bucket_size':10000000,'stage3_prefetch_bucket_size':10000000,'stage3_param_persistence_threshold':100000,'stage3_gather_16bit_weights_on_model_save':True},'gradient_clipping':'auto'}
    for stage,task in enumerate(('MedicalVQA_MedBook','MedicalVQA_Pathology')):
        if stage+1<start_stage:continue
        rows=list(load_split(root/'data/prepared'/task,'train'))
        audit_rows=random.Random(100+stage).sample(rows,32);audit_ids=tuple(item['id'] for item in audit_rows);refresh_cost=[]
        def factory(policy,update):
            evaluator=PolicyStrategyEvaluator(policy,processor,audit_rows,'cuda',split='train')
            refresh_cost.append({'step':update,'calls':evaluator.cost})
            return dict(split='train',audit_ids=audit_ids,evaluate=evaluator,propose=proposer,count_tokens=lambda text:len(processor.tokenizer.encode(text,add_special_tokens=False)),invalid_proposal='retain')
        cycle=StrategyCycle(cards)
        args=GRPOConfig(output_dir=str(out/f'stage{stage+1}'),max_steps=200,per_device_train_batch_size=4,gradient_accumulation_steps=4,steps_per_generation=1,num_generations=8,num_iterations=1,learning_rate=5e-6,bf16=True,full_determinism=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},max_completion_length=128,importance_sampling_level='sequence',loss_type='grpo',beta=0.,temperature=1.,repetition_penalty=1.,seed=100,data_seed=100,deepspeed=ds,report_to='none',logging_steps=1,save_strategy='no',remove_unused_columns=False,model_init_kwargs={'torch_dtype':'bfloat16','attn_implementation':'sdpa','local_files_only':True},dataloader_num_workers=0)
        trainer=DevelopmentTrainer(model=str(source),args=args,processing_class=processor,reward_funcs=reward,train_dataset=Images(rows,templates),strategy_cycle=cycle,refresh_factory=factory,old_records=records,processor=processor)
        if rank==0:save_json(out/'progress.json',{'stage':stage+1,'state':'training','max_updates':200})
        trainer.train()
        trainer.save_model(str(out/f'stage{stage+1}'/'model'))
        cards=trainer.strategy_cycle.cards
        if rank==0:
            save_json(out/f'stage{stage+1}_training.json',{'updates':trainer.state.global_step,'logs':trainer.state.log_history,'cards':trainer.strategy_cycle.checkpoint(),'audits':trainer.audit_history,'refresh_cost':refresh_cost,'proposer_cost':proposer.cost,'scope':'development only; stage-boundary recovery, no in-stage optimizer resume; bounded procedural grammar variant'})
        trainer.accelerator.wait_for_everyone()
        if stage==0:
            candidates=random.Random(100).sample(rows,min(512,len(rows)));rejected={}
            with unwrap_model_for_generation(trainer.model_wrapped,trainer.accelerator,gather_deepspeed3_params=True) as policy:
                policy.eval()
                for item in candidates:
                    reference,reason=greedy_reference(policy,processor,item,trainer.accelerator.device)
                    if reference is not None:records.append((item,reference))
                    else:rejected[reason]=rejected.get(reason,0)+1
            del policy
            if rank==0:
                torch.save(records,out/'private_memory.pt')
                save_json(out/'memory_aggregate.json',{'candidates':len(candidates),'eligible':len(records),'rejected':rejected})
                save_json(out/'stage1_boundary.json',{'next_stage':2,'next_stage_optimizer':'fresh AdamW, seed100 sampler restarted by frozen protocol','cards':cards,'model':'stage1/model','memory':'private_memory.pt','updates':200})
            trainer.accelerator.wait_for_everyone()
            if not records:raise RuntimeError('no eligible memory; no gold substitution permitted')
        source=out/f'stage{stage+1}'/'model';del trainer;gc.collect();torch.cuda.empty_cache()
    if rank==0:save_json(out/'completed.json',{'stages':2,'updates_per_stage':200,'scope':'development bounded-card variant; no test evaluation or paired efficacy comparison'})
    dist.destroy_process_group()


if __name__=='__main__':main()

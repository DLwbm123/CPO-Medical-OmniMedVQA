"""Full-size Qwen checkpoint storage/recovery engineering profile.

Two response-NLL updates only, not GSPO or a PACT efficacy experiment.
"""
import copy
import gc
import json
import os
from pathlib import Path
import random
import torch
import torch.distributed as dist
import deepspeed
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from .multimodal import make_inputs
from .compressed_checkpoint import make_compressed_checkpoint_engine, save_compressed_state, load_compressed_state


def maximum_error(a,b):
    if isinstance(a,torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:raise RuntimeError('checkpoint tensor layout changed')
        return float((a.detach().float().cpu()-b.detach().float().cpu()).abs().max()) if a.numel() else 0.
    if isinstance(a,dict):
        if a.keys()!=b.keys():raise RuntimeError('checkpoint keys changed')
        return max((maximum_error(a[k],b[k]) for k in a),default=0.)
    if isinstance(a,(tuple,list)):
        if type(a)!=type(b) or len(a)!=len(b):raise RuntimeError('checkpoint sequence changed')
        return max((maximum_error(x,y) for x,y in zip(a,b)),default=0.)
    if a!=b:raise RuntimeError('checkpoint scalar changed')
    return 0.


def optimizer_snapshot(engine):
    # ZeRO-3 can clear the underlying optimizer's param_groups between steps;
    # its raw torch state_dict then lacks a parameter mapping. Compare actual
    # slot states by the known FP32 partition order instead.
    optimizer=engine.optimizer.optimizer
    return {'states':[copy.deepcopy(optimizer.state[p]) for p in engine.optimizer.fp32_partitioned_groups_flat],
            'groups':[{k:copy.deepcopy(v) for k,v in group.items() if k!='params'} for group in optimizer.param_groups]}


def main():
    rank,local=int(os.environ['RANK']),int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    dist.init_process_group('nccl')
    torch.use_deterministic_algorithms(True)
    root=Path(os.environ['OUT'])
    source=Path(os.environ['SOURCE'])
    model_path=source/'stage1/model'
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,min_pixels=128*28*28,max_pixels=512*28*28)
    memories=torch.load(source/'private_memory.pt',map_location='cpu',weights_only=False)
    if len(memories)<4:raise RuntimeError('four fixed training references unavailable')
    item,reference=memories[rank]
    inputs=make_inputs(processor,item,device='cuda')
    response_start=inputs['input_ids'].shape[1]
    response=torch.tensor([reference['response_ids']],device='cuda',dtype=torch.long)
    inputs['input_ids']=torch.cat((inputs['input_ids'],response),dim=1)
    inputs['attention_mask']=torch.ones_like(inputs['input_ids'])
    def progress(value):
        if rank==0:
            (root/'progress.json').write_text(json.dumps({'phase':value}))
    config={'bf16':{'enabled':True},'train_micro_batch_size_per_gpu':1,'gradient_accumulation_steps':1,'train_batch_size':4,
            'zero_optimization':{'stage':3,'overlap_comm':False,'reduce_bucket_size':10000000,'stage3_prefetch_bucket_size':10000000,'stage3_param_persistence_threshold':0},
            'zero_allow_untested_optimizer':True,'steps_per_print':1000}
    def create():
        torch.manual_seed(314)
        model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().train()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        optimizer=torch.optim.AdamW(model.parameters(),lr=5e-6,weight_decay=0,foreach=False)
        engine,_,_,_=deepspeed.initialize(model=model,optimizer=optimizer,config=config,dist_init_required=False)
        engine.checkpoint_engine=make_compressed_checkpoint_engine(byte_limit=11000000000)
        if engine.load_universal_checkpoint():raise RuntimeError('universal checkpoint is not supported by this codec')
        return engine
    def update(engine,step):
        output=engine(**inputs,use_cache=False)
        logits=output.logits[:,response_start-1:-1,:].float()
        labels=inputs['input_ids'][:,response_start:]
        loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1))
        if not torch.isfinite(loss):raise RuntimeError('nonfinite response NLL')
        engine.backward(loss);engine.step()
    engine=create()
    torch.manual_seed(700+rank);random.seed(700+rank)
    progress('first_NLL_update')
    update(engine,0)
    rng={'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),'python':random.getstate()}
    save_compressed_state(rng,root/f'rng_{rank}.pt',byte_limit=1000000)
    progress('writing_compressed_checkpoint')
    engine.save_checkpoint(str(root/'checkpoint'),tag='step1',client_state={'phase':'after-one-update'},save_latest=True)
    progress('uninterrupted_second_NLL_update')
    update(engine,1)
    expected=[x.detach().cpu().clone() for x in engine.optimizer.fp32_partitioned_groups_flat]
    expected_optimizer=optimizer_snapshot(engine)
    expected_steps=engine.global_steps
    del engine;gc.collect();torch.cuda.empty_cache()
    progress('fresh_engine_and_checkpoint_load')
    engine=create()
    path,client=engine.load_checkpoint(str(root/'checkpoint'),tag='step1',load_optimizer_states=True,load_lr_scheduler_states=True)
    if path is None or client.get('phase')!='after-one-update':raise RuntimeError('ZeRO checkpoint failed to load')
    rng=load_compressed_state(root/f'rng_{rank}.pt')
    torch.set_rng_state(rng['torch'].cpu());torch.cuda.set_rng_state(rng['cuda'].cpu());random.setstate(rng['python'])
    update(engine,1)
    error=maximum_error(expected,[x.detach().cpu() for x in engine.optimizer.fp32_partitioned_groups_flat])
    optimizer_error=maximum_error(expected_optimizer,optimizer_snapshot(engine))
    if engine.global_steps!=expected_steps:raise RuntimeError('optimizer step cursor not recovered')
    errors=torch.tensor([error,optimizer_error],device='cuda')
    dist.all_reduce(errors,op=dist.ReduceOp.MAX)
    if rank==0:
        state_files=list((root/'checkpoint').rglob('*.pt'))+list(root.glob('rng_*.pt'))
        compressed=all(file.open('rb').read(2)==b'\x1f\x8b' for file in state_files)
        result={'ranks':4,'weight_max_abs_error':float(errors[0]),'optimizer_state_max_abs_error':float(errors[1]),'restored_updates':engine.global_steps,
                'gzip_state_files':len(state_files),'all_state_files_gzip':compressed,'total_state_bytes':sum(file.stat().st_size for file in state_files),
                'peak_allocated_bytes_rank0':torch.cuda.max_memory_allocated(),'scope':'full-size Qwen4rank ZeRO3 response-NLL2step storage/recovery profile on first4 immutable training references; not GSPO, steady-state compression, full Trainer/PACT extra-state recovery, or efficacy evaluation'}
        (root/'result.json').write_text(json.dumps(result,indent=2))
        if not compressed:raise RuntimeError('some checkpoint writer bypassed compressed engine')
    if bool((errors>1e-6).any()):raise RuntimeError('resume and uninterrupted updates differ beyond 1e-6')
    progress('completed')
    dist.destroy_process_group()


if __name__=='__main__':main()

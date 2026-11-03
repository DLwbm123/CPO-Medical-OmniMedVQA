import json,os,subprocess,time,signal,shutil,sys
from pathlib import Path
root=Path(os.environ['CPO_PROJECT_ROOT']);run=Path(os.environ['CPO_RUN_DIR']);data=Path(os.environ['CPO_DATA_ROOT']);ops=Path(os.environ['CPO_OPS_ROOT']);cfg=ops/'config.json'
tasks=['MedicalVQA_CT','MedicalVQA_MRI','MedicalVQA_XRay','MedicalVQA_Ultrasound','MedicalVQA_Fundus','MedicalVQA_OCT','MedicalVQA_Dermoscopy','MedicalVQA_Microscopy']
base=os.environ['CPO_BASE_MODEL'];py=sys.executable;worker=str(ops/'worker.py')
start=time.monotonic();deadline=start+86100;ceiling=40000000000
state={'status':'running','phase':'initializing','stage':0,'start_unix':time.time(),'elapsed_seconds':0}
def write_state():
 state['elapsed_seconds']=round(time.monotonic()-start,1);tmp=run/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');os.replace(tmp,run/'status.json')
def log(*args):print(*args,flush=True)
def record_path():
 size=int(subprocess.check_output(['du','-sb','.'],cwd=run).split()[0]);state['run_bytes']=size;write_state();return size
def options(d):
 out=[]
 for k,v in d.items():out.extend(['--'+k,str(v)])
 return out
def execute(mode,entry,args,distributed=False,gpus='6,7'):
 cfg.write_text(json.dumps({'mode':mode,'entry':entry,'args':args}));env=os.environ.copy();env['CPO_CONTROL_FILE']=str(cfg)
 env.update({'CUDA_VISIBLE_DEVICES':gpus,'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','HF_HOME':str(run/'cache/hf'),'XDG_CACHE_HOME':str(run/'cache/xdg'),'TRITON_CACHE_DIR':str(run/'cache/triton'),'TORCHINDUCTOR_CACHE_DIR':str(run/'cache/inductor'),'TMPDIR':str(run/'tmp'),'TOKENIZERS_PARALLELISM':'false','WANDB_DISABLED':'true','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True','OMP_NUM_THREADS':'4'})
 (run/'tmp').mkdir(exist_ok=True)
 cmd=[py,'-m','torch.distributed.run','--nproc_per_node=2','--master_port=29683',worker] if distributed else [py,worker]
 child=subprocess.Popen(cmd,cwd=root,env=env,start_new_session=True)
 reason=None
 while child.poll() is None:
  if time.monotonic()>=deadline:reason='time_budget';break
  try:
   if record_path()>=ceiling:reason='storage_guard';break
  except Exception as e:reason='storage_audit_failed_'+type(e).__name__;break
  time.sleep(5)
 if reason:
  log('GUARD_STOP',reason);os.killpg(child.pid,signal.SIGTERM)
  try:child.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
  raise RuntimeError(reason)
 if child.returncode!=0:raise RuntimeError(mode+'_exit_'+str(child.returncode))
 record_path()
def gpu_ready():
 lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True).splitlines()
 values={int(a.strip()):int(b.strip()) for a,b in (x.split(',') for x in lines)}
 if min(values[6],values[7])<65000:raise RuntimeError('insufficient_available_gpu_memory')
def dispose(p):
 if p.exists():
  if not p.is_dir() or p.is_symlink() or p.parent!=run/'training':raise ValueError('invalid cleanup target')
  shutil.rmtree(p);log('removed_transient_checkpoint',p.name)
try:
 log('RUN_STARTED',time.time());write_state()
 for idx,task in enumerate(tasks,1):
  state.update(stage=idx,task=task,phase='gpu_check');write_state();gpu_ready()
  model=base if idx==1 else str(run/'training'/tasks[idx-2]);out=run/'training'/task
  args={'model_id':model,'data_path':str(data/task/'jsons/train/data.json'),'image_folder':str(data/task/'images'),'prompt_path':'src/dataset/prompts_2.yaml','output_dir':str(out),'remove_unused_columns':'False','freeze_vision_tower':'False','freeze_llm':'False','freeze_merger':'False','bf16':'True','fp16':'False','disable_flash_attn2':'True','max_steps':'600','num_generations':'8','per_device_train_batch_size':'4','gradient_accumulation_steps':'4','max_completion_length':'2048','image_min_pixels':str(128*32*32),'image_max_pixels':str(512*32*32),'learning_rate':'5e-6','weight_decay':'0.1','warmup_ratio':'0.1','lr_scheduler_type':'cosine','logging_steps':'1','gradient_checkpointing':'True','ddp_find_unused_parameters':'False','report_to':'none','save_strategy':'no','dataloader_num_workers':'0','beta':'0.0','temperature':'0.8','top_p':'1.0','epsilon':'0.0003','epsilon_high':'0.0004','importance_sampling_level':'sequence','loss_type':'grpo','steps_per_generation':'1','use_liger_kernel':'False','seed':'42'}
  if idx>1:args.update(mask_path=str(run/'masks'/f'task_{idx-1}.pt'),mask_lambda='100')
  state['phase']='training';write_state();log('STAGE_TRAIN',idx,task);execute('train','src/train/train_grpo.py',options(args),True)
  if not (out/'config.json').is_file() or not list(out.glob('*.safetensors')):raise RuntimeError('checkpoint_incomplete')
  for eval_task in tasks[:idx]:
   state.update(phase='evaluation',evaluation_task=eval_task);write_state();log('STAGE_EVAL',task,eval_task)
   target=run/'results'/task/eval_task
   args={'model':str(out),'data':str(data/eval_task/'jsons/test/data.json'),'images':str(data/eval_task/'images'),'output':str(target),'batch-size':'4','max-tokens':'2048'}
   execute('eval','medical/eval_torch.py',options(args),False,'6')
   if not (target/'metrics.json').is_file():raise RuntimeError('evaluation_incomplete')
  state.pop('evaluation_task',None)
  if idx<len(tasks):
   state['phase']='mask';write_state();log('STAGE_MASK',idx,task)
   args=['--base_model',base,'--task_id',str(idx),'--datasets',*tasks,'--top_percent','10.0','--checkpoint_dir',str(run)]
   execute('mask','src/compute_importance_mask.py',args,False,'')
   if not (run/'masks'/f'task_{idx}.pt').is_file():raise RuntimeError('mask_incomplete')
  state['phase']='stage_complete';write_state();log('STAGE_COMPLETE',idx,task)
  if idx>=3:
   dispose(run/'training'/tasks[idx-2])
   old=run/'masks'/f'task_{idx-1}.pt'
   if idx<len(tasks) and old.exists() and old.parent==run/'masks' and idx-1>1:old.unlink();log('removed_transient_mask',old.name)
  record_path()
 state['status']='complete';state['phase']='complete';write_state();log('RUN_COMPLETE')
except BaseException as e:
 state.update(status='stopped',reason=type(e).__name__+': '+str(e));write_state();log('RUN_STOPPED',state['reason']);raise

import os,sys,time,json,signal,subprocess,resource
root=os.environ['OUT']
if os.environ.get('WORKER')=='1':
 import torch
 from transformers import Qwen2_5_VLForConditionalGeneration
 results=[]; models=[]
 for i in range(4):
  torch.cuda.set_device(i); torch.cuda.reset_peak_memory_stats(i)
  m=Qwen2_5_VLForConditionalGeneration.from_pretrained(os.environ['MODEL'],torch_dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).to('cuda:'+str(i)).eval()
  models.append(m)
  with torch.no_grad():
   ids=torch.tensor([[151644, 8948, 198]],device='cuda:'+str(i)); out=m(input_ids=ids,use_cache=False)
  torch.cuda.synchronize(i)
  results.append({'gpu':i,'allocated_peak_bytes':torch.cuda.max_memory_allocated(i),'reserved_peak_bytes':torch.cuda.max_memory_reserved(i),'output_finite':bool(torch.isfinite(out.logits).all())})
  del out,ids
 with open(os.path.join(root,'result.json'),'x') as f: json.dump({'kind':'four_independent_inference_replicas_no_training','gpus':results},f)
 sys.exit(0)
start=time.time(); env=os.environ.copy();env['WORKER']='1'
p=subprocess.Popen([sys.executable,'-c','import os,runpy;runpy.run_path(os.environ["ENTRY"],run_name="__main__")'],env=env,start_new_session=True,stdout=open(os.path.join(root,'worker.log'),'x'),stderr=subprocess.STDOUT)
status='finished'
while p.poll() is None:
 if time.time()-start>1200: status='deadline';break
 try:
  size=sum(os.stat(os.path.join(d,n)).st_blocks*512 for d,_,fs in os.walk(root) for n in fs)
 except OSError: status='storage_check_failed';break
 if size>=1000000000: status='storage_limit';break
 time.sleep(2)
if p.poll() is None:
 os.killpg(p.pid,signal.SIGTERM)
 try: p.wait(timeout=10)
 except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL);p.wait()
with open(os.path.join(root,'status.json'),'x') as f: json.dump({'status':status,'exit_code':p.returncode,'elapsed_seconds':time.time()-start},f)

import os,time,subprocess,signal,json,socket,resource
root=os.environ['OUT'];resource.setrlimit(resource.RLIMIT_FSIZE,(10000000000,10000000000))
with open(os.path.join(root,'recovery_protocol.json')) as f: protocol=json.load(f)
assert time.time()<protocol['absolute_deadline_epoch']
s=socket.socket();s.bind(('127.0.0.1',0));port=s.getsockname()[1];s.close();started=time.time();jobs=[]
for rank in range(2):
 env=os.environ.copy();env.update(RANK=str(rank),LOCAL_RANK=str(rank),WORLD_SIZE='2',MASTER_ADDR='127.0.0.1',MASTER_PORT=str(port))
 f=open(os.path.join(root,'repair_rank%d.log'%rank),'x')
 jobs.append(subprocess.Popen(['python','-c','import os,runpy;runpy.run_module(os.environ["MODULE"],run_name="__main__")'],env=env,start_new_session=True,stdout=f,stderr=subprocess.STDOUT))
reason='completed'
while any(p.poll() is None for p in jobs):
 if time.time()>=protocol['absolute_deadline_epoch']:reason='timeout';break
 try:size=sum(os.stat(os.path.join(d,n)).st_blocks*512 for d,_,fs in os.walk(root) for n in fs)
 except OSError:reason='storage_stat_failed';break
 if size>=40000000000:reason='storage_limit';break
 if any(p.poll() not in (None,0) for p in jobs):reason='worker_failure';break
 time.sleep(1)
if reason=='completed' and any(p.poll() not in (None,0) for p in jobs):reason='worker_failure'
if reason!='completed':
 for p in jobs:
  if p.poll() is None:
   try:os.killpg(p.pid,signal.SIGTERM)
   except ProcessLookupError:pass
 for p in jobs:
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:
   try:os.killpg(p.pid,signal.SIGKILL)
   except ProcessLookupError:pass
   p.wait()
with open(os.path.join(root,'repair_status.json'),'x') as f:json.dump({'reason':reason,'codes':[p.wait() for p in jobs],'attempt_elapsed_seconds':time.time()-started,'elapsed_seconds_from_original_conservative_start':time.time()-protocol['first_start_conservative_epoch']},f)

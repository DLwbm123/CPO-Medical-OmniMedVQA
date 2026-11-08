import os,time,subprocess,signal,json,resource
root=os.environ['OUT'];resource.setrlimit(resource.RLIMIT_FSIZE,(100000000,100000000))
started=time.time();deadline=float(os.environ['ABS_DEADLINE'])
with open(os.path.join(root,'nonzero_gradient_worker.log'),'x') as log:
 p=subprocess.Popen(['python','-c','import os,runpy;runpy.run_module(os.environ["MODULE"],run_name="__main__")'],start_new_session=True,stdout=log,stderr=subprocess.STDOUT)
 reason='completed'
 while p.poll() is None:
  if time.time()>=deadline:reason='deadline';break
  try:size=sum(os.stat(os.path.join(d,n)).st_blocks*512 for d,_,fs in os.walk(root) for n in fs)
  except OSError:reason='storage_stat_failed';break
  if size>=1000000000:reason='storage_limit';break
  time.sleep(2)
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=10)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 if p.returncode!=0 and reason=='completed':reason='worker_failure'
 with open(os.path.join(root,'nonzero_gradient_status.json'),'x') as handle:json.dump({'reason':reason,'exit_code':p.returncode,'elapsed_seconds':time.time()-started,'absolute_deadline_epoch':deadline},handle)

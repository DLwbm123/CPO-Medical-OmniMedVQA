import argparse,json,os,itertools
from pathlib import Path
from torch.utils.data import BatchSampler
from accelerate.data_loader import BatchSamplerShard
from trl.trainer.utils import RepeatSampler
parser=argparse.ArgumentParser()
parser.add_argument('--output',required=True)
args=parser.parse_args()
out=Path(args.output);out.mkdir(parents=True,exist_ok=False);results={}
for task,n in [('MedBook',1210),('PathVQA',19648),('SLAKE',4918)]:
 for seed in (42,43,44):
  shards=[]
  for rank in (0,1):
   sampler=RepeatSampler(data_source=range(n),mini_repeat_count=8,batch_size=1,repeat_count=1,shuffle=True,seed=seed)
   batches=BatchSampler(sampler,batch_size=4,drop_last=True)
   shards.append(BatchSamplerShard(batches,num_processes=2,process_index=rank,split_batches=False,even_batches=True))
  paired=list(itertools.islice(zip(*[iter(x) for x in shards]),256));assert len(paired)==256
  counts={'paired_batches':len(paired),'rank0_mixed':0,'rank1_mixed':0,'different_prompt_across_ranks':0}
  for a,b in paired:
   counts['rank0_mixed']+=int(len(set(a))!=1);counts['rank1_mixed']+=int(len(set(b))!=1);counts['different_prompt_across_ranks']+=int(set(a)!=set(b))
  results[f'{task}_{seed}']=counts
(out/'result.json').write_text(json.dumps(results,indent=2)+'\n');(out/'status.json').write_text('{"status":"complete"}\n')

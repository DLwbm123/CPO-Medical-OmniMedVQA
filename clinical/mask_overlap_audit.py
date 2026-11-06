import argparse,json,os,itertools
from pathlib import Path
import torch
torch.set_num_threads(4)
parser=argparse.ArgumentParser()
parser.add_argument('--runs-root',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
root=Path(args.runs_root);out=Path(args.output);out.mkdir(parents=True,exist_ok=False);runs={'42':'cpo_3stage_20261105','43':'seed43_paired','44':'seed44_paired'}
def cat(k):
 if 'visual' in k and 'merger' in k:return 'vision_merger'
 if 'visual' in k:return 'vision_encoder'
 if 'lm_head' in k or 'embed_tokens' in k:return 'token_embedding_head'
 return 'other_language_or_misc'
def read(seed,stage):
 d=torch.load(root/runs[seed]/'masks'/f'task_{stage}.pt',map_location='cpu',weights_only=True,mmap=True)
 if isinstance(d,dict) and 'masks' in d:d=d['masks']
 assert isinstance(d,dict) and all(torch.is_tensor(v) and v.dtype==torch.bool for v in d.values());return d
def stats(d):
 a={}
 for k,v in d.items():
  g=cat(k);a.setdefault(g,{'selected':0,'represented_coordinates':0,'tensors':0});a[g]['selected']+=int(v.count_nonzero());a[g]['represented_coordinates']+=v.numel();a[g]['tensors']+=1
 return a
result={'seeds':{},'pairs':{}}
for seed in runs:
 a,b=read(seed,1),read(seed,2);missing=0
 for k,v in a.items():
  if k not in b:missing+=int(v.count_nonzero())
  else:missing+=int((v & ~b[k]).count_nonzero())
 result['seeds'][seed]={'stage1':stats(a),'cumulative_stage2':stats(b),'stage1_selected_not_in_stage2':missing}
 del a,b;print('seed',seed,flush=True)
for left,right in itertools.combinations(runs,2):
 a,b=read(left,1),read(right,1);groups={}
 for k in a.keys()|b.keys():
  av,bv=a.get(k),b.get(k);g=cat(k);groups.setdefault(g,{'intersection':0,'union':0})
  if av is None:groups[g]['union']+=int(bv.count_nonzero())
  elif bv is None:groups[g]['union']+=int(av.count_nonzero())
  else:
   assert av.shape==bv.shape;groups[g]['intersection']+=int((av & bv).count_nonzero());groups[g]['union']+=int((av | bv).count_nonzero())
 result['pairs'][left+'_'+right]=groups;del a,b;print('pair',left,right,flush=True)
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'status.json').write_text('{"status":"complete"}\n')

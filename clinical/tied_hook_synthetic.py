import argparse,json,os,sys
from pathlib import Path
import torch
from torch import nn
parser=argparse.ArgumentParser()
parser.add_argument('--root',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=False);sys.path[:0]=[str(root),str(root/'src')];from src.trainer.grpo_trainer_cl import CLGRPOTrainer
class Tiny(nn.Module):
 def __init__(self):
  super().__init__();self.embed=nn.Embedding(3,2);self.head=nn.Linear(2,3,bias=False);self.head.weight=self.embed.weight
m=Tiny();physical=dict(m.named_parameters());aliased=dict(m.named_parameters(remove_duplicate=False));assert len(physical)==1 and aliased['head.weight'] is physical['embed.weight']
p=physical['embed.weight'];ref=torch.tensor([[0.,0.],[0.,0.],[0.,0.]]);mask=torch.tensor([[True,False],[True,False],[False,True]])
with torch.no_grad():p.copy_(torch.tensor([[1.,0.],[-2.,0.],[0.,3.]]))
lambda_=2.;indices=mask.flatten().nonzero().flatten();holder=object.__new__(CLGRPOTrainer);holder._mask_flat_idx={'embed.weight':indices};holder._pending_mask_grads={'embed.weight':lambda_/len(indices)*torch.sign(p.detach().flatten()[indices]-ref.flatten()[indices])}
base=(p*p).sum();baseline=torch.autograd.grad(base,p,retain_graph=True)[0];expected=baseline+lambda_/len(indices)*torch.sign(p.detach()-ref)*mask
h=p.register_hook(holder._make_mask_grad_hook('embed.weight'));base.backward();actual=p.grad.detach().clone();h.remove();delta=(actual-expected).abs();assert torch.allclose(actual,expected,atol=1e-6)
assert (delta==0).all() and len(physical)==1
result={'physical_parameter_count':len(physical),'alias_parameter_names':len(aliased),'matched_physical_hooks':1,'selected_physical_coordinates':len(indices),'lambda':lambda_,'max_abs_gradient_difference_from_reference':float(delta.max()),'hook_matches_explicit_masked_l1':True,'scope':'synthetic CPU no DDP or ZeRO; pending gradient manually constructed'}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'status.json').write_text('{"status":"complete"}\n')

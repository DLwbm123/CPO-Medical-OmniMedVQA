import argparse,json,os,sys,time
from pathlib import Path
from types import SimpleNamespace
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
parser=argparse.ArgumentParser()
parser.add_argument('--root',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=False);sys.path.insert(0,str(root));from src.dataset.grpo_dataset import GRPODataset
modelpath=root/'runs/gspo_3stage_20261104/training/MedicalVQA_MedBook';data=root/'data/prepared/MedicalVQA_MedBook';p=AutoProcessor.from_pretrained(str(modelpath),min_pixels=128*32*32,max_pixels=512*32*32,local_files_only=True);p.tokenizer.padding_side='left'
args=SimpleNamespace(prompt_path=str(root/'src/dataset/prompts_2.yaml'),image_folder=str(data/'images'))
d=GRPODataset(data_path=str(data/'jsons/train/data.json'),processor=p,data_args=args,model_id=str(modelpath));m=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(modelpath),dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().train();m.config.use_cache=False;m.gradient_checkpointing_enable();m.enable_input_require_grads()
mask=torch.load(root/'runs/cpo_3stage_20261105/masks/task_1.pt',map_location='cpu',weights_only=True,mmap=True);params=dict(m.named_parameters());assert all(k in params and params[k].shape==v.shape for k,v in mask.items());fisher={};losses=[]
for i in range(8):
 row=d[i];gold=row['assistant'].strip();assert gold and len(gold)<16
 text=p.apply_chat_template(row['prompt'],tokenize=False,add_generation_prompt=True);prefix=text+'\\boxed{';full=prefix+gold
 x=p(text=[prefix],images=row['images'],return_tensors='pt');y=p(text=[full],images=row['images'],return_tensors='pt');n=x.input_ids.shape[1];assert y.input_ids.shape[1]>n and torch.equal(x.input_ids,y.input_ids[:,:n]);targets=y.input_ids[:,n:].cuda();y=y.to('cuda');m.zero_grad(set_to_none=True)
 logits=m(**y,use_cache=False).logits[:,n-1:-1,:].float();assert logits.shape[1]==targets.shape[1];loss=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),targets.reshape(-1));assert bool(torch.isfinite(loss));loss.backward();losses.append(float(loss.detach().cpu()))
 for k in mask:
  g=params[k].grad
  if g is None:continue
  v=g.detach().float().square().cpu()
  if k not in fisher:fisher[k]=v
  else:fisher[k].add_(v)
 (out/'status.json').write_text(json.dumps({'status':'running','processed':i+1,'total':8,'unix':time.time()})+'\n');print('processed',i+1,flush=True)
counts={'samples':8,'mask_tensors':len(mask),'gradient_tensors':len(fisher),'mask_coordinates':0,'sensitivity_coordinates':0,'intersection':0,'insufficient_positive_tensors':0,'nonfinite_tensors':0}
for k,ref in mask.items():
 num=int(ref.count_nonzero());counts['mask_coordinates']+=num
 if not num or k not in fisher:continue
 scores=fisher[k].flatten();counts['nonfinite_tensors']+=int(not bool(torch.isfinite(scores).all()));positive=int((scores>0).sum());actual=min(num,positive);counts['insufficient_positive_tensors']+=int(actual<num)
 if actual:
  selected=torch.topk(scores,actual,sorted=False).indices;counts['sensitivity_coordinates']+=actual;counts['intersection']+=int(ref.flatten()[selected].count_nonzero())
counts['union']=counts['mask_coordinates']+counts['sensitivity_coordinates']-counts['intersection'];counts['loss_min']=min(losses);counts['loss_max']=max(losses)
(out/'result.json').write_text(json.dumps(counts,indent=2)+'\n');(out/'status.json').write_text(json.dumps({'status':'complete','unix':time.time()})+'\n')

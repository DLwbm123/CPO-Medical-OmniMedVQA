import argparse,json,collections,os,re
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from transformers import AutoProcessor,AutoConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel
import torch
parser=argparse.ArgumentParser()
parser.add_argument('--root',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=False);path=root/'runs/gspo_3stage_20261104/training/MedicalVQA_MedBook';p=AutoProcessor.from_pretrained(str(path),min_pixels=128*32*32,max_pixels=512*32*32,local_files_only=True);p.tokenizer.padding_side='left';dummy=SimpleNamespace(config=AutoConfig.from_pretrained(str(path),local_files_only=True));stats={}
for task in ('MedicalVQA_MedBook','MedicalVQA_Pathology','MedicalVQA_Radiology'):
 data=root/'data/prepared'/task;rows=json.loads((data/'jsons/train/data.json').read_text())[:32];c=collections.Counter()
 for start in range(0,32,4):
  images=[];texts=[]
  for row in rows[start:start+4]:
   with Image.open(data/'images'/row['image']) as im:image=im.convert('RGB')
   question=re.sub(r'\n?<image>\n?','',row['conversations'][0]['value']);question=re.sub(r'\s+',' ',question).strip()
   msgs=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':'Answer the medical image question. Put only the final answer in \\boxed{}. '+question}]}];images.append(image);texts.append(p.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True))
  b=p(text=texts,images=images,padding=True,return_tensors='pt');bp,bd=Qwen2_5_VLModel.get_rope_index(dummy,b.input_ids,image_grid_thw=b.image_grid_thw,attention_mask=b.attention_mask)
  for i in range(4):
   x=p(text=[texts[i]],images=[images[i]],padding=True,return_tensors='pt');sp,sd=Qwen2_5_VLModel.get_rope_index(dummy,x.input_ids,image_grid_thw=x.image_grid_thw,attention_mask=x.attention_mask);pad=int((1-b.attention_mask[i]).sum());c['n']+=1;c['left_padded']+=int(pad>0);c['prefill_equal']+=int(torch.equal(bp[:,i,b.attention_mask[i].bool()],sp[:,0,:]));c['delta_mismatch']+=int(int(bd[i])!=int(sd[0]));c['delta_negative_pad']+=int(int(bd[i]-sd[0])==-pad)
 stats[task]=dict(c)
(out/'result.json').write_text(json.dumps(stats,indent=2)+'\n');(out/'status.json').write_text('{"status":"complete"}\n')

import argparse,json,time,collections,sys
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
parser=argparse.ArgumentParser()
parser.add_argument('--root',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
root=Path(args.root);from reward_funcs_medical import score_response
out=Path(args.output);out.mkdir(parents=True,exist_ok=False);data=root/'data/prepared/MedicalVQA_MedBook';rows=json.loads((data/'jsons/validation/data.json').read_text())[:32]
models={'42_gspo':root/'runs/gspo_3stage_20261104/training/MedicalVQA_Radiology','42_cpo':root/'runs/cpo_3stage_20261105/training/MedicalVQA_Radiology','44_cpo':root/'runs/seed44_paired/training/MedicalVQA_Radiology'}
p=AutoProcessor.from_pretrained(str(models['42_gspo']),min_pixels=128*32*32,max_pixels=512*32*32,local_files_only=True);p.tokenizer.padding_side='left'
for name,path in models.items():
 (out/'status.json').write_text(json.dumps({'status':'running','model':name,'unix':time.time()}));m=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(path),dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval();c=collections.Counter()
 for start in range(0,32,4):
  batch=rows[start:start+4];images=[];texts=[]
  for row in batch:
   with Image.open(data/'images'/row['image']) as im: image=im.convert('RGB')
   images.append(image);msgs=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':'Answer the medical image question. Put only the final answer in \\boxed{}. '+row['conversations'][0]['value']}]}];texts.append(p.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True))
  x=p(text=texts,images=images,padding=True,return_tensors='pt').to('cuda');outputs={}
  original=m.model.get_rope_index
  def corrected(*args,**kwargs):
   pos,delta=original(*args,**kwargs)
   mask=kwargs.get('attention_mask')
   if mask is None: raise RuntimeError('attention mask required')
   return pos,delta+(1-mask).sum(-1,keepdim=True)
  for penalty in (1.05,1.0):
   m.model.get_rope_index=original if penalty==1.05 else corrected
   with torch.inference_mode(): ids=m.generate(**x,do_sample=False,max_new_tokens=128,repetition_penalty=1.05)
   batch_outputs=p.batch_decode(ids[:,x['input_ids'].shape[1]:],skip_special_tokens=True)
   for j,(row,image,text) in enumerate(zip(batch,images,texts)):
    y=p(text=[text],images=[image],padding=True,return_tensors='pt').to('cuda')
    with torch.inference_mode(): one=m.generate(**y,do_sample=False,max_new_tokens=128,repetition_penalty=1.05)
    single=p.batch_decode(one[:,y['input_ids'].shape[1]:],skip_special_tokens=True)[0];bat=batch_outputs[j];tag='original' if penalty==1.05 else 'corrected';gold=row['conversations'][1]['value'];c[tag+'_n']+=1;c[tag+'_equal']+=int(single==bat);c[tag+'_single_correct']+=int(score_response(single,gold));c[tag+'_batch_correct']+=int(score_response(bat,gold));c[tag+'_single_truncated']+=int(one.shape[1]-y['input_ids'].shape[1]>=128);c[tag+'_batch_truncated']+=int(ids.shape[1]-x['input_ids'].shape[1]>=128)
  m.model.get_rope_index=original
  print(name,start,flush=True)
 (out/(name+'.json')).write_text(json.dumps(dict(c),indent=2)+'\n');del m;torch.cuda.empty_cache()
(out/'status.json').write_text(json.dumps({'status':'complete','unix':time.time()})+'\n')

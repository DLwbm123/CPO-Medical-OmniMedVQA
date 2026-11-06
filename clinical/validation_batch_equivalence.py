import argparse,json,time,collections,re,sys
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
parser=argparse.ArgumentParser()
parser.add_argument('--root',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
root=Path(args.root);from reward_funcs_medical import score_response
out=Path(args.output);out.mkdir(parents=True,exist_ok=False);data=root/'data/prepared/MedicalVQA_MedBook';rows=json.loads((data/'jsons/validation/data.json').read_text());assert len(rows)==130
models={}
for seed,control,paired in [(42,'gspo_3stage_20261104','cpo_3stage_20261105'),(43,'seed43_control','seed43_paired'),(44,'seed44_control','seed44_paired')]:
 for label,run in [('gspo_final',control),('cpo_final',paired)]:models[f'{seed}_{label}']=root/'runs'/run/'training/MedicalVQA_Radiology'
processor=AutoProcessor.from_pretrained(str(next(iter(models.values()))),min_pixels=128*32*32,max_pixels=512*32*32,local_files_only=True);processor.tokenizer.padding_side='left'
for name,path in models.items():
 (out/'status.json').write_text(json.dumps({'status':'running','model':name,'unix':time.time()})+'\n');model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(path),dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval();c=collections.Counter()
 for start in range(0,len(rows),4):
  batch=rows[start:start+4];images=[];texts=[]
  for row in batch:
   with Image.open(data/'images'/row['image']) as im:image=im.convert('RGB')
   images.append(image);msgs=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':'Answer the medical image question. Put only the final answer in \\boxed{}. '+row['conversations'][0]['value']}]}];texts.append(processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True))
  x=processor(text=texts,images=images,padding=True,return_tensors='pt').to('cuda')
  with torch.inference_mode():ids=model.generate(**x,do_sample=False,max_new_tokens=2048,repetition_penalty=1.05)
  bat=processor.batch_decode(ids[:,x['input_ids'].shape[1]:],skip_special_tokens=True);assert len(bat)==len(batch)
  for row,image,text,batpred in zip(batch,images,texts,bat):
   y=processor(text=[text],images=[image],padding=True,return_tensors='pt').to('cuda')
   with torch.inference_mode():ids=model.generate(**y,do_sample=False,max_new_tokens=2048,repetition_penalty=1.05)
   single=processor.batch_decode(ids[:,y['input_ids'].shape[1]:],skip_special_tokens=True)[0];gold=row['conversations'][1]['value'];a=bool(score_response(single,gold));b=bool(score_response(batpred,gold));c['n']+=1;c['exact_text_equal']+=int(single==batpred);c['single_correct']+=a;c['batch_correct']+=b;c['batch_rescue']+=int(b and not a);c['batch_harm']+=int(a and not b)
   def valid(s):
    m=re.findall(r'\\boxed\s*\{([^{}]*)\}',s);return bool(re.fullmatch('[A-D]',(m[-1] if m else s).strip().upper()))
   c['single_valid_letter']+=valid(single);c['batch_valid_letter']+=valid(batpred)
  if start%20==0:print(name,start,flush=True)
 assert c['n']==130 and c['batch_correct']-c['single_correct']==c['batch_rescue']-c['batch_harm'];(out/(name+'.json')).write_text(json.dumps(dict(c),indent=2)+'\n');del model;torch.cuda.empty_cache();print(name,dict(c),flush=True)
(out/'status.json').write_text(json.dumps({'status':'complete','unix':time.time()})+'\n')

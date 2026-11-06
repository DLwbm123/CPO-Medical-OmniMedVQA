import json,time,collections,os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
root=Path(os.environ['MEDICAL_PROJECT_ROOT']);out=Path(os.environ['VALIDATION_DIAG_DIR']);data=root/'data/prepared/MedicalVQA_MedBook';rows=json.loads((data/'jsons/validation/data.json').read_text());assert len(rows)==130
models={}
for seed,control,paired in [(42,'gspo_3stage_20261104','cpo_3stage_20261105'),(43,'seed43_control','seed43_paired'),(44,'seed44_control','seed44_paired')]:
 for label,run,task in [('initial',control,'MedicalVQA_MedBook'),('gspo_final',control,'MedicalVQA_Radiology'),('cpo_final',paired,'MedicalVQA_Radiology')]:models[f'{seed}_{label}']=root/'runs'/run/'training'/task
processor=AutoProcessor.from_pretrained(str(next(iter(models.values()))),min_pixels=128*32*32,max_pixels=512*32*32,local_files_only=True);tok=processor.tokenizer;ids=[tok.encode(v,add_special_tokens=False) for v in 'ABCD'];assert all(len(x)==1 for x in ids) and len({x[0] for x in ids})==4;ids=[x[0] for x in ids]
for name,path in models.items():
 (out/'status.json').write_text(json.dumps({'status':'running','model':name,'unix':time.time()})+'\n');model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(path),dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval();c=collections.Counter()
 for i,row in enumerate(rows):
  with Image.open(data/'images'/row['image']) as im:real=im.convert('RGB')
  black=Image.new('RGB',real.size,(0,0,0));gold=str(row['conversations'][1]['value']).strip().upper();assert gold in 'ABCD';pred=[]
  for image in (real,black):
   msgs=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':'Answer the medical image question. Put only the final answer in \\boxed{}. '+row['conversations'][0]['value']}]}];text=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)+'\\boxed{';x=processor(text=[text],images=[image],return_tensors='pt').to('cuda')
   with torch.inference_mode():pred.append(int(model(**x).logits[0,-1,ids].argmax()))
  a,b=pred;expected='ABCD'.index(gold);c['n']+=1;c['real_correct']+=int(a==expected);c['black_correct']+=int(b==expected);c['decision_flip']+=int(a!=b);c['real_rescue']+=int(a==expected and b!=expected);c['real_harm']+=int(a!=expected and b==expected)
  if i%25==0:print(name,i,flush=True)
 assert c['n']==130 and c['real_correct']-c['black_correct']==c['real_rescue']-c['real_harm'];(out/(name+'.json')).write_text(json.dumps(dict(c),indent=2)+'\n');del model;torch.cuda.empty_cache();print(name,dict(c),flush=True)
(out/'status.json').write_text(json.dumps({'status':'complete','unix':time.time()})+'\n')

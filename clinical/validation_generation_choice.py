import json,time,collections,re,sys,os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
root=Path(os.environ['MEDICAL_PROJECT_ROOT']);sys.path.insert(0,str(root));from src.train.reward_funcs_medical import score_response
out=Path(os.environ['VALIDATION_DIAG_DIR']);data=root/'data/prepared/MedicalVQA_MedBook';rows=json.loads((data/'jsons/validation/data.json').read_text());assert len(rows)==130
models={}
for seed,control,paired in [(42,'gspo_3stage_20261104','cpo_3stage_20261105'),(43,'seed43_control','seed43_paired'),(44,'seed44_control','seed44_paired')]:
 for label,run in [('gspo_final',control),('cpo_final',paired)]:models[f'{seed}_{label}']=root/'runs'/run/'training/MedicalVQA_Radiology'
processor=AutoProcessor.from_pretrained(str(next(iter(models.values()))),min_pixels=128*32*32,max_pixels=512*32*32,local_files_only=True);tok=processor.tokenizer;ids=[tok.encode(v,add_special_tokens=False) for v in 'ABCD'];assert all(len(x)==1 for x in ids) and len({x[0] for x in ids})==4;ids=[x[0] for x in ids]
for name,path in models.items():
 (out/'status.json').write_text(json.dumps({'status':'running','model':name,'unix':time.time()})+'\n');model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(path),dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval();c=collections.Counter()
 for i,row in enumerate(rows):
  with Image.open(data/'images'/row['image']) as im:image=im.convert('RGB')
  gold=str(row['conversations'][1]['value']).strip().upper();assert gold in 'ABCD';msgs=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':'Answer the medical image question. Put only the final answer in \\boxed{}. '+row['conversations'][0]['value']}]}];prompt=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True);a=processor(text=[prompt],images=[image],return_tensors='pt').to('cuda');b=processor(text=[prompt+'\\boxed{'],images=[image],return_tensors='pt').to('cuda')
  with torch.inference_mode():
   gen=model.generate(**a,do_sample=False,max_new_tokens=2048,repetition_penalty=1.05); text=processor.batch_decode(gen[:,a['input_ids'].shape[1]:],skip_special_tokens=True)[0];rank=int(model(**b).logits[0,-1,ids].argmax())
  g=bool(score_response(text,gold));f=rank=='ABCD'.index(gold);boxes=re.findall(r'\\boxed\s*\{([^{}]*)\}',text);value=(boxes[-1] if boxes else text).strip().upper();valid=bool(re.fullmatch('[A-D]',value));c['n']+=1;c['generated_correct']+=g;c['forced_correct']+=f;c['generated_valid_letter']+=valid;c['forced_rescues']+=int(f and not g);c['forced_harms']+=int(g and not f)
  if i%25==0:print(name,i,flush=True)
 assert c['n']==130 and c['forced_correct']-c['generated_correct']==c['forced_rescues']-c['forced_harms'];(out/(name+'.json')).write_text(json.dumps(dict(c),indent=2)+'\n');del model;torch.cuda.empty_cache();print(name,dict(c),flush=True)
(out/'status.json').write_text(json.dumps({'status':'complete','unix':time.time()})+'\n')

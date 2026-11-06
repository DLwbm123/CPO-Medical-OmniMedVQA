import json,os,time
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
root=Path(os.environ['MEDICAL_PROJECT_ROOT']); out=Path(os.environ['VALIDATION_DIAG_DIR']); data=root/'data/prepared/MedicalVQA_MedBook'; rows=json.loads((data/'jsons/validation/data.json').read_text()); assert len(rows)==130
models={}
for seed,control,paired in [(42,'gspo_3stage_20261104','cpo_3stage_20261105'),(43,'seed43_control','seed43_paired'),(44,'seed44_control','seed44_paired')]:
 for label,run,task in [('initial',control,'MedicalVQA_MedBook'),('gspo_final',control,'MedicalVQA_Radiology'),('cpo_final',paired,'MedicalVQA_Radiology')]: models[f'{seed}_{label}']=root/'runs'/run/'training'/task
processor=AutoProcessor.from_pretrained(str(next(iter(models.values()))),min_pixels=128*32*32,max_pixels=512*32*32,local_files_only=True);tok=processor.tokenizer; ids=[tok.encode(v,add_special_tokens=False) for v in 'ABCD'];assert all(len(t)==1 for t in ids) and len({t[0] for t in ids})==4; ids=[t[0] for t in ids];(out/'token_ids.json').write_text(json.dumps(dict(zip('ABCD',ids)))+'\n')
for name,path in models.items():
 state={'status':'running','model':name,'unix':time.time()};(out/'status.json').write_text(json.dumps(state)+'\n');model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(path),dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval();correct=0; sums=0.0; classes={k:0 for k in 'ABCD'}
 for i,row in enumerate(rows):
  with Image.open(data/'images'/row['image']) as im:image=im.convert('RGB')
  msgs=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':'Answer the medical image question. Put only the final answer in \\boxed{}. '+row['conversations'][0]['value']}]}];text=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)+'\\boxed{';x=processor(text=[text],images=[image],return_tensors='pt').to('cuda');gold=str(row['conversations'][1]['value']).strip().upper();assert gold in 'ABCD'
  with torch.inference_mode(): logits=model(**x).logits[0,-1,ids].float(); probs=logits.log_softmax(dim=0); pred=int(probs.argmax()); correct+=pred=='ABCD'.index(gold);sums+=float(probs['ABCD'.index(gold)]);classes['ABCD'[pred]]+=1
  if i%25==0:print(name,i,flush=True)
 result={'n':len(rows),'correct':int(correct),'accuracy':correct/len(rows),'mean_gold_logprob_given_ABCD':sums/len(rows),'predicted_label_counts':classes};(out/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n');del model;torch.cuda.empty_cache();print(name,result,flush=True)
(out/'status.json').write_text(json.dumps({'status':'complete','unix':time.time()})+'\n')

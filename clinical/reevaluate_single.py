import argparse,json,os,re,sys,time
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
parser=argparse.ArgumentParser()
parser.add_argument('--root',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
from reward_funcs_medical import score_response
runs={'42_gspo':'gspo_3stage_20261104','42_cpo':'cpo_3stage_20261105','43_gspo':'seed43_control','43_cpo':'seed43_paired','44_gspo':'seed44_control','44_cpo':'seed44_paired'}
tasks=[('MedBook','MedicalVQA_MedBook',589),('PathVQA','MedicalVQA_Pathology',6719),('SLAKE','MedicalVQA_Radiology',1061)]
plan=[]
for label,folder,n in tasks:
 for name,run in runs.items(): plan.append((label,folder,n,name,root/'runs'/run/'training/MedicalVQA_Radiology'))
 if label=='MedBook':
  for seed,run in [('42','gspo_3stage_20261104'),('43','seed43_control'),('44','seed44_control')]:plan.append((label,folder,n,seed+'_shared_stage1',root/'runs'/run/'training/MedicalVQA_MedBook'))
def save(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n');os.replace(tmp,path)
for idx,(label,folder,n,name,modelpath) in enumerate(plan):
 data=root/'data/prepared'/folder;rows=json.loads((data/'jsons/test/data.json').read_text());assert len(rows)==n and (modelpath/'config.json').is_file()
 save(out/'status.json',{'status':'loading','job':idx+1,'total_jobs':len(plan),'task':label,'model':name,'time':time.time()})
 p=AutoProcessor.from_pretrained(str(modelpath),min_pixels=128*32*32,max_pixels=512*32*32,local_files_only=True);p.tokenizer.padding_side='left'
 model=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(modelpath),dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval();correct=valid=0;t0=time.time()
 for i,row in enumerate(rows):
  with Image.open(data/'images'/row['image']) as im:image=im.convert('RGB')
  text='Answer the medical image question. Put only the final answer in \\boxed{}. '+row['conversations'][0]['value'];msgs=[{'role':'user','content':[{'type':'image','image':image},{'type':'text','text':text}]}];prompt=p.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
  x=p(text=[prompt],images=[image],padding=True,return_tensors='pt').to('cuda')
  with torch.inference_mode(): ids=model.generate(**x,do_sample=False,max_new_tokens=2048,repetition_penalty=1.05)
  answer=p.batch_decode(ids[:,x['input_ids'].shape[1]:],skip_special_tokens=True)[0];correct+=int(score_response(answer,row['conversations'][1]['value']))
  if label=='MedBook':
   box=re.findall(r'\\boxed\s*\{([^{}]*)\}',answer);valid+=int(bool(re.fullmatch('[A-E]',(box[-1] if box else answer).strip().upper())))
  if (i+1)%50==0 or i+1==n:save(out/'status.json',{'status':'running','job':idx+1,'total_jobs':len(plan),'task':label,'model':name,'processed':i+1,'total':n,'time':time.time()})
 result={'task':label,'model':name,'n':n,'correct':correct,'accuracy':correct/n,'elapsed_s':time.time()-t0,'scorer':'normalized exact match, final boxed answer else complete output','valid_letter':valid if label=='MedBook' else None}
 save(out/(label+'_'+name+'.json'),result);print('FINISHED',json.dumps(result),flush=True);del model,p;torch.cuda.empty_cache()
save(out/'status.json',{'status':'complete','job':len(plan),'total_jobs':len(plan),'time':time.time()})

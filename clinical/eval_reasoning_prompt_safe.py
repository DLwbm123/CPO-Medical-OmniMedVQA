"""Single-item evaluation using the clinical scorer; separate from historical evaluation."""
import argparse,json
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
from reward_funcs_medical import score_response
PROMPT='Answer the medical image question. Think step by step, then put only the final answer in \\boxed{}. For multiple choice, output only the option letter; otherwise give a short answer. '
def main():
 a=argparse.ArgumentParser()
 a.add_argument('--model',required=True);a.add_argument('--data',required=True);a.add_argument('--images',required=True);a.add_argument('--output',required=True);a.add_argument('--limit',type=int,default=0);a.add_argument('--batch-size',type=int,default=1);a.add_argument('--max-tokens',type=int,default=2048)
 x=a.parse_args()
 if x.batch_size != 1: a.error('This safe entry requires batch size 1; see batch consistency investigation')
 rows=json.loads(Path(x.data).read_text(encoding='utf-8'))
 if x.limit:rows=rows[:x.limit]
 if not rows:raise ValueError('Empty evaluation data')
 processor=AutoProcessor.from_pretrained(x.model,min_pixels=128*32*32,max_pixels=512*32*32,local_files_only=True);processor.tokenizer.padding_side='left'
 model=Qwen2_5_VLForConditionalGeneration.from_pretrained(x.model,dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
 out=Path(x.output);out.mkdir(parents=True,exist_ok=False);predictions=[]
 for start in range(0,len(rows),x.batch_size):
  batch=rows[start:start+x.batch_size];texts=[];images=[]
  for item in batch:
   with Image.open(Path(x.images)/item['image']) as image: images.append(image.convert('RGB'))
   msgs=[{'role':'user','content':[{'type':'image','image':images[-1]},{'type':'text','text':PROMPT+item['conversations'][0]['value']}]}]
   texts.append(processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True))
  inputs=processor(text=texts,images=images,padding=True,return_tensors='pt').to('cuda')
  with torch.inference_mode(): ids=model.generate(**inputs,do_sample=False,max_new_tokens=x.max_tokens,repetition_penalty=1.05)
  answers=processor.batch_decode(ids[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)
  if len(answers)!=len(batch):raise RuntimeError('Prediction count mismatch')
  for item,pred in zip(batch,answers):
   gold=item['conversations'][1]['value'];predictions.append({'question_id':item['id'],'model_pred':pred,'ground_truth':gold,'correct':score_response(pred,gold)})
  if start % (x.batch_size*50)==0:print('evaluated',len(predictions),'/',len(rows),flush=True)
 with (out/'predictions.jsonl').open('w',encoding='utf-8') as f:
  for row in predictions:f.write(json.dumps(row,ensure_ascii=False)+'\n')
 metric={'n':len(predictions),'correct':sum(p['correct'] for p in predictions),'accuracy':sum(p['correct'] for p in predictions)/len(predictions),'scoring':'normalized exact match, boxed answer or full response'}
 (out/'metrics.json').write_text(json.dumps(metric,indent=2)+'\n');print('metric',metric,flush=True)
if __name__=='__main__':main()

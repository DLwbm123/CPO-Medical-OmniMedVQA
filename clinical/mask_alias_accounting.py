import argparse,os,json
from pathlib import Path
import torch
from accelerate import init_empty_weights
from transformers import AutoConfig,Qwen2_5_VLForConditionalGeneration
parser=argparse.ArgumentParser()
parser.add_argument('--root',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
root=Path(args.root);out=Path(args.output);out.mkdir(parents=True,exist_ok=False);modelpath=root/'runs/gspo_3stage_20261104/training/MedicalVQA_MedBook';config=AutoConfig.from_pretrained(str(modelpath),local_files_only=True)
with init_empty_weights():m=Qwen2_5_VLForConditionalGeneration(config)
m.tie_weights();physical=dict(m.named_parameters());aliases=dict(m.named_parameters(remove_duplicate=False));assert aliases['lm_head.weight'] is aliases['model.language_model.embed_tokens.weight'];result={}
runs={'42':'cpo_3stage_20261105','43':'seed43_paired','44':'seed44_paired'}
for seed,run in runs.items():
 for stage in (1,2):
  mask=torch.load(root/'runs'/run/'masks'/f'task_{stage}.pt',map_location='cpu',weights_only=True,mmap=True);missing=set(mask)-set(physical);assert missing<=set(aliases)-set(physical)
  canonical='model.language_model.embed_tokens.weight';alias='lm_head.weight';assert canonical in mask and alias in mask
  a,b=mask[canonical],mask[alias];assert a.shape==b.shape
  result[f'{seed}_stage{stage}']={'mask_keys':len(mask),'unmatched_physical_keys':len(missing),'total_selected_state_keys':sum(int(x.count_nonzero()) for x in mask.values()),'selected_matched_physical':sum(int(v.count_nonzero()) for k,v in mask.items() if k in physical),'selected_unmatched_alias':sum(int(mask[k].count_nonzero()) for k in missing),'embedding_selected':int(a.count_nonzero()),'output_alias_selected':int(b.count_nonzero()),'embedding_alias_intersection':int((a&b).count_nonzero()),'embedding_alias_equal':torch.equal(a,b)}
  del mask;print(seed,stage,flush=True)
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n');(out/'status.json').write_text('{"status":"complete"}\n')

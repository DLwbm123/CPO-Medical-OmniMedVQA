import collections, json, os, random
from pathlib import Path
root=Path(os.environ['OMNIMEDVQA_ROOT']); source=root/'cpo_modality_v1'; dest=root/'cpo_question_type_v1'; stage=root/'.cpo_question_type_v1.preparing'
if dest.exists() or stage.exists(): raise FileExistsError('output already exists')
order=[('Modality Recognition','Modality'),('Anatomy Identification','Anatomy'),('Disease Diagnosis','Diagnosis'),('Lesion Grading','Grading'),('Other Biological Attributes','Attributes')]
names=dict(order); data=collections.defaultdict(list); groups=collections.defaultdict(list); split_by_image={}
for p in sorted(source.glob('MedicalVQA_*/jsons/*/data.json')):
 split=p.parent.name
 if split not in ('train','validation','test'):continue
 for row in json.loads(p.read_text()):
  img=row['image']; typ=row['question_type']
  if typ not in names:raise ValueError('unknown type')
  old=split_by_image.setdefault(img,split)
  if old!=split:raise ValueError('split conflict')
  groups[img].append((split,typ,row))
rng=random.Random(20261104); lost=collections.Counter(); eligible=collections.Counter(); assigned=collections.Counter()
for img, rows in sorted(groups.items()):
 choices=sorted({t for _,t,_ in rows}); selected=choices[rng.randrange(len(choices))]
 for split,t,row in rows:
  eligible[(t,split)]+=1
  if t==selected:data[(t,split)].append(row);assigned[(t,split)]+=1
  else:lost[(t,split)]+=1
stage.mkdir()
for typ,name in order:
 task=stage/('MedicalVQA_'+name);task.mkdir();os.symlink(str(root/'data/OmniMedVQA/Images'),task/'images')
 for split in ('train','validation','test'):
  d=task/'jsons'/split;d.mkdir(parents=True);records=data[(typ,split)]
  if not records:raise ValueError('empty task split')
  (d/'data.json').write_text(json.dumps(records,ensure_ascii=False)+'\n')
counts={t:{s:{'available':eligible[(t,s)],'retained':assigned[(t,s)],'discarded':lost[(t,s)],'images':len({x['image'] for x in data[(t,s)]})} for s in ('train','validation','test')} for t,_ in order}
manifest={'version':1,'seed':20261104,'order':[x for x,_ in order],'task_names':[f'MedicalVQA_{n}' for _,n in order],'rule':'Reuse existing source/modality-stratified image split; assign each image to one observed question type using seeded uniform selection; discard other-type questions for that image. No image path shared across tasks or splits.','patient_study_disjoint_verified':False,'cross_source_near_duplicates_verified':False,'counts':counts,'available_records':sum(eligible.values()),'retained_records':sum(assigned.values()),'discarded_records':sum(lost.values())}
(stage/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');stage.rename(dest)
print(json.dumps(manifest,indent=2))

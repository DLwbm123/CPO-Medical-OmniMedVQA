"""Build reproducible CPO tasks from the official open OmniMedVQA release.

Runs on pro5000 with python3 - < this_file. The output is a versioned,
image-grouped research split, not a patient-disjoint or official benchmark split.
"""
import collections
import json
import os
import random
import unicodedata
from pathlib import Path

ROOT = Path(os.environ['OMNIMEDVQA_ROOT'])
RAW = ROOT / 'data/OmniMedVQA'
DEST = ROOT / 'cpo_modality_v1'
STAGE = ROOT / '.cpo_modality_v1.preparing'
SEED = 20261103
MODES = {
    'CT(Computed Tomography)': 'CT',
    'MR (Mag-netic Resonance Imaging)': 'MRI',
    'X-Ray': 'XRay',
    'ultrasound': 'Ultrasound',
    'Fundus Photography': 'Fundus',
    'OCT (Optical Coherence Tomography': 'OCT',
    'Dermoscopy': 'Dermoscopy',
    'Microscopy Images': 'Microscopy',
}
ORDER = ['CT', 'MRI', 'XRay', 'Ultrasound', 'Fundus', 'OCT', 'Dermoscopy', 'Microscopy']


def normalized(value):
    return ' '.join(unicodedata.normalize('NFKC', str(value)).casefold().split())


def main():
    if DEST.exists() or STAGE.exists():
        raise FileExistsError('Output or staging directory already exists; refusing to overwrite')
    all_rows = []
    ids = set()
    image_modality = {}
    corrected = []
    qafiles = sorted((RAW / 'QA_information/Open-access').glob('*.json'))
    if not qafiles:
        raise FileNotFoundError('No open-access annotations found')
    for f in qafiles:
        for row in json.loads(f.read_text(encoding='utf-8')):
            ident = row['question_id']
            if ident in ids:
                raise ValueError(f'Duplicate question ID: {ident}')
            ids.add(ident)
            raw_modality = row.get('modality_type', '')
            if not raw_modality and row['dataset'] == 'Chest CT Scan' and ident == 'Chest CT Scan_0000':
                raw_modality = 'CT(Computed Tomography)'
                corrected.append({'id': ident, 'field': 'modality_type', 'value': 'CT',
                                  'reason': 'same source and image as Chest CT Scan_0139'})
            if raw_modality not in MODES:
                raise ValueError(f'Unknown modality: {ident}: {raw_modality!r}')
            modality = MODES[raw_modality]
            image = row['image_path']
            if not image.startswith('Images/') or not (RAW / image).is_file():
                raise ValueError(f'Invalid or missing image: {ident}: {image}')
            prior = image_modality.setdefault(image, modality)
            if prior != modality:
                raise ValueError(f'Image assigned to multiple modalities: {image}')
            options = [row['option_' + letter] for letter in 'ABCD' if 'option_' + letter in row]
            if len(options) < 2 or not all(isinstance(o, str) and o.strip() for o in options):
                raise ValueError(f'Invalid options: {ident}')
            matches = [i for i, opt in enumerate(options) if normalized(opt) == normalized(row['gt_answer'])]
            if len(matches) != 1:
                raise ValueError(f'Answer must uniquely match an option: {ident}')
            if not isinstance(row['question'], str) or not row['question'].strip():
                raise ValueError(f'Question missing: {ident}')
            all_rows.append((row, modality, options, chr(ord('A') + matches[0])))

    # Stratify within (modality, original source), but assign images, not QA rows.
    by_source = collections.defaultdict(set)
    for row, modality, _, _ in all_rows:
        by_source[(modality, row['dataset'])].add(row['image_path'])
    rng = random.Random(SEED)
    assignments = {}
    for key in sorted(by_source):
        images = sorted(by_source[key])
        rng.shuffle(images)
        n = len(images)
        # The smallest source groups need at least one training image.
        n_test = int(n * 0.1)
        n_val = int(n * 0.1)
        for idx, image in enumerate(images):
            split = 'test' if idx < n_test else 'validation' if idx < n_test + n_val else 'train'
            previous = assignments.setdefault(image, split)
            if previous != split:
                raise ValueError(f'Conflicting source assignment for image: {image}')

    data = collections.defaultdict(list)
    counts = collections.Counter()
    groups = collections.defaultdict(set)
    source_counts = collections.Counter()
    for row, modality, options, letter in all_rows:
        split = assignments[row['image_path']]
        question = row['question'].strip() + '\n' + '\n'.join(
            f'{chr(65+i)}. {option.strip()}' for i, option in enumerate(options))
        result = {'id': row['question_id'], 'image': row['image_path'][len('Images/'):],
                  'conversations': [{'from': 'human', 'value': question},
                                    {'from': 'gpt', 'value': letter}],
                  'source_dataset': row['dataset'], 'question_type': row.get('question_type', ''),
                  'original_answer': row['gt_answer']}
        data[(modality, split)].append(result)
        counts[(modality, split)] += 1
        groups[(modality, split)].add(row['image_path'])
        source_counts[(modality, row['dataset'], split)] += 1

    for mode in ORDER:
        sets = [groups[(mode, split)] for split in ('train', 'validation', 'test')]
        if any(not s for s in sets) or any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError(f'Empty or overlapping image groups for {mode}')
    STAGE.mkdir()
    for mode in ORDER:
        task = STAGE / ('MedicalVQA_' + mode)
        task.mkdir()
        os.symlink('../../data/OmniMedVQA/Images', task / 'images')
        for split in ('train', 'validation', 'test'):
            directory = task / 'jsons' / split
            directory.mkdir(parents=True)
            (directory / 'data.json').write_text(json.dumps(data[(mode, split)], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    report = {
        'version': 1, 'seed': SEED, 'task_order': ['MedicalVQA_' + m for m in ORDER],
        'split': 'source-stratified shuffled distinct image paths; approximately 80/10/10 by images',
        'sampling_unit': 'source image path; patient/study identity not reliably available',
        'patient_disjoint_verified': False, 'cross_source_duplicate_images_checked': False,
        'source': 'official open-access OmniMedVQA release', 'records': len(all_rows),
        'unique_image_paths': len(assignments), 'image_files_reused_via_symlinks': True,
        'modality_corrections': corrected,
        'counts': {m: {s: {'qa': counts[(m, s)], 'image_paths': len(groups[(m, s)])}
                       for s in ('train', 'validation', 'test')} for m in ORDER},
        'source_counts': [{'modality': m, 'source': d, 'split': s, 'qa': n}
                          for (m, d, s), n in sorted(source_counts.items())],
    }
    (STAGE / 'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (STAGE / 'README.md').write_text('''# OmniMedVQA modality continual-learning data v1

See manifest.json for seed, task order, counts, source breakdown and the single missing-modality correction. Each task has jsons/train, jsons/validation and jsons/test, and an images symlink to the original dataset. Records follow the CPO medical adapter format; answer letters uniquely match original options.

The split groups by original image path and stratifies by modality and source. Patient/study IDs are not established for all sources, and cross-source duplicate or near-duplicate images have not been checked. This is not a patient-disjoint claim. Some images derive from upstream benchmark test sets; training on these data can invalidate comparisons against those upstream benchmarks. This is a newly constructed research split, not the original OmniMedVQA or Med-R1 evaluation protocol. Do not combine its test data with independently partitioned Med-R1 data without overlap auditing.

All samples come from the open-access portion; follow each underlying dataset's license. The baseline sequence is a proposal recorded for reproducibility, not evidence of real temporal arrival. No model training or evaluation was performed in creating these files.
''', encoding='utf-8')
    STAGE.rename(DEST)
    print(json.dumps({'destination': str(DEST), 'records': len(all_rows), 'counts': report['counts']}, ensure_ascii=False))


if __name__ == '__main__':
    main()

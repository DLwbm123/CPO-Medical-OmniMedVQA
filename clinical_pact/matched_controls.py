"""Same-start GSPO and unit-weight protection controls; not a method search."""
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoProcessor, set_seed
from trl import GRPOConfig
import yaml

from .core import TaskDualController
from .data import load_split
from .dev_stream import Images, reward, save_json
from .fast_rollout import FirstStageRolloutTrainer
from .gpu56_development import DevelopmentTrainer
from .strategy_cycle import StrategyCycle


class UnitWeightController(TaskDualController):
    """Unit coefficient by definition; no selection using observed performance."""
    def add_task(self, task_id):
        super().add_task(task_id)
        self.states[task_id].weight = 1.0

    def observe_uniform_estimates(self, estimates):
        super().observe_uniform_estimates(estimates)
        for task_id in estimates:
            self.states[task_id].weight = 1.0


class FixedPromptControlTrainer(DevelopmentTrainer):
    def __init__(self, *args, control_arm, **kwargs):
        if control_arm not in ('plain_gspo', 'fixed_margin'):
            raise ValueError('unknown frozen control arm')
        if bool(kwargs['old_records']) != (control_arm == 'fixed_margin'):
            raise ValueError('old-memory access must match the declared arm')
        super().__init__(*args, **kwargs)
        if control_arm == 'fixed_margin':
            self.controller = UnitWeightController()
            self.controller.add_task('old')

    def _generate_and_score_completions(self, inputs):
        # The historical adaptive stage used empty cards throughout. Controls
        # have fixed canonical prompts and never invoke strategy selection.
        return FirstStageRolloutTrainer._generate_and_score_completions(self, inputs)


def main():
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    set_seed(100)
    root, out, reference = (Path(os.environ[key]) for key in ('ROOT', 'OUT', 'REFERENCE'))
    protocol = json.loads((out / 'protocol.json').read_text())
    arm = os.environ['ARM']
    if protocol['arms'] != ['plain_gspo', 'fixed_margin'] or arm not in protocol['arms']:
        raise ValueError('arm differs from the frozen control protocol')
    index = protocol['arms'].index(arm) + 1
    destination = out / f'stage{index}'
    if (destination / 'training.json').exists():
        raise ValueError('completed control arm must not be rerun')
    source = reference / 'stage1/model'
    if not (source / 'config.json').is_file():
        raise ValueError('the shared historical first-stage model is required')
    records = []
    if arm == 'fixed_margin':
        records = torch.load(reference / 'private_memory.pt', map_location='cpu', weights_only=False)
        if len(records) != protocol['memory_n']:
            raise ValueError('memory denominator differs from the frozen comparison')
    templates = yaml.safe_load((Path(__file__).parent / 'prompts.yaml').read_text())
    processor = AutoProcessor.from_pretrained(root / 'models/base', local_files_only=True,
                                             min_pixels=128*28*28, max_pixels=512*28*28)
    processor.tokenizer.padding_side = 'left'
    rows = list(load_split(root / 'data/prepared/MedicalVQA_Pathology', 'train'))
    ds = {'bf16': {'enabled': True}, 'train_micro_batch_size_per_gpu': 'auto',
          'gradient_accumulation_steps': 'auto', 'train_batch_size': 'auto',
          'zero_optimization': {'stage': 3, 'overlap_comm': False, 'contiguous_gradients': True,
                                'reduce_bucket_size': 10000000, 'stage3_prefetch_bucket_size': 10000000,
                                'stage3_param_persistence_threshold': 100000,
                                'stage3_gather_16bit_weights_on_model_save': True}, 'gradient_clipping': 'auto'}
    args = GRPOConfig(output_dir=str(destination), max_steps=200,
                      per_device_train_batch_size=4, gradient_accumulation_steps=4,
                      steps_per_generation=1, num_generations=8, num_iterations=1,
                      learning_rate=5e-6, bf16=True, full_determinism=True,
                      gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
                      max_completion_length=128, importance_sampling_level='sequence', loss_type='grpo',
                      beta=0., temperature=1., repetition_penalty=1., seed=100, data_seed=100,
                      deepspeed=ds, report_to='none', logging_steps=1, save_strategy='no',
                      remove_unused_columns=False, dataloader_num_workers=0,
                      model_init_kwargs={'torch_dtype': 'bfloat16', 'attn_implementation': 'sdpa', 'local_files_only': True})
    trainer = FixedPromptControlTrainer(model=str(source), args=args, processing_class=processor,
                                       reward_funcs=reward, train_dataset=Images(rows, templates),
                                       strategy_cycle=StrategyCycle(('', '')), refresh_factory=None,
                                       old_records=records, processor=processor, control_arm=arm)
    if rank == 0:
        save_json(out / 'progress.json', {'phase': 'training', 'arm': arm, 'max_updates': 200})
    trainer.train()
    trainer.save_model(str(destination / 'model'))
    if rank == 0:
        save_json(destination / 'training.json', {'arm': arm, 'updates': trainer.state.global_step,
                  'logs': trainer.state.log_history, 'audits': trainer.audit_history,
                  'fixed_weight': 1.0 if records else None,
                  'old_memory_n': len(records), 'prompt': 'fixed canonical',
                  'scope': 'exploratory matched-start control; no in-arm exact resume',
                  'cost_basis': 'same new-data updates; auxiliary forwards differ and must be reported'})
    trainer.accelerator.wait_for_everyone()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()

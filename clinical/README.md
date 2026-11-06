# Cross-dataset implementation record

These are the actual cross-dataset implementation files, saved separately from the original modality implementation: `train_grpo.py`, `compute_importance_mask.py`, `eval_torch.py`, `reward_funcs_medical.py`, `train_sft.py`, and the corrected `sft_dataset.py`. They depend on the existing `src` package and the experiment environment; this copy is an implementation record, not a standalone verified end-to-end reproduction. The experiment used a shared unmasked first-stage checkpoint and CPO masks for subsequent stages. Data conversion, data acquisition, environment setup and orchestration for this cross-dataset run are not yet packaged for public reproduction. Dataset images, question/answer records, predictions, model weights, private paths and logs are not distributed.

The SFT loader correction changes iteration over image path dictionary keys to its values when the question has no image placeholder. The run failed at the beginning of stage 2 before any optimizer update, was repaired and resumed from stage 1 under its original time and disk budgets.

`prompts_reasoning.yaml` and `eval_reasoning_prompt.py` record the later step-by-step prompt experiment. The instruction does not establish that generated reasoning is valid. A double-backslash typo was corrected during the counted preflight before formal training; the reported runs use the corrected template.

## Validation finding and conservative evaluation entry

The historical evaluator is retained as an implementation record. A subsequent same-example validation found that its batch-size-4 generation is not equivalent to batch size 1 in the recorded environment; see the research progress report. `eval_torch_safe.py` and `eval_reasoning_prompt_safe.py` are separate conservative entry points: they explicitly use the clinical normalized exact-match scorer and reject batch sizes other than 1. They do not reproduce the historical batch-4 protocol. The direct-answer entry was smoke-tested on two validation examples; the reasoning entry and end-to-end data preparation remain unverified. Their per-question prediction files must remain private.

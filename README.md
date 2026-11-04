# CPO on sequential medical-image VQA

This repository records an eight-stage, single-seed adaptation of Continual Policy Optimization (CPO) to the open-image portion of OmniMedVQA. The tasks are CT, MRI, X-ray, ultrasound, fundus, OCT, dermoscopy and microscopy. It provides the adapted code and aggregate results, not the third-party images, questions, predictions or model weights.

The complete stage-by-task accuracy matrix and its limitations are in [RESULTS.md](RESULTS.md). Machine-readable metrics and the frozen run protocol are in `results/`. The completed paired GSPO control and its limitations are reported in [GSPO_CONTROL_RESULTS.md](GSPO_CONTROL_RESULTS.md).

The first CPO stage is GSPO with an empty mask. Subsequent stages use a cumulative per-tensor top-10% nonzero parameter-movement mask and globally normalized masked L1, with coefficient 100. To avoid storing redundant reference-weight files, the DDP trainer captures the protected reference coordinates from the immediately preceding model checkpoint when it initializes. The task-end mask code operates on full model weights. Answers are scored by a shared training/evaluation rule accepting a boxed answer or an explicit leading option letter.

## Data and attribution

The implementation builds on the [CPO authors' code](https://github.com/MaolinLuo/CPO), released under the MIT license reproduced in `LICENSE`. The medical benchmark comes from [OmniMedVQA](https://huggingface.co/datasets/foreverbeliever/OmniMedVQA); consult the upstream and individual source-dataset licenses before obtaining or using it. The extracted open images and QA files are expected under `OMNIMEDVQA_ROOT/data/OmniMedVQA`. Run `OMNIMEDVQA_ROOT=/path/to/OmniMedVQA python medical/prepare_omnimed.py` to generate the image-path-grouped split and task manifest. This is a new research split and not the benchmark's original evaluation protocol.

The split is not established to be patient- or study-disjoint. Cross-source duplicate images and upstream training/test overlap remain unverified. Some sources may contain upstream benchmark test images. Do not claim patient-independent clinical generalization from these metrics.

## Environment and execution

`requirements-recorded.txt` records the principal versions used, not a complete lockfile. The original run used CUDA-enabled PyTorch 2.8.0, two 72 GB Blackwell GPUs, DDP, Transformers 4.56.1 and TRL 0.26.2. It differs from the original CPO paper, which used Qwen3-VL and different infrastructure. Qwen2.5-VL-3B-Instruct model weights are required separately.

`scripts/supervisor.py` and `scripts/worker.py` are the original run orchestration adapted to environment-variable paths. The supervisor uses GPUs 6 and 7 as in the recorded run; adjust the device indices and availability check for another machine before use. Set `CPO_PROJECT_ROOT` to this repository, `CPO_DATA_ROOT` to the prepared `cpo_modality_v1` directory, `CPO_BASE_MODEL` to the local model directory, `CPO_RUN_DIR` to a new empty output directory, and `CPO_OPS_ROOT` to a private writable directory containing a copy of `scripts/worker.py`. Create a `logs` directory under the run directory before launching the supervisor.

The run enforces a 23h55m wall-time cutoff and 40 GB on-disk guard with output headroom. As explicitly chosen for this experiment, it retains the first and latest task model and removes intermediate models only after the successor stage is evaluated and its mask created; it retains all metrics. Use a new output path and understand this retention behavior before running. No optimizer checkpoint is saved, so an interrupted stage cannot be resumed with the same optimizer state.

These scripts were syntax-checked after making their paths portable, but the portable wrapper has not been rerun end-to-end. The recorded results come from the original server run. The training implementation was checked with a two-GPU diagnostic, synthetic masked-gradient checks, checkpoint-to-mask generation and sampled image/answer loading; these checks do not establish clinical validity.

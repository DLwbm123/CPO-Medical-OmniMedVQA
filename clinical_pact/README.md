# PACT-Med (implementation in progress)

The [matched trainer, no-memory control and experiment summary](NO_MEMORY_RESULT_ZH.md) reports the final comparison. In the single-seed MedBook→PathVQA study, the no-memory repeat reproduced the historical GSPO's 32.29%, versus 39.89% for semantic-margin replay, 36.75% for four-choice CE, and 31.97% for four-choice KL. The [retained and unreplayed memory probe](MEMORY_ATTRIBUTION_ZH.md) reports paired old-task transitions.

Standalone research components; historical CPO and frozen experiments are unchanged. `core.py` is the provided numerical prototype, `memory.py` implements eligible immutable references and bounded per-task balanced selection, `grouping.py` validates 4/2/2 same-question rollouts and their prompt identity, and `scoring.py` matches the existing clinical normalized exact-match scorer. The supplied method design is kept locally; it is not included in this code release.

**A bounded development trainer has run.** `gpu56_development.py` completed real-model two-GPU MedBook→PathVQA training with 200 optimizer updates per stage, eligible immutable references, distributed strategy refresh and adaptive replay loss. See [GPU56_DEVELOPMENT_RESULT.md](GPU56_DEVELOPMENT_RESULT.md) for the retained OOM, boundary restart, invalid strategy proposals and replay-sampling deviation. `run_stream.py` is a placeholder, and the bounded studies use their own entry points. The replay arms access old training examples.

Run lightweight checks from the repository root with `python3 -m unittest discover -s clinical_pact -p 'test_*.py'`. PyTorch is required; Trainer adapter tests additionally need the pinned TRL/Transformers environment. Synthetic tests are separate from real-model and scientific validation. The development driver only supports completed-stage boundary restart, not exact mid-stage optimizer/sampler recovery. Generic runtime-state helpers and compressed ZeRO backend controls do not remove that restriction.

The four-rank synthetic check validates group advantage, task synchronization and a single combined DDP gradient; it does not validate the actual Qwen model, medical data, rollout generation, gradient accumulation or ZeRO. See four_rank_synthetic_result.json.

A bounded real-model driver (`dev_stream.py`) has now completed a four-GPU two-stage engineering run. This does not open the full training CLI: see ENGINEERING_RESULT.md for exact scope, zero dual weight, missing resume/dynamic-card checks and preserved failed attempts.

The newer two-GPU driver is a bounded procedural-grammar strategy variant, not the original unrestricted free-text proposer. Its initial and recovery supervisor sources are included for the frozen development run. Use environment variables to supply storage paths and module names, with a neutral process entry; do not expose project names in running command lines. Shared models and prepared data must already exist, and private references/checkpoints are not distributed here. No end-to-end packaged data preparation is claimed.

The completed exploratory checkpoint comparison is reported in [GPU56_VALIDATION_RESULT.md](GPU56_VALIDATION_RESULT.md): PathVQA validation increased from 1,003/6,259 to 1,787/6,259, while MedBook changed from 82/130 to 83/130. This is one run before/after new-task training, without a matched GSPO/replay control; it does not attribute benefit to PACT or establish successful dynamic strategy discovery.

The subsequent [matched-start control protocol](MATCHED_CONTROLS_PROTOCOL.md) freezes plain GSPO and unit-weight margin protection from the same initial checkpoint. These are new comparison conditions; no control efficacy result is implied by their code or registration.

The [retained-checkpoint evaluation](RETAINED_CHECKPOINT_SCORING_ZH.md) completes the plain-GSPO comparison and reports paired MedBook correctness, the upstream option-spacing audit and a text-only diagnostic. The observed single-seed trade-off is +3.85 points on MedBook and −3.74 points on PathVQA for PACT-Med relative to plain GSPO. The fixed-margin control has no validation result.

## v3 retention extensions

The [single-GPU seed-100 result](V3_SINGLE_GPU_RESULT_ZH.md) reports the frozen three-task run, its PathVQA answer bias, and the paired unreplayed-probe counts.

Use a fresh run directory and the [v3 protocol](v3_main_protocol.json) with `supervise_v3.py`. All extension switches default to off. For v3+R1+C3 set `r1_beta` to `0.04` and `fisher_tr` to `true`. Set either switch back to its default for the corresponding ablation. The v3 trainer already computes the replay gradient once at each optimizer boundary, so `replay_once_per_update` records the existing behavior without changing its arithmetic.

R1 uses the native TRL KL loss against the frozen stage-start model. Its coefficient is independent of group reward standardization. C3 estimates empirical Fisher on up to 512 correctly answered training examples, with a positive margin in all four MedBook rotations and binary PathVQA questions only. It selects the top 3.7% of physical coordinates globally and reanchors them at the latest task boundary. The Seed-100 GSPO-prime calibration publishes one `fisher_gamma` for reuse across later seeds. The quadratic penalty uses the FP32 master weights, public ZeRO gradient APIs and the maximum old-task dual weight.

`dynamic_sampling` allows up to two synchronized retries in the fixed dataset order using only binary verifier rewards. Keep it disabled until a collapse diagnostic calls for it. `batched_same_prompt_generation` can be enabled for a matched comparison once every arm uses the same generation rule. All ranks fall back to individual generation if any local prompts differ or need padding. Stage KL reports exact categorical KL on endpoint-generated prefixes, separately from TRL's training KL estimate. Enable `stage_kl_diagnostic` to score a baseline with all retention switches off.

The gate uses MedBook four-order validation accuracy, PathVQA yes/no validation accuracy, and macro correct-to-wrong rate on the unreplayed probes. First-stage MedBook results are cached together with the private paired-decision bits. Global gradient clipping remains unchanged. Step records include its scalar, the clipped retention-gradient norm and the effective replay and Fisher multipliers before AdamW.

Fisher estimation uses the same 338-question candidate pools as replay and excludes the frozen unreplayed probes. It differentiates the gold log-probability restricted to the answer candidates, records the accepted sample count, and uses a shared calibration against the saved BF16 control endpoint. Step diagnostics report separate projection, replay and Fisher norms. The single-GPU queue scores the frozen checkpoints and runs a matched control before routing to v3 repeats or a seed-100 R1+C3 run. Single-GPU R1+C3 training requires a completed actual-model one-update memory check.

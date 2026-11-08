# Two-GPU PACT development training: completed, efficacy untested

Seed 100 completed MedBook→PathVQA with **200 optimizer updates in each stage**. The boundary-restart attempt exited `[0, 0]`, both final model shard sets and processors were saved, and `completed.json` confirmed both stages. Total supervised elapsed time from the conservatively recorded original start was 13,928.97 seconds. Final allocated output storage was 15,052,242,944 bytes, under the unchanged 40GB guard and strict 50GB experiment limit. Models, source snapshots, private training references and original failure logs remain preserved privately.

| Training observation | MedBook | PathVQA |
| --- | ---: | ---: |
| Optimizer updates | 200 | 200 |
| First 10 logged updates: mean sampled reward | 42.1875% | 13.4375% |
| Last 10 logged updates: mean sampled reward | 66.5625% | 36.2500% |
| Scheduled strategy refreshes executed | 1 | 1 |
| Accepted local proposals | 0 | 0 |
| Nonempty selected cards after refresh | 0 | 0 |

These rewards describe sampled training trajectories on changing questions, including grouped trajectories from the same question. They are not held-out accuracy, independent statistical samples, proof of retention, or a comparison with GSPO/replay. No validation or test evaluation was performed in this run.

## What actually ran

The frozen protocol used full-parameter bf16 ZeRO-3 GSPO, eight trajectories per question, two ranks with local batch four, and four accumulated microsteps. Rank zero initially received four canonical trajectories; rank one received two trajectories per strategy card. Single-trajectory generation avoided the previously diagnosed mixed-length cached decoding issue. Deterministic settings were enabled. The proposer was a frozen local base model, receiving aggregate training feedback only and restricted to approved procedural sentence IDs. This explicitly bounded variant is not the original unrestricted free-text PACT-full method.

The 512 fixed training candidates produced **338 eligible immutable references** and 174 incorrect rejections. Each eligible reference passed actual greedy correctness, EOS, teacher-forced agreement and margin gates. Eligibility is a construction/filter result, not independent evidence that memory improves learning.

Both stage-local refreshes ran at update 100. Both proposals failed strict validation and were rejected without repairs or repeated proposal calls. The existing cards tied canonical on the 32-question training audit (MedBook 26 correct; PathVQA 10 correct), and selection fell back to two empty cards. Consequently, stage one used canonical prompts after its refresh and stage two began and remained with canonical prompts. The run does **not** demonstrate successful adaptive proposal generation or a useful new strategy card.

## Adaptive protection observations

Stage-two uniform old-memory audits were recorded at updates 0, 10, …, 190. The dual weight was zero at update 0 and nonzero at the remaining 19 audits, ending at **1.5865784877** at update 190. The sampled deficit reached 0.8815104365 at update 60 and was 0.12890625 at update 190. All 20 measurements are released in the aggregate JSON.

This establishes that the controller activated the auxiliary objective during actual training, unlike the earlier two-update engineering run with zero dual weight. It does not isolate auxiliary parameter gradients, demonstrate an accuracy gain, or causally attribute changes in sampled deficits to protection. Audits resample the same training-memory population, and no matched no-protection run was performed.

## Retained failure and boundary restart

The first attempt completed stage one and one stage-two optimizer update, then failed during the next ZeRO parameter gather with CUDA OOM; rank codes were `[1, -15]`. Static inspection found a retained stage-one `policy` reference after its context manager. Another process consuming approximately 3.45GiB was also reported in the OOM, so a unique causal attribution was not established.

The repair added an explicit completed-stage-one boundary entry and released that policy reference. New processes loaded the original stage-one model, cards and the same 338 references, and restarted stage two with the already specified fresh AdamW and seed-100 sampler. The failed stage-two update was discarded, retained in the failure record and recomputed; this was **not exact mid-stage resume**. No stage-one retraining occurred. Original source, recovery source, failure logs, root byte budget and absolute deadline were preserved. The conservative first-start epoch was recorded from the original supervisor file's timestamp, rather than a newly reset clock.

## Replay distinctness deviation

The frozen protocol described four distinct replay examples per optimizer update. Actual code drew two per rank **with replacement** and reused those draws across four accumulated microsteps. Deterministic reconstruction from the frozen code, seeds and memory size gives four distinct examples in **197/200 updates** and three in **3/200 updates**. This is reconstructed bookkeeping, not a historical per-sample trace. The deviation remains part of this result; the frozen run was not retrospectively corrected or relabeled.

The code implies 3,200 additional replay forwards and 640 old-memory audit forwards across both ranks over the completed stage-two attempt. These are loop-derived call counts, not profiler measurements, and exclude the discarded failed attempt and other rollout/strategy/memory-building work.

## Verification and remaining limits

Completion was checked using both zero worker exit codes, the completion artifact, both stage summaries and readable model shard indexes referencing existing shards and processor metadata. No checkpoint content hashing or fresh full-model inference was performed at delivery. Thirteen CPU tests covering allocation, dynamic refresh boundaries, strategy interfaces/state, ordinary runtime restoration and logit derivatives passed in the installed environment; these do not validate exact full-Trainer recovery or scientific benefit.

There is one seed, two tasks and no matched GSPO, replay-only or full-method comparison. Patient/case and near-duplicate isolation remain unverified. Stage-boundary artifacts support the documented fresh-optimizer transition; exact mid-stage Adam/sampler/rollout recovery is not implemented. The standard full research CLI and portable end-to-end data preparation are not completed. Training reward and a nonzero dual cannot establish medical knowledge, clinical validity or an improvement in retention/plasticity.

Released: source, frozen initial/recovery protocols and aggregate observations, including negative results. Excluded: medical images/questions, per-example predictions, private references, model weights, full raw logs and private server paths.

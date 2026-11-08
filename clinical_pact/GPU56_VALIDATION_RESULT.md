# Exploratory validation after two-task development training

Both fixed checkpoint workers completed with exit code 0. Each evaluated the full prepared MedBook validation split (130 questions) and PathVQA validation split (6,259 questions). Supervised validation runtime was 1,324.02 seconds. Evaluation used the frozen canonical prompt, one example per generation, each checkpoint's own processor, greedy decoding with a 128-token limit and the same normalized exact-match scorer. All evaluated responses terminated with the configured EOS. Model weights and hyperparameters were unchanged, and no test split was accessed.

| Validation task | After MedBook training | After subsequent PathVQA training | Change |
| --- | ---: | ---: | ---: |
| MedBook | 82/130 = 63.08% | 83/130 = 63.85% | +1 correct; +0.77 percentage points |
| PathVQA | 1,003/6,259 = 16.02% | 1,787/6,259 = 28.55% | +784 correct; +12.53 percentage points |

The descriptive positive observation is that new-task validation accuracy increased while aggregate old-task accuracy did not decrease on this small MedBook split. This is a **before/after observation of one development run**, not evidence that PACT outperforms GSPO or replay, or that its protective objective caused the observed retention. Learning PathVQA after training on PathVQA is expected to improve its accuracy; the incremental benefit of the studied mechanism remains unknown. The one-question MedBook difference is not a demonstrated improvement, and aggregate counts do not reveal which old examples were forgotten or rescued.

Both local strategy proposals in training were invalid and rejected, and both selected cards were empty after refresh. This validation therefore does not establish successful dynamic strategy discovery or PACT-full efficacy. The original OOM, stage-boundary restart, discarded single failed update and with-replacement replay deviation remain documented in [GPU56_DEVELOPMENT_RESULT.md](GPU56_DEVELOPMENT_RESULT.md).

This validation extension was frozen after seeing training results and used the same original experiment clock and root byte guard, with an additional four-hour deadline and 1GB output-subtree guard. The prepared validation data had also been used in earlier project diagnostics; it is not a fresh confirmatory evaluation. There is one seed, no matched control and no verified patient/case or near-duplicate separation. No confidence or causal claim is based on treating medical questions as independent patients.

Released artifacts are the frozen validation protocol, aggregate counts/timing/resource measurements and source. No medical questions/images, per-example predictions, private decision references or model weights are released. Necessary next evidence is a pre-frozen matched comparison from the same initial stage-one checkpoint, with the same new-task data, training budget and evaluator, reporting costs, negative results and original protocol deviations.

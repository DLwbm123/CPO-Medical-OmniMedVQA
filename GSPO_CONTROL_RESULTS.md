# Modality sequence: paired GSPO control

This is a single-seed, eight-stage comparison on the same image-path-grouped OmniMedVQA open-portion split, task sequence, full-model training settings, reward and test scorer. The control reuses the exact CT checkpoint and evaluation from the CPO run: the first stage has no CPO mask. In stages 2–8 the control uses the same GSPO objective without the CPO mask regularizer. See `results/protocol.json` and README for the CPO protocol. Both runs completed; no patient/study-level separation or cross-source near-duplicate screening was established. These are research metrics, not clinical generalization claims.

Final test accuracy (%, same test set per row):

| Task | n | CPO | GSPO | CPO − GSPO (pp) |
|---|---:|---:|---:|---:|
| CT | 1585 | 90.79 | 90.66 | +0.13 |
| MRI | 3194 | 94.71 | 95.74 | −1.03 |
| X-ray | 777 | 86.49 | 90.22 | −3.73 |
| Ultrasound | 1098 | 93.26 | 97.45 | −4.19 |
| Fundus | 529 | 88.66 | 89.22 | −0.57 |
| OCT | 453 | 96.69 | 97.35 | −0.66 |
| Dermoscopy | 648 | 80.09 | 81.94 | −1.85 |
| Microscopy | 553 | 86.44 | 88.25 | −1.81 |

MFT (mean immediate post-learning accuracy): CPO 90.21%, GSPO 91.56%. MFN (mean final accuracy): CPO 89.64%, GSPO 91.35%. MTA (mean of each task's accuracy over all stages after it is learned): CPO 90.01%, GSPO 91.30%. The average final-minus-immediate change over the seven tasks preceding microscopy is −0.65 pp for CPO and −0.24 pp for GSPO. X-ray final-minus-immediate is −3.47 pp for CPO and −1.93 pp for GSPO. These are descriptive, single-run differences; no statistical significance or causal explanation is established.

The control elapsed 25,332.8 seconds and reports 8,156,656,372 bytes at completion (shared first-stage checkpoint excluded). No raw medical images, source questions, model outputs, or patient information are published here.

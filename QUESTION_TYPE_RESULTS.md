# Question-type continual learning: CPO result

The completed run uses OmniMedVQA's open portion and a five-task sequence: modality recognition, anatomy identification, disease diagnosis, lesion grading and other biological attributes. It uses the existing source/modality-stratified image-path train/validation/test split. Each image is randomly assigned to one of its question types with seed 20261104, excluding questions for the image's other types. Of 88,996 original questions, 82,619 remain; 6,377 are excluded. `medical/prepare_question_type.py` reproduces the question-type assignment from the prepared modality split.

The run uses Qwen2.5-VL-3B-Instruct, full parameters, seed 42, 600 updates per task, GSPO with cumulative masked L1 regularization (lambda 100, top 10% per tensor) after the first stage. The shared first-stage checkpoint uses GSPO without a mask. After an insufficient-free-memory check before stage 3, the run continued from the completed stage 2 checkpoint on different GPUs; the original elapsed-time and storage guards remained in effect. First and final checkpoints and all metric files were retained. A paired unmasked GSPO control is running and no comparison outcome is asserted here.

Accuracy matrix (%) on the same held-out test set per task:

| Training stage | Modality | Anatomy | Diagnosis | Grading | Attributes |
|---|---:|---:|---:|---:|---:|
| Modality | 96.77 | — | — | — | — |
| Anatomy | 97.20 | 86.51 | — | — | — |
| Diagnosis | 97.41 | 85.78 | 84.53 | — | — |
| Grading | 97.31 | 84.98 | 84.79 | 94.12 | — |
| Attributes | 97.20 | 84.85 | 85.02 | 91.98 | 96.95 |

Test counts: 928, 1,505, 5,319, 187 and 295 respectively. Final macro accuracy 91.20%, immediate-after-learning macro accuracy 91.77%, and final-minus-immediate average over the four earlier tasks −0.72 percentage points. These are single-run descriptive results, not evidence that CPO caused the changes. Patient/study independence and near-duplicate separation have not been established; question type is confounded with source. The grading test has only 187 questions. No raw images, questions, per-example predictions, or weights are published.

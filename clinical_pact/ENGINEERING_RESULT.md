# Four-GPU real-model engineering run

Qwen2.5-VL-3B-Instruct, seed 100, MedBook → PathVQA training splits, four RTX PRO 5000 72GB GPUs. Both stages completed two optimizer updates; all four ranks exited 0. From 32 old-task candidates, 15 correct, EOS-complete and path-aligned references passed the margin gate, 15 were incorrect, and two were near ties.

The second stage executed teacher-forced replay and uniform audits, but its dual weight stayed zero (final EMA deficit 0.040625 below budget 0.05). Thus this run **does not establish nonzero auxiliary-gradient behavior on the actual model**, forgetting prevention, scientific benefit, or complete PACT-Med readiness. It uses fixed seed cards without dynamic refresh, 32 rather than 512 memory candidates, and an audit every optimizer update. Full optimizer/RNG recovery was not exercised. The run saves both stage models privately and accesses no test records.

Two prior attempts on four 24GB RTX3090s failed before a completed update, first with foreach AdamW and then with foreach disabled. The user directed migration to the larger server; failures, original start clock, hard two-hour deadline and 46GB per-server write guard were retained. PyTorch changed from 2.6.0+cu124 to 2.8.0+cu128 with migration; this is an engineering environment change, not a paired performance comparison. Old artifacts are retained.

Source entry: dev_stream.py. It is intentionally a bounded engineering driver, not the complete research training CLI. Private images, sample text, per-example outputs, stored memories, model weights and private logs are excluded from this release. Next gates include actual nonzero auxiliary gradients, accumulation/resume equivalence, and full dynamic-card integration before a frozen efficacy study.

# Experiment log

Append-only. Each run records:
- date/time
- Git commit
- architecture config
- dataset and revision
- seed
- Kaggle accelerator
- optimizer and learning-rate schedule
- train/validation loss
- Python execution-eval results
- feedback dataset revision
- checkpoint path
- promotion/rejection decision

## Run 000 — scaffold
Status: code path built and local forward/backward smoke test passed.

## Run 001 — MicroCoder-284K / 5,000 batches
Date: 2026-09-19
Status: completed successfully on Kaggle CUDA.
Architecture: 283,584 parameters; dim=64; 5 layers; 8 heads; FFN=192; context=768; byte vocabulary=260.
Data: youmyron/bits-py-dataset, 8,143 train / 1,003 validation / 962 test source rows.
Training command: 5,000 batches, batch_size=32, grad_accum=2, AdamW lr=3e-4.
Observed training loss: 4.7524 @ 50 -> 1.1319 @ 5,000.
Observed internal validation loss: 2.7977 @ 500 -> 1.1753 @ 5,000.
Independent post-run sample over 64 blocks from the official test file: approximate loss 1.1843.
Functional generation check: model produces Python-like lexical structure but simple generated programs remain mostly syntactically invalid or semantically wrong. This checkpoint is therefore a successful training baseline, not yet a usable Python coder.
Data-split caveat: Run 001's internal validation was created by random-splitting byte blocks from the training file rather than using the dataset's official validation split. Future runs must use the official validation/test files for promotion decisions.
Checkpoint storage:
- Google Drive /gpt/TinyPyGPT/model/microcoder_284k_step5000_best.pt
- Google Drive /gpt/TinyPyGPT/model/microcoder_284k_step5000_last.pt
Promotion decision: DO NOT promote as coding-capable. Preserve as baseline; next run must fix held-out evaluation and increase capacity.

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


## Run 002 — MicroCoder-1.1M / official evaluation
Date: 2026-09-19
Status: completed successfully on Kaggle CUDA.
Architecture: 1,099,648 parameters; dim=128; 5 layers; 8 heads; FFN=384; context=768; byte vocabulary=260.
Data: youmyron/bits-py-dataset, official 8,143 train / 1,003 validation / 962 test source rows.
Training command: 5,000 mini-batches, batch_size=16, grad_accum=4 = 1,250 optimizer updates, AdamW lr=3e-4.
Observed train loss: 4.6698 @ batch 50 -> 0.8503 @ batch 5,000.
Official validation loss: 2.7054 @ 500 -> 1.0161 @ 5,000.
Official test loss: 1.0164.
Generalization note: validation and test losses match closely, with no obvious held-out-loss overfitting signal.
Functional generation check: FAILED. The model generates Python-like surface structure but does not reliably solve even add(a,b), is_even(n), reverse_string(s), factorial(n), or pandas CSV reading. Several outputs are syntactically invalid.
Diagnosis: the current streaming LM objective spends capacity predicting user prompts, markup, prose and assistant answers indiscriminately. For a ~1M model this is inefficient for the target skill. The next experiment must keep the same parameter count and change the curriculum/objective before scaling.
Checkpoint storage:
- Google Drive /gpt/TinyPyGPT/model/microcoder_1m_step5000_best.pt
- Google Drive /gpt/TinyPyGPT/model/microcoder_1m_step5000_last.pt
Promotion decision: preserve as language-model baseline, DO NOT promote as Python coder.
Next experiment: executable-code curriculum -> answer-only masked SFT -> official held-out loss + deterministic syntax/function checks.

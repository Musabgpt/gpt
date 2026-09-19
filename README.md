# MicroCoder / TinyPyGPT

A tiny, continually-trainable Python-specialist language-model research project.

## Current core
- MicroCoder-75K: 75,232 parameters for smoke tests.
- MicroCoder-284K: 283,584 parameters for the default Kaggle experiments.
- Byte-level tokenizer to preserve Python syntax without a large vocabulary table.
- Decoder-only Transformer with RMSNorm, RoPE, causal attention, SwiGLU, and tied embeddings.
- Feedback updates support SFT and DPO-style pairwise optimization.

## Storage
- GitHub: source, project memory, experiment metadata, decisions.
- Google Drive /gpt/TinyPyGPT: data snapshots, notebooks, reports, checkpoints.
- Kaggle: GPU training and evaluation.

See PROJECT_MEMORY.md for the canonical design and checkpoint-promotion rules.

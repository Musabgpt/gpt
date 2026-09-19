# MicroCoder Project Memory

## Goal
Build the smallest practical open research system that becomes increasingly specialized in Python, can train on Kaggle, accepts continual feedback updates, and is designed for later Android/Termux deployment.

## Canonical architecture
MicroCoder is a tiny LLaMA-style decoder-only Transformer: RMSNorm, RoPE, causal self-attention, SwiGLU, tied token embeddings, and a byte-level vocabulary. The initial smoke-test configuration has 75,232 parameters; the default research configuration has 283,584 parameters. The architecture is deliberately scalable without changing the data or feedback pipeline.

## Open-source reference
The verified tiny reference family is ANRGUSC/tlm26 (MIT), which reports models from 78K to about 1M parameters. MicroCoder keeps that tiny-model research spirit but uses a project-specific byte tokenizer and feedback loop.

## Data
Primary HF source: youmyron/bits-py-dataset (MIT, Python instruction-following corpus). Secondary source: iamtarun/python_code_instructions_18k_alpaca.

## Continual learning
New corrections are appended to JSONL as chosen/rejected examples. Run SFT for single corrected outputs or DPO for pairwise preferences. Mix trusted historical examples into every feedback cycle and evaluate before promoting a checkpoint.

## Promotion rule
Never call a checkpoint better because train loss fell. Promote only when held-out loss and Python task tests improve without unacceptable regression on the anchor suite.

## Storage roles
- GitHub Musabgpt/gpt: source code, project memory, experiment metadata, decisions.
- Google Drive /gpt/TinyPyGPT: data snapshots, notebooks, reports, checkpoints.
- Kaggle: GPU training/execution environment.

## Current constraints
A sub-million-parameter model is a research vehicle, not a frontier coding model. If tests plateau, scale the same architecture to 1M, 3M, 14M+ rather than hiding the capacity limit.

## Tool/research status
- Claude-Fable-5.1.md is the behavioral workflow reference for this project.
- Consensus is available for literature grounding.
- Scite MCP reached its monthly call limit on 2026-09-19; use Consensus plus primary papers until reset.
- Hugging Face, Google Drive, GitHub, and Notion are connected.

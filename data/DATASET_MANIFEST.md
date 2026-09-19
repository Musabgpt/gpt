# Training Data Manifest

## Primary dataset
- Repository: `youmyron/bits-py-dataset`
- Revision label: `v2026-03-30-r2`
- License: MIT
- Format: JSONL chat messages
- Total rows: 10,108
- Train: 8,143
- Validation: 1,003
- Test: 962
- Published package size: about 14 MB
- Expected column: `messages`

Topic coverage reported by the dataset card:
- general_python: 7,459
- pandas_numpy: 1,506
- data_pipelines: 1,357
- ml_scripts: 483
- async_fastapi: 77

Source mix:
- 5,000 examples from `python_code_instructions_18k_alpaca.jsonl`
- 5,000 from `self_oss_instruct_sc2_exec_filter_50k.jsonl`
- 108 curated domain-gap examples

## Secondary dataset
- Repository: `iamtarun/python_code_instructions_18k_alpaca`
- 18,612 rows
- Parquet file: `data/train-00000-of-00001-8b6e212f3e1ece96.parquet`
- Size: 11.4 MB
- SHA256 reported by Hugging Face: `bbc6fa528fcc17a845e863da8652bc02c2523373cd8cc0b9af1ffb374e9ae378`

## Reproducible acquisition
Run `scripts/fetch_hf_data.py` or the Kaggle notebook. The project intentionally keeps dataset provenance separate from model checkpoints.

## Current transfer note
The connected Hugging Face integration can inspect repository metadata but does not expose raw file download bytes to this chat runtime. The project therefore stores the exact source/revision and a Kaggle downloader in Google Drive. The first Kaggle execution materializes the full JSONL split files. Do not claim a full Drive data snapshot exists until those bytes have been copied and verified.

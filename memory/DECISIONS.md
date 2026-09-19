# Decisions

- 2026-09-19: Use a byte-level vocabulary to preserve Python syntax without a large embedding table.
- 2026-09-19: Keep a ~75K smoke-test model and a ~284K default research model; scale only when evaluation proves capacity is the bottleneck.
- 2026-09-19: Continual learning uses feedback SFT/DPO plus replay of trusted prior examples.
- 2026-09-19: GitHub is canonical project memory; Google Drive is artifact/data/checkpoint storage.
- 2026-09-19: Use youmyron/bits-py-dataset as the primary Python instruction source.
- 2026-09-19: Checkpoint promotion is evaluation-gated, never train-loss-gated.

from pathlib import Path
import json
from datasets import load_dataset

REPO = "youmyron/bits-py-dataset"
OUT = Path("data/hf_bits_py")
OUT.mkdir(parents=True, exist_ok=True)

bundle = load_dataset(REPO)
for split, ds in bundle.items():
    target = OUT / f"{split}.jsonl"
    with target.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(split, len(ds), target)

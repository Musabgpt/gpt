from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path


FENCE = re.compile(r"```(?:python|py)?\s*\n?(.*?)```", re.I | re.S)


def extract_code(answer: str) -> str | None:
    candidates = FENCE.findall(answer)
    candidates.append(answer)
    for candidate in candidates:
        code = candidate.strip()
        if not code:
            continue
        try:
            ast.parse(code)
            return code
        except (SyntaxError, ValueError):
            continue
    return None


def prompt_from_row(row: dict) -> str | None:
    messages = row.get("messages")
    if isinstance(messages, list):
        parts = [str(m.get("content", "")) for m in messages if m.get("role") in {"user", "system"}]
        text = "\n\n".join(x for x in parts if x.strip()).strip()
        return text or None
    return row.get("instruction") or row.get("prompt") or row.get("question")


def answer_from_row(row: dict) -> str | None:
    messages = row.get("messages")
    if isinstance(messages, list):
        assistants = [str(m.get("content", "")) for m in messages if m.get("role") == "assistant"]
        return assistants[-1] if assistants else None
    return row.get("output") or row.get("answer") or row.get("response") or row.get("completion")


def load_valid(path: Path):
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = prompt_from_row(row)
            answer = answer_from_row(row)
            if not prompt or not answer:
                continue
            code = extract_code(str(answer))
            if not code:
                continue
            pbytes = len(prompt.encode("utf-8"))
            cbytes = len(code.encode("utf-8"))
            if pbytes > 1200 or cbytes < 8 or cbytes > 2400:
                continue
            fp = hashlib.sha256((prompt.strip().lower() + "\0" + code.strip()).encode("utf-8")).hexdigest()
            items.append({"prompt": prompt.strip(), "chosen": code.strip(), "fp": fp})
    return items


def dedupe(items, blocked=None):
    blocked = blocked or set()
    seen = set()
    out = []
    for item in items:
        fp = item["fp"]
        if fp in blocked or fp in seen:
            continue
        seen.add(fp)
        out.append(item)
    return out


def write_split(name: str, items: list[dict], out: Path):
    sft = out / f"sft_{name}.jsonl"
    code = out / f"code_{name}.jsonl"
    with sft.open("w", encoding="utf-8") as fs, code.open("w", encoding="utf-8") as fc:
        for item in items:
            fs.write(json.dumps({"prompt": item["prompt"], "chosen": item["chosen"]}, ensure_ascii=False) + "\n")
            fc.write(json.dumps({"text": item["chosen"] + "\n"}, ensure_ascii=False) + "\n")
    return sft, code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = {
        "train": load_valid(Path(args.train)),
        "validation": load_valid(Path(args.validation)),
        "test": load_valid(Path(args.test)),
    }

    # Protect held-out integrity: any exact prompt+code example in validation/test
    # is removed from training. Then deduplicate each split internally.
    validation = dedupe(raw["validation"])
    test = dedupe(raw["test"], {x["fp"] for x in validation})
    blocked = {x["fp"] for x in validation} | {x["fp"] for x in test}
    train = dedupe(raw["train"], blocked)

    final = {"train": train, "validation": validation, "test": test}
    for name, items in final.items():
        sft, code = write_split(name, items, out)
        print(name, len(items), sft, code)

    manifest = {
        name: len(items) for name, items in final.items()
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

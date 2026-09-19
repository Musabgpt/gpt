from __future__ import annotations

import argparse
import ast
import torch

from model import MicroCoder, MicroCoderConfig
from data import BOS, EOS, encode_bytes, decode_bytes


TASKS = [
    ("add", "Write a Python function add(a, b) that returns a + b."),
    ("is_even", "Write a Python function is_even(n) that returns True for even integers."),
    ("reverse_string", "Write a Python function reverse_string(s) that returns the reversed string."),
    ("factorial", "Write a Python function factorial(n) using a loop."),
    ("pandas_csv", "Write Python code to read a CSV file using pandas."),
]


@torch.no_grad()
def generate(model, prompt, max_new_tokens=220):
    prefix = f"### Instruction:\n{prompt}\n\n### Response:\n"
    ids = torch.tensor([[BOS, *encode_bytes(prefix)]], dtype=torch.long, device=next(model.parameters()).device)
    prefix_len = ids.size(1)
    for _ in range(max_new_tokens):
        x = ids[:, -model.cfg.max_seq_len:]
        logits, _ = model(x)
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt.item()) == EOS:
            break
    out = ids[0, prefix_len:].tolist()
    if EOS in out:
        out = out[:out.index(EOS)]
    return decode_bytes(out).strip()


def has_named_function(tree, name):
    return any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name for n in ast.walk(tree))


def heuristic(task, tree):
    if task == "add":
        if not has_named_function(tree, "add"):
            return False
        return any(isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add) for n in ast.walk(tree))
    if task == "is_even":
        if not has_named_function(tree, "is_even"):
            return False
        return any(isinstance(n, ast.Mod) for n in ast.walk(tree))
    if task == "reverse_string":
        if not has_named_function(tree, "reverse_string"):
            return False
        return any(isinstance(n, ast.Subscript) for n in ast.walk(tree))
    if task == "factorial":
        if not has_named_function(tree, "factorial"):
            return False
        return any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(tree)) and any(
            isinstance(n, (ast.Mult, ast.Mult)) for n in ast.walk(tree)
        )
    if task == "pandas_csv":
        has_pandas = any(
            isinstance(n, ast.Import) and any(a.name == "pandas" for a in n.names)
            or isinstance(n, ast.ImportFrom) and n.module == "pandas"
            for n in ast.walk(tree)
        )
        has_read_csv = any(isinstance(n, ast.Attribute) and n.attr == "read_csv" for n in ast.walk(tree))
        return has_pandas and has_read_csv
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = MicroCoderConfig(**ckpt["config"])
    model = MicroCoder(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    syntax_ok = 0
    heuristic_ok = 0
    for task, prompt in TASKS:
        text = generate(model, prompt)
        print("\n" + "=" * 80)
        print(task, prompt)
        print(text)
        try:
            tree = ast.parse(text)
            syntax = True
            syntax_ok += 1
        except SyntaxError as exc:
            tree = None
            syntax = False
            print("SYNTAX_ERROR:", exc)
        semantic = bool(tree is not None and heuristic(task, tree))
        heuristic_ok += int(semantic)
        print(f"SYNTAX_VALID={syntax} HEURISTIC_PASS={semantic}")

    print(f"SUMMARY syntax_valid={syntax_ok}/{len(TASKS)} heuristic_pass={heuristic_ok}/{len(TASKS)}")


if __name__ == "__main__":
    main()

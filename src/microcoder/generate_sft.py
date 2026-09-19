from __future__ import annotations

import argparse
import torch

from model import MicroCoder, MicroCoderConfig
from data import BOS, EOS, encode_bytes, decode_bytes


@torch.no_grad()
def greedy(model, ids, max_new_tokens):
    for _ in range(max_new_tokens):
        x = ids[:, -model.cfg.max_seq_len:]
        logits, _ = model(x)
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt.item()) == EOS:
            break
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = MicroCoderConfig(**ckpt["config"])
    model = MicroCoder(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    prefix = f"### Instruction:\n{args.prompt}\n\n### Response:\n"
    ids = torch.tensor([[BOS, *encode_bytes(prefix)]], dtype=torch.long, device=device)
    out = greedy(model, ids, args.tokens)[0].tolist()
    generated = out[len(ids[0]):]
    if EOS in generated:
        generated = generated[:generated.index(EOS)]
    print(decode_bytes(generated))


if __name__ == "__main__":
    main()

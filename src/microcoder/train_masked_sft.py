from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import MicroCoder, MicroCoderConfig
from instruction_data import MaskedSFTDataset


def save_checkpoint(path, model, optimizer, batch_step, optimizer_step, cfg, best_val):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "batch_step": batch_step,
        "optimizer_step": optimizer_step,
        "step": batch_step,
        "config": asdict(cfg),
        "best_val": best_val,
        "objective": "answer_only_masked_sft",
    }, path)


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            n = int((y != -100).sum().item())
            total_loss += float(loss.item()) * n
            total_tokens += n
    model.train()
    return total_loss / max(1, total_tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--init-checkpoint", required=True)
    ap.add_argument("--out", default="outputs_sft")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MicroCoderConfig(**json.loads(Path(args.config).read_text()))
    model = MicroCoder(cfg).to(device)

    init = torch.load(args.init_checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(init["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")

    train_ds = MaskedSFTDataset(args.train, seq_len=cfg.max_seq_len)
    val_ds = MaskedSFTDataset(args.validation, seq_len=cfg.max_seq_len)
    test_ds = MaskedSFTDataset(args.test, seq_len=cfg.max_seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        pin_memory=torch.cuda.is_available(), num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=1)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=1)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=0.05
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    iterator = iter(train_loader)
    optimizer.zero_grad(set_to_none=True)
    best_val = float("inf")
    bad_evals = 0
    optimizer_step = 0
    final_batch = 0

    print(
        f"device={device} params={model.num_parameters():,} "
        f"train_examples={len(train_ds)} val_examples={len(val_ds)} test_examples={len(test_ds)}"
    )

    for batch_step in range(1, args.steps + 1):
        final_batch = batch_step
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)

        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        (loss / args.grad_accum).backward()

        did_update = batch_step % args.grad_accum == 0
        if did_update:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

        if batch_step % 50 == 0:
            print(
                f"batch_step={batch_step} optimizer_step={optimizer_step} "
                f"answer_loss={loss.item():.4f}"
            )

        if did_update and batch_step % args.eval_every == 0:
            val = evaluate(model, val_loader, device)
            print(f"batch_step={batch_step} optimizer_step={optimizer_step} OFFICIAL_MASKED_VAL={val:.4f}")
            if val < best_val - 0.001:
                best_val = val
                bad_evals = 0
                save_checkpoint(out / "best.pt", model, optimizer, batch_step, optimizer_step, cfg, best_val)
            else:
                bad_evals += 1
                print(f"no_improve={bad_evals}/{args.patience}")
                if bad_evals >= args.patience:
                    print("EARLY_STOP")
                    break

        if did_update and batch_step % args.save_every == 0:
            save_checkpoint(out / "last.pt", model, optimizer, batch_step, optimizer_step, cfg, best_val)

    save_checkpoint(out / "last.pt", model, optimizer, final_batch, optimizer_step, cfg, best_val)

    best_path = out / "best.pt"
    if best_path.exists():
        best = torch.load(best_path, map_location="cpu")
        model.load_state_dict(best["model"])
    test_loss = evaluate(model, test_loader, device)
    print(f"FINAL_MASKED_TEST_LOSS={test_loss:.4f}")


if __name__ == "__main__":
    main()

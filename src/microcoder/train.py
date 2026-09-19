from __future__ import annotations

import argparse, json, random
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from model import MicroCoder, MicroCoderConfig
from data import JsonlByteDataset


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
    }, path)


def evaluate(model, loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in loader:
            _, loss = model(x.to(device), y.to(device))
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


def make_loader(path, cfg, seed, batch_size, shuffle, workers):
    ds = JsonlByteDataset(path, seq_len=cfg.max_seq_len, seed=seed)
    return ds, DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        num_workers=workers,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--val-data", default=None)
    ap.add_argument("--test-data", default=None)
    ap.add_argument("--config", default="configs/microcoder_75k.json")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--steps", type=int, default=20000, help="Mini-batch steps")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MicroCoderConfig(**json.loads(Path(args.config).read_text()))
    model = MicroCoder(cfg).to(device)

    train_ds = JsonlByteDataset(args.data, seq_len=cfg.max_seq_len, seed=args.seed)
    if args.val_data:
        val_ds = JsonlByteDataset(args.val_data, seq_len=cfg.max_seq_len, seed=args.seed + 1)
    else:
        val_n = max(1, int(len(train_ds) * 0.02))
        train_n = len(train_ds) - val_n
        train_ds, val_ds = random_split(
            train_ds, [train_n, val_n],
            generator=torch.Generator().manual_seed(args.seed)
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        pin_memory=torch.cuda.is_available(), num_workers=2
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        pin_memory=torch.cuda.is_available(), num_workers=1
    )

    test_loader = None
    if args.test_data:
        test_ds = JsonlByteDataset(args.test_data, seq_len=cfg.max_seq_len, seed=args.seed + 2)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            pin_memory=torch.cuda.is_available(), num_workers=1
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        betas=(0.9, 0.95), weight_decay=0.1
    )

    start_batch = 0
    optimizer_step = 0
    best_val = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_batch = ckpt.get("batch_step", ckpt.get("step", 0))
        optimizer_step = ckpt.get("optimizer_step", start_batch // max(1, args.grad_accum))
        best_val = ckpt.get("best_val", best_val)

    print(
        f"device={device} params={model.num_parameters():,} "
        f"train_blocks={len(train_ds)} val_blocks={len(val_ds)} "
        f"official_val={bool(args.val_data)}"
    )

    iterator = iter(train_loader)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    optimizer.zero_grad(set_to_none=True)

    for batch_step in range(start_batch + 1, args.steps + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            x, y = next(iterator)

        _, loss = model(x.to(device), y.to(device))
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
                f"train_loss={loss.item():.4f}"
            )

        if did_update and batch_step % args.eval_every == 0:
            val = evaluate(model, val_loader, device)
            print(f"batch_step={batch_step} optimizer_step={optimizer_step} val_loss={val:.4f}")
            if val < best_val:
                best_val = val
                save_checkpoint(
                    out / "best.pt", model, optimizer,
                    batch_step, optimizer_step, cfg, best_val
                )

        if did_update and batch_step % args.save_every == 0:
            save_checkpoint(
                out / "last.pt", model, optimizer,
                batch_step, optimizer_step, cfg, best_val
            )

    save_checkpoint(
        out / "last.pt", model, optimizer,
        args.steps, optimizer_step, cfg, best_val
    )

    if test_loader is not None:
        best_path = out / "best.pt"
        if best_path.exists():
            best = torch.load(best_path, map_location="cpu")
            model.load_state_dict(best["model"])
        test_loss = evaluate(model, test_loader, device)
        print(f"FINAL_OFFICIAL_TEST_LOSS={test_loss:.4f}")


if __name__ == "__main__":
    main()

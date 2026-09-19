from __future__ import annotations

import argparse, json, random
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from model import MicroCoder, MicroCoderConfig
from data import JsonlByteDataset


def save_checkpoint(path, model, optimizer, step, cfg, best_val):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "config": asdict(cfg),
        "best_val": best_val,
    }, path)


def evaluate(model, loader, device, max_batches=20):
    model.eval(); losses=[]
    with torch.no_grad():
        for i,(x,y) in enumerate(loader):
            if i >= max_batches: break
            _,loss=model(x.to(device),y.to(device))
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",required=True)
    ap.add_argument("--config",default="configs/microcoder_75k.json")
    ap.add_argument("--out",default="outputs")
    ap.add_argument("--steps",type=int,default=20000)
    ap.add_argument("--batch-size",type=int,default=32)
    ap.add_argument("--grad-accum",type=int,default=1)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--eval-every",type=int,default=500)
    ap.add_argument("--save-every",type=int,default=1000)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--resume",default=None)
    args=ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device="cuda" if torch.cuda.is_available() else "cpu"
    cfg=MicroCoderConfig(**json.loads(Path(args.config).read_text()))
    model=MicroCoder(cfg).to(device)
    dataset=JsonlByteDataset(args.data,seq_len=cfg.max_seq_len,seed=args.seed)
    val_n=max(1,int(len(dataset)*0.02)); train_n=len(dataset)-val_n
    train_ds,val_ds=random_split(dataset,[train_n,val_n],generator=torch.Generator().manual_seed(args.seed))
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,pin_memory=torch.cuda.is_available(),num_workers=2)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size,shuffle=False,num_workers=1)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(0.9,0.95),weight_decay=0.1)
    start_step,best_val=0,float("inf")
    if args.resume:
        ckpt=torch.load(args.resume,map_location="cpu")
        model.load_state_dict(ckpt["model"]); optimizer.load_state_dict(ckpt["optimizer"])
        start_step=ckpt.get("step",0); best_val=ckpt.get("best_val",best_val)

    print(f"device={device} params={model.num_parameters():,} train_blocks={train_n} val_blocks={val_n}")
    iterator=iter(train_loader); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step+1,args.steps+1):
        try: x,y=next(iterator)
        except StopIteration:
            iterator=iter(train_loader); x,y=next(iterator)
        _,loss=model(x.to(device),y.to(device)); (loss/args.grad_accum).backward()
        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step(); optimizer.zero_grad(set_to_none=True)
        if step % 50 == 0: print(f"step={step} train_loss={loss.item():.4f}")
        if step % args.eval_every == 0:
            val=evaluate(model,val_loader,device); print(f"step={step} val_loss={val:.4f}")
            if val < best_val:
                best_val=val; save_checkpoint(out/"best.pt",model,optimizer,step,cfg,best_val)
        if step % args.save_every == 0:
            save_checkpoint(out/"last.pt",model,optimizer,step,cfg,best_val)
    save_checkpoint(out/"last.pt",model,optimizer,args.steps,cfg,best_val)

if __name__=="__main__": main()

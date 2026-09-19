from __future__ import annotations

import argparse, json, random
from pathlib import Path
import torch
import torch.nn.functional as F
from model import MicroCoder, MicroCoderConfig
from data import encode_bytes, BOS, EOS


def seq_logprob(model, ids):
    x,y=ids[:,:-1],ids[:,1:]
    logits,_=model(x)
    return F.log_softmax(logits,dim=-1).gather(-1,y.unsqueeze(-1)).squeeze(-1).mean(-1)


def make_seq(prompt, answer, max_len, device):
    text=f"<user>\n{prompt}\n</user>\n<assistant>\n{answer}\n</assistant>\n"
    ids=[BOS,*encode_bytes(text),EOS][-max_len:]
    return torch.tensor(ids,dtype=torch.long,device=device).unsqueeze(0)


def load_rows(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--feedback",required=True)
    ap.add_argument("--replay",default=None,help="Optional trusted historical feedback JSONL")
    ap.add_argument("--replay-ratio",type=float,default=0.30)
    ap.add_argument("--out",default="outputs/feedback.pt")
    ap.add_argument("--mode",choices=["sft","dpo"],default="dpo")
    ap.add_argument("--steps",type=int,default=1000)
    ap.add_argument("--lr",type=float,default=5e-5)
    ap.add_argument("--beta",type=float,default=0.1)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()

    random.seed(args.seed); device="cuda" if torch.cuda.is_available() else "cpu"
    ckpt=torch.load(args.checkpoint,map_location="cpu")
    cfg=MicroCoderConfig(**ckpt["config"])
    model=MicroCoder(cfg).to(device); model.load_state_dict(ckpt["model"]); model.train()
    ref=None
    if args.mode=="dpo":
        ref=MicroCoder(cfg).to(device); ref.load_state_dict(ckpt["model"]); ref.eval()
        for p in ref.parameters(): p.requires_grad_(False)
    rows=load_rows(args.feedback)
    replay=load_rows(args.replay) if args.replay else []
    if not rows: raise ValueError("feedback file is empty")
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.01)

    for step in range(1,args.steps+1):
        use_replay=bool(replay) and random.random() < args.replay_ratio
        r=random.choice(replay if use_replay else rows)
        prompt=r.get("prompt") or r.get("instruction") or ""
        chosen_text=r.get("chosen") or r.get("output") or r.get("answer")
        if not chosen_text: continue
        chosen=make_seq(prompt,chosen_text,cfg.max_seq_len,device)
        weight=float(r.get("weight",1.0))
        rejected_text=r.get("rejected")
        if args.mode=="sft" or not rejected_text:
            _,loss=model(chosen[:,:-1],chosen[:,1:]); loss=loss*weight
        else:
            rejected=make_seq(prompt,rejected_text,cfg.max_seq_len,device)
            pi_c,pi_r=seq_logprob(model,chosen),seq_logprob(model,rejected)
            with torch.no_grad():
                ref_c,ref_r=seq_logprob(ref,chosen),seq_logprob(ref,rejected)
            logits=args.beta*((pi_c-pi_r)-(ref_c-ref_r))
            loss=-F.logsigmoid(logits).mean()*weight
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        if step%50==0: print(f"feedback_step={step} loss={loss.item():.4f} replay={use_replay}")

    Path(args.out).parent.mkdir(parents=True,exist_ok=True)
    torch.save({"model":model.state_dict(),"config":ckpt["config"],"parent":args.checkpoint,"mode":args.mode},args.out)
    print(f"saved={args.out}")

if __name__=="__main__": main()

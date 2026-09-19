from __future__ import annotations
import argparse, torch
from model import MicroCoder, MicroCoderConfig
from data import encode_bytes, decode_bytes, BOS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--prompt",required=True)
    ap.add_argument("--tokens",type=int,default=256)
    a=ap.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"
    ckpt=torch.load(a.checkpoint,map_location="cpu")
    cfg=MicroCoderConfig(**ckpt["config"])
    m=MicroCoder(cfg).to(device); m.load_state_dict(ckpt["model"]); m.eval()
    prompt=f"<user>\n{a.prompt}\n</user>\n<assistant>\n"
    ids=torch.tensor([[BOS,*encode_bytes(prompt)]],device=device)
    out=m.generate(ids,max_new_tokens=a.tokens)
    print(decode_bytes(out[0].tolist()))

if __name__=="__main__": main()

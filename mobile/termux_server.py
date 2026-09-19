"""Minimal local HTTP wrapper for later Android/Termux deployment."""
from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import sys

app = FastAPI(title="MicroCoder local API")

class Req(BaseModel):
    prompt: str
    tokens: int = 192

@app.post("/generate")
def generate(req: Req):
    cmd = [
        sys.executable, "src/microcoder/generate.py",
        "--checkpoint", "outputs/best.pt",
        "--prompt", req.prompt,
        "--tokens", str(req.tokens),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return {"text": p.stdout, "stderr": p.stderr}

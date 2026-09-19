from __future__ import annotations

import json
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset

PAD, BOS, EOS, SEP = 256, 257, 258, 259


def encode_bytes(text: str, add_bos=False, add_eos=False):
    ids = list(text.encode("utf-8", errors="replace"))
    if add_bos:
        ids.insert(0, BOS)
    if add_eos:
        ids.append(EOS)
    return ids


def decode_bytes(ids):
    return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")


def format_messages(messages):
    return "".join(
        f"<{m.get('role','user')}>\n{m.get('content','')}\n</{m.get('role','user')}>\n"
        for m in messages
    )


def row_to_text(row):
    if "messages" in row and isinstance(row["messages"], list):
        return format_messages(row["messages"])
    if "text" in row:
        return str(row["text"])
    instruction = row.get("instruction") or row.get("prompt") or row.get("question")
    answer = row.get("output") or row.get("answer") or row.get("response") or row.get("completion")
    extra = row.get("input")
    if instruction is not None and answer is not None:
        prompt = str(instruction)
        if extra:
            prompt += "\n\nInput:\n" + str(extra)
        return f"<user>\n{prompt}\n</user>\n<assistant>\n{answer}\n</assistant>\n"
    if "input" in row and "output" in row:
        return f"<user>\n{row['input']}\n</user>\n<assistant>\n{row['output']}\n</assistant>\n"
    return None


class JsonlByteDataset(Dataset):
    def __init__(self, path: str | Path, seq_len=512, seed=42):
        self.seq_len = seq_len
        texts = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                text = row_to_text(json.loads(line))
                if text:
                    texts.append(text)
        if not texts:
            raise ValueError(f"No usable rows found in {path}")
        rng = random.Random(seed)
        rng.shuffle(texts)
        stream = []
        for text in texts:
            stream.extend([BOS, *encode_bytes(text), EOS])
        self.tokens = torch.tensor(stream, dtype=torch.long)

    def __len__(self):
        return max(1, (len(self.tokens) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = (idx * self.seq_len) % max(1, len(self.tokens) - self.seq_len - 1)
        chunk = self.tokens[start:start + self.seq_len + 1]
        if len(chunk) < self.seq_len + 1:
            chunk = torch.cat([
                chunk,
                torch.full((self.seq_len + 1 - len(chunk),), PAD, dtype=torch.long)
            ])
        return chunk[:-1], chunk[1:]

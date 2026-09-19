from __future__ import annotations

import json
from pathlib import Path
import torch
from torch.utils.data import Dataset

from data import PAD, BOS, EOS, encode_bytes


class MaskedSFTDataset(Dataset):
    """One instruction per sample; loss is applied only to assistant answer bytes."""

    def __init__(self, path: str | Path, seq_len: int = 1536, max_prompt_bytes: int = 640):
        self.seq_len = seq_len
        self.max_prompt_bytes = min(max_prompt_bytes, max(64, seq_len // 2))
        self.rows = []
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                prompt = row.get("prompt") or row.get("instruction") or row.get("question")
                answer = row.get("chosen") or row.get("output") or row.get("answer")
                if prompt is not None and answer is not None:
                    self.rows.append((str(prompt), str(answer)))
        if not self.rows:
            raise ValueError(f"No usable SFT rows in {path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        prompt, answer = self.rows[idx]
        p = prompt.encode("utf-8", errors="replace")
        if len(p) > self.max_prompt_bytes:
            half = self.max_prompt_bytes // 2
            p = p[:half] + b"\n...[prompt truncated]...\n" + p[-half:]
            p = p[:self.max_prompt_bytes]

        prefix = b"### Instruction:\n" + p + b"\n\n### Response:\n"
        prefix_ids = [BOS, *prefix]
        max_total = self.seq_len + 1
        room = max_total - len(prefix_ids) - 1  # reserve EOS
        if room < 1:
            prefix_ids = prefix_ids[: max_total - 2]
            room = 1

        answer_ids = list(answer.encode("utf-8", errors="replace"))[:room]
        seq = prefix_ids + answer_ids + [EOS]

        input_ids = seq[:-1]
        labels = seq[1:]
        target_start = len(prefix_ids)
        mask_until = max(0, target_start - 1)
        labels[:mask_until] = [-100] * mask_until

        pad_n = self.seq_len - len(input_ids)
        if pad_n > 0:
            input_ids += [PAD] * pad_n
            labels += [-100] * pad_n
        else:
            input_ids = input_ids[: self.seq_len]
            labels = labels[: self.seq_len]

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

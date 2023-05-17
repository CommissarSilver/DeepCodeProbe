import numpy as np
import torch


def collator_fn(batch):
    tokens = [b["code_tokens"] for b in batch]
    cs = [b["c"] for b in batch]
    ds = [b["d"] for b in batch]
    us = [b["u"] for b in batch]

    batch_len_tokens = [len(m) for m in tokens]
    max_len_tokens = np.max(batch_len_tokens)

    cs = torch.tensor([c + [-1] * (max_len_tokens - 1 - len(c)) for c in cs])
    ds = torch.tensor([d + [-1] * (max_len_tokens - 1 - len(d)) for d in ds])
    us = torch.tensor([u + [-1] * (max_len_tokens - len(u)) for u in us])

    return ds, cs, us, torch.tensor(batch_len_tokens)

import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch_scatter import scatter_mean


def get_embeddings(all_inputs, model_name, model, layer):
    if model_name == "astnn":
        with torch.no_grad():
            _, embs = model.encode(all_inputs)

    return embs


def collator_fn(batch):
    original_code_string = [b["original_string"] for b in batch]

    cs = [b["c"] for b in batch]
    ds = [b["d"] for b in batch]
    us = [b["u"] for b in batch]

    batch_len_tokens = np.max([len(m) for m in ds])
    max_len_c_tokens = 0
    for children_tokens in cs:
        for child_tokens in children_tokens:
            max_len_c_tokens = max(max_len_c_tokens, len(child_tokens))

    ds = [d + [-1] * (batch_len_tokens - len(d)) for d in ds]

    all_cs = []
    for children_tokens in cs:
        all_cs_children = []
        for child_tokens in children_tokens:
            new_child = child_tokens + [-1] * (max_len_c_tokens - len(child_tokens))
            all_cs_children.append(new_child)
        all_cs.append(
            all_cs_children
            + [[-1] * max_len_c_tokens] * (batch_len_tokens - len(all_cs_children))
        )

    us = [u + [-1] * (batch_len_tokens - len(u)) for u in us]

    ds_tensor = torch.tensor(ds)
    cs_tensor = torch.tensor(all_cs)
    us_tensor = torch.tensor(us)
    
    return (
        ds_tensor,
        cs_tensor,
        us_tensor,
        torch.tensor(batch_len_tokens),
        original_code_string,
    )

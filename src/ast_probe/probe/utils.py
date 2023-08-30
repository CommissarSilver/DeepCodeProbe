import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch_scatter import scatter_mean


def get_embeddings_astnn(all_inputs, model, **kwargs):
    """
    function for getting the intermediate inputs of the ASTNN model

    Args:
        all_inputs (_type_): inputs to be fed into the model. this must be the original string of code
        model_name (_type_): name of the model. depracated. to be removed later.
        model (_type_): the model itself.

    Returns:
        _type_: the intermediate outputs of the model
    """
    with torch.no_grad():
        _, embs = model.encode(all_inputs)
    return embs


def get_embeddings_funcgnn(all_inputs, model, **kwargs):
    """
    function for getting the intermediate inputs of the FuncGNN model

    Args:
        all_inputs (_type_): inputs to be fed into the model. this must be the original string of code
        model_name (_type_): name of the model. depracated. to be removed later.
        model (_type_): the model itself.
        kwargs (_type_): include the max_len of inputs

    Returns:
        _type_: the intermediate outputs of the model

    """
    with torch.no_grad():
        # get the intermediate ouputs of the model
        embs = [model.encode(i) for i in all_inputs]
        # FuncGNN gives intemediate ouputs which are not a ll of the same size in their first dimension.
        # therefore, padding is required to make them all of the same size
        max_len = kwargs["max_len"]
        padded_tensor = torch.zeros(len(embs), max_len, embs[0].size(1))
        # for each tensor, if its smaller than max_len, fill it with -1s
        for i, tensor in enumerate(embs):
            padded_tensor[i, : tensor.size(0), :] = tensor

        padded_tensor[padded_tensor == 0] = -1
        embs = padded_tensor

    return embs


def collator_fn_astnn(batch):
    """
    collator function for ASTNN

    Args:
        batch (_type_): the batch of codes to be processed into d,c,u tuple

    """
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


def collator_fn_funcgnn(batch):
    cs = [b["c"] for b in batch]
    ds = [b["d"] for b in batch]
    us = [b["u"] for b in batch]

    batch_len_tokens = np.max([len(m) for m in ds])
    batch_len_cs = np.max([len(m) for m in cs])

    ds = [d + [-1] * (batch_len_tokens - len(d)) for d in ds]
    cs = [c + [[-1, -1]] * (batch_len_cs - len(c)) for c in cs]
    us = [u + [-1] * (batch_len_tokens - len(u)) for u in us]

    ds_tensor = torch.tensor(ds)
    cs_tensor = torch.tensor(cs)
    us_tensor = torch.tensor(us)

    original_batch = batch

    return (
        ds_tensor,
        cs_tensor,
        us_tensor,
        torch.tensor(batch_len_tokens),
        batch,
    )

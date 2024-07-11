import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

# from torch_scatter import scatter_mean


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
        embs, _ = model.encode(all_inputs)
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
        if embs[0].size(0) != 64:
            max_len = 29
            padded_tensor = torch.zeros(len(embs), max_len, embs[0].size(1))
            # for each tensor, if its smaller than max_len, fill it with -1s
            for i, tensor in enumerate(embs):
                padded_tensor[i, : tensor.size(0), :] = tensor

            padded_tensor[padded_tensor == 0] = -1
            embs = padded_tensor
        
        elif embs[0].size(0) == 64:
            embs = torch.stack(embs)
            embs=embs.squeeze(-1)

    return embs


def get_embeddings_sum_tf(all_inputs, model, **kwargs):
    from summarization_tf.src.utils import pad_tensor

    ys, cs, hs = [], [], []
    with torch.no_grad():
        for input in all_inputs:
            y, (c, h) = model.encode(input)
            ys.append(y[0])
            cs.append(c)
            hs.append(h)
    ys, cs, hs = pad_tensor(ys)[0], torch.stack(cs), torch.stack(hs)
    embs = torch.cat((ys, cs, hs), dim=1)
    # # Create a zero tensor with the shape of the padding needed
    # #! SUM-TF uses 96 as the max length of the code
    # zero_padding = torch.zeros((embs.shape[0], 96 - embs.shape[1], embs.shape[2]))

    # # Concatenate zero tensor to the original tensor along the second dimension
    # embs = torch.cat((embs, zero_padding), dim=1)
    return embs


def get_embeddings_code_sum_drl(all_inputs, model, **kwargs):
    with torch.no_grad():
        embs = model.initialize(all_inputs, False)[1][3]

    # zero_padding = torch.zeros((embs.shape[0], 141 - embs.shape[1], embs.shape[2]))
    # embs = torch.cat((embs, zero_padding), dim=1)
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

    batch_len_tokens_d = np.max([len(m) for m in ds])
    batch_len_tokens_c = np.max([len(m) for m in cs])
    batch_len_tokens_u = np.max([len(m) for m in us])

    ds = [d + [-1] * (batch_len_tokens_d - len(d)) for d in ds]
    cs = [c + [-1] * (batch_len_tokens_c - len(c)) for c in cs]
    us = [u + [-1] * (batch_len_tokens_u - len(u)) for u in us]

    ds_tensor = torch.tensor(ds)
    cs_tensor = torch.tensor(cs)
    us_tensor = torch.tensor(us)

    return (
        ds_tensor,
        cs_tensor,
        us_tensor,
        torch.tensor(batch_len_tokens_c),
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


def collator_fn_sum_tf(batch):
    original_code_string = [b["y_raw"] for b in batch]
    tree_tensors = [b["tree_tensor"] for b in batch]

    cs = [b["c"] for b in batch]
    ds = [b["d"] for b in batch]
    us = [b["u"] for b in batch]

    batch_len_tokens = np.max([len(m) for m in cs])
    batch_len_cs = np.max([len(m) for m in cs])
    batch_len_us = np.max([len(m) for m in us])

    ds = [d + [[-1, -1]] * (batch_len_tokens - len(d)) for d in ds]
    cs = [c + [-1] * (batch_len_cs - len(c)) for c in cs]
    us = [u + [-1] * (batch_len_us - len(u)) for u in us]

    ds_tensor = torch.tensor(ds)
    cs_tensor = torch.tensor(cs)
    us_tensor = torch.tensor(us)

    return (
        ds_tensor,
        cs_tensor,
        us_tensor,
        torch.tensor(batch_len_tokens),
        tree_tensors,
    )


def collator_fun_code_sum_drl(batch):
    pass

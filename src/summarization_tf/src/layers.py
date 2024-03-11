"""layers"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchvision.transforms.functional import pad
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence, pad_packed_sequence


from utils import *


class TreeEmbeddingLayer(nn.Module):
    def __init__(self, dim_E, in_vocab):
        super(TreeEmbeddingLayer, self).__init__()
        self.E = nn.Parameter(torch.rand(in_vocab, dim_E)).to('cuda:0')

    def forward(self, x):
        x_len = [xx.shape[0] for xx in x]
        x_tensor = [torch.from_numpy(xi).to("cuda:0") for xi in x]
        x_concat = torch.cat(x_tensor, dim=0)
        ex = torch.index_select(self.E, 0, x_concat)
        exs = torch.split(ex, x_len, dim=0)
        return exs

class TreeEmbeddingLayerTreeBase(nn.Module):
    def __init__(self, dim_E, in_vocab):
        super(TreeEmbeddingLayerTreeBase, self).__init__()
        self.E = nn.Parameter(torch.Tensor(in_vocab, dim_E))
        init.uniform_(self.E, -0.05, 0.05)  # Similar to tf.keras.initializers.RandomUniform()

    def forward(self, roots):
        return [self.apply_single(root) for root in roots]

    def apply_single(self, root):
        labels = traverse_label(root)
        embedded = self.E[labels]
        new_nodes = self.Node2TreeLSTMNode(root, parent=None)
        for rep, node in zip(embedded, traverse(new_nodes)):
            node.h = rep
        return new_nodes

    def Node2TreeLSTMNode(self, node, parent):
        children = [self.Node2TreeLSTMNode(c, node) for c in node.children]
        return TreeLSTMNode(node.label, parent=parent, children=children, num=node.num)

class ChildSumLSTMLayerWithEmbedding(nn.Module):
    def __init__(self, in_vocab, dim_in, dim_out):
        super(ChildSumLSTMLayerWithEmbedding, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.E = nn.Parameter(torch.Tensor(in_vocab, dim_in))
        init.uniform_(self.E, -0.05, 0.05)  # Similar to tf.keras.initializers.RandomUniform()
        self.U_f = nn.Linear(dim_in, dim_out, bias=False)
        self.U_iuo = nn.Linear(dim_in, dim_out * 3, bias=False)
        self.W = nn.Linear(dim_in, dim_out * 4)
        self.h_init = nn.Parameter(torch.zeros(1, dim_out))
        self.c_init = nn.Parameter(torch.zeros(1, dim_out))
    
    @staticmethod
    def get_nums(roots):
        res = [[x.num for x in n.children] if n.children != [] else [0] for n in roots]
        max_len = max([len(x) for x in res])
        padded_res = torch.nn.utils.pad_sequence(res, batch_first=True, padding_value=-1)
        padded_res = padded_res[:, :max_len]  # Truncate or pad sequences to max_len
        return padded_res
    
    def get_sorted_depthes(roots):
        depthes = [x[1] for x in sorted(depth_split_batch2(roots).items(), key=lambda x: -x[0])]  # list of list of Nodes
        return depthes
    def forward(self, roots):
        depthes = self.get_sorted_depthes(roots)
        indices = [self.get_nums(nodes) for nodes in depthes]

        h_tensor = self.h_init
        c_tensor = self.c_init
        for indice, nodes in zip(indices, depthes):
            x = self.E[[node.label for node in nodes]]  # [nodes, dim_in]
            h_tensor, c_tensor = self.apply(x, h_tensor, c_tensor, indice, nodes)
            h_tensor = torch.cat([self.h_init, h_tensor], dim=0)
            c_tensor = torch.cat([self.c_init, c_tensor], dim=0)
        return depthes[-1]

    def apply(self, x, h_tensor, c_tensor, indice, nodes):
        mask_bool = indice != -1
        mask = mask_bool.float()  # [batch, child]

        h = h_tensor[mask_bool, indice[mask_bool]]  # [nodes, child, dim]
        c = c_tensor[mask_bool, indice[mask_bool]]
        h_sum = (h * mask.unsqueeze(-1)).sum(dim=1)  # [nodes, dim_out]

        W_x = self.W(x)  # [nodes, dim_out * 4]
        W_f_x = W_x[:, :self.dim_out * 1]  # [nodes, dim_out]
        W_i_x = W_x[:, self.dim_out * 1:self.dim_out * 2]
        W_u_x = W_x[:, self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[:, self.dim_out * 3:]

        branch_f_k = self.U_f(h.view(-1, h.shape[-1])).view(*h.shape)
        branch_f_k = torch.sigmoid(W_f_x.unsqueeze(1) + branch_f_k)
        branch_f = (branch_f_k * c * mask.unsqueeze(-1)).sum(dim=1)  # [node, dim_out]

        branch_iuo = self.U_iuo(h_sum)  # [nodes, dim_out * 3]
        branch_i = torch.sigmoid(branch_iuo[:, :self.dim_out * 1] + W_i_x)  # [nodes, dim_out]
        branch_u = torch.tanh(branch_iuo[:, self.dim_out * 1:self.dim_out * 2] + W_u_x)
        branch_o = torch.sigmoid(branch_iuo[:, self.dim_out * 2:] + W_o_x)

        new_c = branch_i * branch_u + branch_f  # [node, dim_out]
        new_h = branch_o * torch.tanh(new_c)  # [node, dim_out]

        for n, c, h in zip(nodes, new_c, new_h):
            n.c = c
            n.h = h

        return new_h, new_c

class ChildSumLSTMLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ChildSumLSTMLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.U_f = nn.Linear(dim_in, dim_out, bias=False)
        self.U_iuo = nn.Linear(dim_in, dim_out * 3, bias=False)
        self.W = nn.Linear(dim_in, dim_out * 4)
        self.h_init = nn.Parameter(torch.zeros(1, dim_out))
        self.c_init = nn.Parameter(torch.zeros(1, dim_out))

    def forward(self, tensor, indices):
        h_tensor = self.h_init
        c_tensor = self.c_init
        res_h, res_c = [], []
        for indice, x in zip(indices, tensor):
            h_tensor, c_tensor = self.apply(x, h_tensor, c_tensor, indice)
            h_tensor = torch.cat([self.h_init, h_tensor], dim=0)
            c_tensor = torch.cat([self.c_init, c_tensor], dim=0)
            res_h.append(h_tensor[1:, :])
            res_c.append(c_tensor[1:, :])
        return res_h, res_c

    def apply(self, x, h_tensor, c_tensor, indice):
        mask_bool = indice != -1
        mask = mask_bool.float()  # [batch, child]

        h = h_tensor[mask_bool, indice[mask_bool]]  # [nodes, child, dim]
        c = c_tensor[mask_bool, indice[mask_bool]]
        h_sum = (h * mask.unsqueeze(-1)).sum(dim=1)  # [nodes, dim_out]

        W_x = self.W(x)  # [nodes, dim_out * 4]
        W_f_x = W_x[:, :self.dim_out * 1]  # [nodes, dim_out]
        W_i_x = W_x[:, self.dim_out * 1:self.dim_out * 2]
        W_u_x = W_x[:, self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[:, self.dim_out * 3:]

        branch_f_k = self.U_f(h.view(-1, h.shape[-1])).view(*h.shape)
        branch_f_k = torch.sigmoid(W_f_x.unsqueeze(1) + branch_f_k)
        branch_f = (branch_f_k * c * mask.unsqueeze(-1)).sum(dim=1)  # [node, dim_out]

        branch_iuo = self.U_iuo(h_sum)  # [nodes, dim_out * 3]
        branch_i = torch.sigmoid(branch_iuo[:, :self.dim_out * 1] + W_i_x)  # [nodes, dim_out]
        branch_u = torch.tanh(branch_iuo[:, self.dim_out * 1:self.dim_out * 2] + W_u_x)
        branch_o = torch.sigmoid(branch_iuo[:, self.dim_out * 2:] + W_o_x)

        new_c = branch_i * branch_u + branch_f  # [node, dim_out]
        new_h = branch_o * torch.tanh(new_c)  # [node, dim_out]

        return new_h, new_c

class ChildSumLSTMLayerTreeBase(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ChildSumLSTMLayerTreeBase, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.U_f = nn.Linear(dim_in, dim_out, bias=False)
        self.U_iuo = nn.Linear(dim_in, dim_out * 3, bias=False)
        self.W = nn.Linear(dim_in, dim_out * 4)
        self.h_init = nn.Parameter(torch.zeros(1, dim_out))
        self.c_init = nn.Parameter(torch.zeros(1, dim_out))

    @staticmethod
    def get_nums(roots):
        res = [[x.num for x in n.children] if n.children != [] else [0] for n in roots]
        max_len = max([len(x) for x in res])
        res = torch.tensor(res, dtype=torch.int32)
        return F.pad_sequence(res, batch_first=True, padding_value=-1, total_length=max_len)

    def forward(self, roots):
        depthes = [x[1] for x in sorted(depth_split_batch2(
            roots).items(), key=lambda x:-x[0])]  # list of list of Nodes
        indices = [self.get_nums(nodes) for nodes in depthes]

        h_tensor = self.h_init
        c_tensor = self.c_init
        for indice, nodes in zip(indices, depthes):
            x = torch.stack([node.h for node in nodes])  # [nodes, dim_in]
            h_tensor, c_tensor = self.apply(x, h_tensor, c_tensor, indice, nodes)
            h_tensor = torch.cat([self.h_init, h_tensor], dim=0)
            c_tensor = torch.cat([self.c_init, c_tensor], dim=0)
        return depthes[-1]

    def apply(self, x, h_tensor, c_tensor, indice, nodes):
        mask_bool = indice != -1
        mask = mask_bool.float()  # [batch, child]

        h = h_tensor[mask_bool, indice[mask_bool]]  # [nodes, child, dim]
        c = c_tensor[mask_bool, indice[mask_bool]]
        h_sum = (h * mask.unsqueeze(-1)).sum(dim=1)  # [nodes, dim_out]

        W_x = self.W(x)  # [nodes, dim_out * 4]
        W_f_x = W_x[:, :self.dim_out * 1]  # [nodes, dim_out]
        W_i_x = W_x[:, self.dim_out * 1:self.dim_out * 2]
        W_u_x = W_x[:, self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[:, self.dim_out * 3:]

        branch_f_k = self.U_f(h.view(-1, h.shape[-1])).view(*h.shape)
        branch_f_k = torch.sigmoid(W_f_x.unsqueeze(1) + branch_f_k)
        branch_f = (branch_f_k * c * mask.unsqueeze(-1)).sum(dim=1)  # [node, dim_out]

        branch_iuo = self.U_iuo(h_sum)  # [nodes, dim_out * 3]
        branch_i = torch.sigmoid(branch_iuo[:, :self.dim_out * 1] + W_i_x)  # [nodes, dim_out]
        branch_u = torch.tanh(branch_iuo[:, self.dim_out * 1:self.dim_out * 2] + W_u_x)
        branch_o = torch.sigmoid(branch_iuo[:, self.dim_out * 2:] + W_o_x)

        new_c = branch_i * branch_u + branch_f  # [node, dim_out]
        new_h = branch_o * torch.tanh(new_c)  # [node, dim_out]

        for n, c, h in zip(nodes, new_c, new_h):
            n.c = c
            n.h = h

        return new_h, new_c

class NaryLSTMLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(NaryLSTMLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.U_f1 = nn.Linear(dim_in, dim_out, bias=False)
        self.U_f2 = nn.Linear(dim_in, dim_out, bias=False)
        self.U_iuo = nn.Linear(dim_in, dim_out * 3, bias=False)
        self.W = nn.Linear(dim_in, dim_out * 4)
        self.h_init = nn.Parameter(torch.zeros(1, dim_out))
        self.c_init = nn.Parameter(torch.zeros(1, dim_out))

    def forward(self, tensor, indices):
        h_tensor = self.h_init
        c_tensor = self.c_init
        res_h, res_c = [], []
        for indice, x in zip(indices, tensor):
            h_tensor, c_tensor = self.apply(x, h_tensor, c_tensor, indice)
            h_tensor = torch.cat([self.h_init, h_tensor], dim=0)
            c_tensor = torch.cat([self.c_init, c_tensor], dim=0)
            res_h.append(h_tensor[1:, :])
            res_c.append(c_tensor[1:, :])
        return res_h, res_c

    def apply(self, x, h_tensor, c_tensor, indice):
        mask_bool = indice != -1

        h = h_tensor[mask_bool, indice[mask_bool]]  # [nodes, child, dim]
        c = c_tensor[mask_bool, indice[mask_bool]]

        W_x = self.W(x)  # [nodes, dim_out * 4]
        W_f_x = W_x[:, :self.dim_out * 1]  # [nodes, dim_out]
        W_i_x = W_x[:, self.dim_out * 1:self.dim_out * 2]
        W_u_x = W_x[:, self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[:, self.dim_out * 3:]

        if h.shape[1] <= 1:
            h = torch.cat([h, torch.zeros_like(h)], 1)  # [nodes, 2, dim]
            c = torch.cat([c, torch.zeros_like(c)], 1)

        h_concat = h.view(h.shape[0], -1)

        branch_f1 = self.U_f1(h_concat)
        branch_f1 = torch.sigmoid(W_f_x + branch_f1)
        branch_f2 = self.U_f2(h_concat)
        branch_f2 = torch.sigmoid(W_f_x + branch_f2)
        branch_f = branch_f1 * c[:, 0] + branch_f2 * c[:, 1]

        branch_iuo = self.U_iuo(h_concat)  # [nodes, dim_out * 3]
        branch_i = torch.sigmoid(branch_iuo[:, :self.dim_out * 1] + W_i_x)   # [nodes, dim_out]
        branch_u = torch.tanh(branch_iuo[:, self.dim_out * 1:self.dim_out * 2] + W_u_x)
        branch_o = torch.sigmoid(branch_iuo[:, self.dim_out * 2:] + W_o_x)

        new_c = branch_i * branch_u + branch_f  # [node, dim_out]
        new_h = branch_o * torch.tanh(new_c)  # [node, dim_out]

        return new_h, new_c

class BiLSTM_(nn.Module):
    def __init__(self, dim, return_seq=False):
        super(BiLSTM_, self).__init__()
        self.dim = dim
        self.c_init_f = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.h_init_f = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.c_init_b = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.h_init_b = nn.Parameter(torch.randn(1, dim) * 0.01)
        self.Cell_f = nn.LSTM(dim, dim, bidirectional=False)
        self.Cell_b = nn.LSTM(dim, dim, bidirectional=False)
        self.fc = nn.Linear(dim * 2, dim, bias=False)  # Concatenating outputs of both directions
        self.return_seq = return_seq

    def forward(self, x, length):
        '''x: [batch, length, dim]'''
        batch = x.shape[0]
        h_0_f = self.h_init_f.expand(-1, batch, -1).contiguous()  # [1, batch, dim]
        c_0_f = self.c_init_f.expand(-1, batch, -1).contiguous()
        h_0_b = self.h_init_b.expand(-1, batch, -1).contiguous()
        c_0_b = self.c_init_b.expand(-1, batch, -1).contiguous()

        packed_x = pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        ys_packed, (_, _) = self.Cell_f(packed_x, (h_0_f, c_0_f))
        ys_packed, (_, _) = self.Cell_b(ys_packed, (h_0_b, c_0_b))
        ys, _ = pad_packed_sequence(ys_packed, batch_first=True)

        if self.return_seq:
            return self.fc(ys)
        else:
            state_concat = torch.cat([ys[:, -1, :self.dim], ys[:, 0, self.dim:]], dim=-1)
            return self.fc(state_concat)

class BiLSTM(nn.Module):
    def __init__(self, dim, return_seq=False):
        super(BiLSTM, self).__init__()
        self.dim = dim
        self.h_init = nn.Parameter(torch.randn(1, dim))
        self.c_init = nn.Parameter(torch.randn(1, dim))
        self.lay = nn.LSTM(dim, dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(2*dim, dim, bias=False)
        self.return_seq = return_seq

    def forward(self, x, length):
        '''x: [batch, length, dim]'''
        batch = x.shape[0]
        length_max = x.shape[1]

        h_init = self.h_init.expand(batch, -1).unsqueeze(0).repeat(2, 1, 1)
        c_init = self.c_init.expand(batch, -1).unsqueeze(0).repeat(2, 1, 1)

        # Packing input sequence
        length = length.cpu()
        x_packed = nn.utils.rnn.pack_padded_sequence(x, length, batch_first=True, enforce_sorted=False)
        
        # Feedforward
        y_packed, (h_n, c_n) = self.lay(x_packed, (h_init, c_init))

        # Unpacking sequence
        y, _ = nn.utils.rnn.pad_packed_sequence(y_packed, batch_first=True, total_length=length_max)
        
        if self.return_seq:
            return self.fc(y)
        else:
            y_last = y[range(batch), length - 1, :]
            return self.fc(y_last)


class ShidoTreeLSTMLayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ShidoTreeLSTMLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.U_f = BiLSTM(dim_out, return_seq=True)
        self.U_i = BiLSTM(dim_out)
        self.U_u = BiLSTM(dim_out)
        self.U_o = BiLSTM(dim_out)
        self.W = nn.Linear(dim_in, dim_out * 4).to("cuda:0")
        self.h_init = nn.Parameter(torch.zeros(1, dim_out)).to('cuda:0')
        self.c_init = nn.Parameter(torch.zeros(1, dim_out)).to("cuda:0")

    def forward(self, tensor, indices):
        h_tensor = self.h_init
        c_tensor = self.c_init
        res_h, res_c = [], []
        for indice, x in zip(indices, tensor):
            h_tensor, c_tensor = self.apply(
                x, h_tensor, c_tensor, indice.to("cuda:0")
            )
            res_h.append(h_tensor[:, :])
            res_c.append(c_tensor[:, :])
            h_tensor = torch.cat([self.h_init, h_tensor], 0)
            c_tensor = torch.cat([self.c_init, c_tensor], 0)
        return res_h, res_c

    def apply(self, x, h_tensor, c_tensor, indice):
        mask_bool = indice != -1
        mask = mask_bool.float()  # [nodes, child]
        length = mask.sum(1).int()

        h = h_tensor.index_select(0, indice.where(mask_bool, torch.zeros_like(indice)).flatten()).view(*indice.shape, -1)
        c = c_tensor.index_select(0, indice.where(mask_bool, torch.zeros_like(indice)).flatten()).view(*indice.shape, -1)

        W_x = self.W(x.to("cuda:0"))  # [nodes, dim_out * 4]
        W_f_x = W_x[:, :self.dim_out * 1]  # [nodes, dim_out]
        W_i_x = W_x[:, self.dim_out * 1:self.dim_out * 2]
        W_u_x = W_x[:, self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[:, self.dim_out * 3:]

        branch_f_k = self.U_f(h, length)
        branch_f_k = torch.sigmoid(W_f_x.unsqueeze(1) + branch_f_k)
        branch_f = (branch_f_k * c * mask.unsqueeze(-1)).sum(1)  # [node, dim_out]

        branch_i = self.U_i(h, length)  # [nodes, dim_out]
        branch_i = torch.sigmoid(branch_i + W_i_x)   # [nodes, dim_out]
        branch_u = self.U_u(h, length)  # [nodes, dim_out]
        branch_u = torch.tanh(branch_u + W_u_x)
        branch_o = self.U_o(h, length)  # [nodes, dim_out]
        branch_o = torch.sigmoid(branch_o + W_o_x)

        new_c = branch_i * branch_u + branch_f  # [node, dim_out]
        new_h = branch_o * torch.tanh(new_c)  # [node, dim_out]

        return new_h, new_c

class ShidoTreeLSTMLayerTreeBase(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ShidoTreeLSTMLayerTreeBase, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.U_f = BiLSTM(dim_out, return_seq=True)
        self.U_i = BiLSTM(dim_out)
        self.U_u = BiLSTM(dim_out)
        self.U_o = BiLSTM(dim_out)
        self.W = nn.Linear(dim_in, dim_out * 4)
        self.h_init = nn.Parameter(torch.zeros(1, dim_out))
        self.c_init = nn.Parameter(torch.zeros(1, dim_out))

    @staticmethod
    def get_nums(roots):
        res = [[x.num for x in n.children] if n.children != [] else [0] for n in roots]
        max_len = max([len(x) for x in res])
        res = pad_sequence([torch.tensor(x, dtype=torch.int32) for x in res], batch_first=True, padding_value=-1)
        return res

    def forward(self, roots):
        depthes = [x[1] for x in sorted(depth_split_batch2(
            roots).items(), key=lambda x:-x[0])]  # list of list of Nodes
        indices = [self.get_nums(nodes) for nodes in depthes]

        h_tensor = self.h_init
        c_tensor = self.c_init
        for indice, nodes in zip(indices, depthes):
            x = torch.stack([node.h for node in nodes])  # [nodes, dim_in]
            h_tensor, c_tensor = self.apply(x, h_tensor, c_tensor, indice, nodes)
            h_tensor = torch.cat([self.h_init, h_tensor], 0)
            c_tensor = torch.cat([self.c_init, c_tensor], 0)
        return depthes[-1]

    def apply(self, x, h_tensor, c_tensor, indice, nodes):
        mask_bool = indice != -1
        mask = mask_bool.float()  # [nodes, child]
        length = mask.sum(1).int()

        h = h_tensor.index_select(0, indice.where(mask_bool, torch.zeros_like(indice)).flatten()).view(*indice.shape, -1)
        c = c_tensor.index_select(0, indice.where(mask_bool, torch.zeros_like(indice)).flatten()).view(*indice.shape, -1)

        W_x = self.W(x)  # [nodes, dim_out * 4]
        W_f_x = W_x[:, :self.dim_out * 1]  # [nodes, dim_out]
        W_i_x = W_x[:, self.dim_out * 1:self.dim_out * 2]
        W_u_x = W_x[:, self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[:, self.dim_out * 3:]

        branch_f_k = self.U_f(h, length)
        branch_f_k = torch.sigmoid(W_f_x.unsqueeze(1) + branch_f_k)
        branch_f = (branch_f_k * c * mask.unsqueeze(-1)).sum(1)  # [node, dim_out]

        branch_i = self.U_i(h, length)  # [nodes, dim_out]
        branch_i = torch.sigmoid(branch_i + W_i_x)   # [nodes, dim_out]
        branch_u = self.U_u(h, length)  # [nodes, dim_out]
        branch_u = torch.tanh(branch_u + W_u_x)
        branch_o = self.U_o(h, length)  # [nodes, dim_out]
        branch_o = torch.sigmoid(branch_o + W_o_x)

        new_c = branch_i * branch_u + branch_f  # [node, dim_out]
        new_h = branch_o * torch.tanh(new_c)  # [node, dim_out]

        for n, c, h in zip(nodes, new_c, new_h):
            n.c = c
            n.h = h

        return new_h, new_c

class ShidoTreeLSTMWithEmbedding(ShidoTreeLSTMLayer):
    def __init__(self, in_vocab, dim_in, dim_out):
        super(ShidoTreeLSTMWithEmbedding, self).__init__(dim_in, dim_out)
        self.E = nn.Parameter(torch.Tensor(in_vocab, dim_in))
        nn.init.uniform_(self.E, a=-0.05, b=0.05)  # Random uniform initialization
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.U_f = BiLSTM(dim_out, return_seq=True)
        self.U_i = BiLSTM(dim_out)
        self.U_u = BiLSTM(dim_out)
        self.U_o = BiLSTM(dim_out)
        self.W = nn.Linear(dim_in, dim_out * 4)
        self.h_init = nn.Parameter(torch.zeros(1, dim_out))
        self.c_init = nn.Parameter(torch.zeros(1, dim_out))

    def forward(self, roots):
        depthes = [x[1] for x in sorted(depth_split_batch2(
            roots).items(), key=lambda x:-x[0])]  # list of list of Nodes
        indices = [self.get_nums(nodes) for nodes in depthes]

        h_tensor = self.h_init
        c_tensor = self.c_init
        for indice, nodes in zip(indices, depthes):
            x = self.E[[node.label for node in nodes]]  # [nodes, dim_in]
            h_tensor, c_tensor = self.apply(x, h_tensor, c_tensor, indice, nodes)
            h_tensor = torch.cat([self.h_init, h_tensor], 0)
            c_tensor = torch.cat([self.c_init, c_tensor], 0)
        return depthes[-1]


class TreeDropout(nn.Module):
    def __init__(self, rate):
        super(TreeDropout, self).__init__()
        self.dropout_layer = nn.Dropout(rate)

    def forward(self, roots):
        nodes = [node for root in roots for node in traverse(root)]
        ys = [node.h for node in nodes]
        tensor = torch.stack(ys)
        dropped = self.dropout_layer(tensor)
        for e, v in enumerate(torch.split(dropped, len(ys))):
            nodes[e].h = v.squeeze()
        return roots


class SetEmbeddingLayer(nn.Module):
    def __init__(self, dim_E, in_vocab):
        super(SetEmbeddingLayer, self).__init__()
        self.E = nn.Embedding(in_vocab, dim_E)
        self.E.weight.data.uniform_(-1, 1)

    def forward(self, sets):
        length = [len(s) for s in sets]
        concatenated = torch.cat(sets, 0)
        embedded = self.E(concatenated)
        y = torch.split(embedded, length)
        return y


class LSTMEncoder(nn.Module):
    def __init__(self, dim, layer=1):
        super(LSTMEncoder, self).__init__()
        self.dim = dim
        self.lstm = nn.LSTM(input_size=dim, hidden_size=dim, num_layers=layer, batch_first=True)

    def forward(self, x, lengths):
        '''x: [batch, length, dim]'''
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, c_n) = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        return output, (h_n, c_n)


class SequenceEmbeddingLayer(nn.Module):
    def __init__(self, dim_E, in_vocab):
        super(SequenceEmbeddingLayer, self).__init__()
        self.E = nn.Embedding(in_vocab, dim_E)

    def forward(self, y):
        y = self.E(y)
        return y

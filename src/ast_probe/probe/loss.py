import torch
import torch.nn as nn


class LossFunc(nn.Module):
    """each loss function should be a subclass of this class

    Args:
        nn (_type_): _description_
    """

    pass


class ParserLoss(nn.Module):
    def __init__(self, max_c_len, pretrained=False):
        super(ParserLoss, self).__init__()
        self.ds = nn.L1Loss()
        self.cs = nn.L1Loss()
        self.us = nn.L1Loss()
        self.max_c_len = max_c_len
        self.pretrained = pretrained

    def forward(self, d_pred, c_pred, u_pred, d_real, c_real, u_real, length_batch):
        loss_d = self.ds(d_pred, d_real)

        if c_pred.shape[2] != c_real.shape[2]:
            num_elements = c_pred.shape[2] - c_real.shape[2]
            fill_tensor = torch.full((c_real.size(0), c_real.size(1), num_elements), -1)
            c_real = torch.cat((c_real, fill_tensor), dim=2)
        loss_c = self.cs(c_pred, c_real)

        loss_u = self.us(u_pred, u_real)

        return loss_d + loss_c + loss_u

    @staticmethod
    def calculate_hits(d_pred, c_pred, u_pred, d_real, c_real, u_real, length_batch):
        d_hits = torch.ceil(d_pred).eq(d_real).sum().item()
        u_hits = torch.ceil(u_pred).eq(u_real).sum().item()

        if c_pred.shape[2] != c_real.shape[2]:
            num_elements = c_pred.shape[2] - c_real.shape[2]
            fill_tensor = torch.full((c_real.size(0), c_real.size(1), num_elements), -1)
            c_real = torch.cat((c_real, fill_tensor), dim=2)
        c_hits = torch.ceil(c_pred).eq(c_real).sum().item()

        return (
            d_hits,
            c_hits,
            u_hits,
            (length_batch * d_pred.size(1)),
            (length_batch * c_pred.size(1) * c_pred.size(2)),
        )

import torch
import torch.nn as nn


class LossFunc(nn.Module):
    """each loss function should be a subclass of this class

    Args:
        nn (_type_): _description_
    """

    pass


class ParserLoss(nn.Module):
    """
    Loss function for AST-NN
    """

    def __init__(self, max_c_len):
        """
        we're using L1Loss for calcualting the distances for each tuple element in the predictions

        Args:
            max_c_len (_type_): maximum number of cs in the dataset
        """
        super(ParserLoss, self).__init__()
        self.ds = nn.L1Loss()
        self.cs = nn.L1Loss()
        self.us = nn.L1Loss()
        self.max_c_len = max_c_len

    def forward(self, d_pred, c_pred, u_pred, d_real, c_real, u_real, length_batch):
        """
        forward function for calculating the loss

        Args:
            d_pred (_type_): ds predicted by the model
            c_pred (_type_): cs predicted by the model
            u_pred (_type_): us predicted by the model
            d_real (_type_): the real d to be used as ground truth
            c_real (_type_): the real c to be used as ground truth
            u_real (_type_): the real u to be used as ground truth
            #!length_batch (_type_): i don't think this is used anymore. to be checked later

        Returns:
            the accumulated loss of d,c,u
        """
        loss_d = self.ds(
            d_pred, d_real
        )  # for ASTNN, there is no need to match d =_pred and d_real lengths.

        # as c is 3 dimensional, there is a need to fill c_real with -1s to match the dimensions of c_pred
        if c_pred.shape[2] != c_real.shape[2]:
            num_elements = c_pred.shape[2] - c_real.shape[2]
            fill_tensor = torch.full((c_real.size(0), c_real.size(1), num_elements), -1)
            c_real = torch.cat((c_real, fill_tensor), dim=2)
        loss_c = self.cs(c_pred, c_real)

        loss_u = self.us(
            u_pred, u_real
        )  # same as loss_d, there is no need to match u_pred and u_real lengths

        return loss_d + loss_c + loss_u

    @staticmethod
    def calculate_hits(d_pred, c_pred, u_pred, d_real, c_real, u_real, length_batch):
        """
        calculates the number of correct predictions of the probe by comparing the predictions with the real values.
        to calculate the number of hits I'm using round function. if the predicted value is 1.1 and the real value is 1, torch.round(1.1) will return 1.
        Basically, torch.round the closest integer greater than or equal to each element.

        Args:
            d_pred (_type_): ds predicted by the model
            c_pred (_type_): cs predicted by the model
            u_pred (_type_): us predicted by the model
            d_real (_type_): the real d to be used as ground truth
            c_real (_type_): the real c to be used as ground truth
            u_real (_type_): the real u to be used as ground truth
            length_batch (_type_): _description_

        Returns:
            _type_: the number of correct predictions for d,c,u
        """

        d_hits = torch.round(d_pred).eq(d_real).sum().item()
        u_hits = torch.round(u_pred).eq(u_real).sum().item()

        if c_pred.shape[2] != c_real.shape[2]:
            num_elements = c_pred.shape[2] - c_real.shape[2]
            fill_tensor = torch.full((c_real.size(0), c_real.size(1), num_elements), -1)
            c_real = torch.cat((c_real, fill_tensor), dim=2)
        c_hits = torch.round(c_pred).eq(c_real).sum().item()

        return (
            d_hits,
            c_hits,
            u_hits,
            (length_batch * d_pred.size(1)),
            (length_batch * c_pred.size(1) * c_pred.size(2)),
        )


class ParserLossFuncGNN(nn.Module):
    """
    Loss function for FuncGNN
    """

    def __init__(self, max_c_len):
        """
        we're using L1Loss for calcualting the distances for each tuple element in the predictions

        Args:
            max_c_len (_type_): maximum number of cs in the dataset

        """
        super(ParserLossFuncGNN, self).__init__()
        self.ds = nn.L1Loss()
        self.cs = nn.L1Loss()
        self.us = nn.L1Loss()
        self.max_c_len = max_c_len

    def forward(self, d_pred, c_pred, u_pred, d_real, c_real, u_real, length_batch):
        """
        forward function for calculating the loss

        Args:
            d_pred (_type_): ds predicted by the model
            c_pred (_type_): cs predicted by the model
            u_pred (_type_): us predicted by the model
            d_real (_type_): the real d to be used as ground truth
            c_real (_type_): the real c to be used as ground truth
            u_real (_type_): the real u to be used as ground truth
            #!length_batch (_type_): i don't think this is used anymore. to be checked later

        Returns:
            the accumulated loss of d,c,u
        """
        # unlike ASTNN, there is a need to match the lengths of d_pred and d_real
        if d_real.shape[1] != d_pred.shape[1]:
            padding_size = d_pred.shape[1] - d_real.shape[1]
            d_real = torch.nn.functional.pad(d_real, (0, padding_size), value=-1)
        # same goes for cs
        if c_pred.shape[1] != c_real.shape[1]:
            num_elements = c_pred.shape[1] - c_real.shape[1]
            fill_tensor = torch.full((c_real.size(0), num_elements, 2), -1)
            c_real = torch.cat((c_real, fill_tensor), dim=1)
        # same goes for us
        if u_real.shape[1] != u_pred.shape[1]:
            padding_size = u_pred.shape[1] - u_real.shape[1]
            u_real = torch.nn.functional.pad(u_real, (0, padding_size), value=-1)

        loss_d = self.ds(d_pred, d_real)
        loss_c = self.cs(c_pred, c_real)
        loss_u = self.us(u_pred, u_real)

        return loss_d + loss_c + loss_u

    @staticmethod
    def calculate_hits(d_pred, c_pred, u_pred, d_real, c_real, u_real, length_batch):
        """
        calculates the number of correct predictions of the probe by comparing the predictions with the real values.
        to calculate the number of hits I'm using round function. if the predicted value is 1.1 and the real value is 1, torch.round(1.1) will return 1.
        Basically, torch.round the closest integer greater than or equal to each element.

        Args:
            d_pred (_type_): ds predicted by the model
            c_pred (_type_): cs predicted by the model
            u_pred (_type_): us predicted by the model
            d_real (_type_): the real d to be used as ground truth
            c_real (_type_): the real c to be used as ground truth
            u_real (_type_): the real u to be used as ground truth
            length_batch (_type_): _description_

        Returns:
            _type_: the number of correct predictions for d,c,u
        """
        if d_real.shape[1] != d_pred.shape[1]:
            padding_size = d_pred.shape[1] - d_real.shape[1]
            d_real = torch.nn.functional.pad(d_real, (0, padding_size), value=-1)

        if c_pred.shape[1] != c_real.shape[1]:
            num_elements = c_pred.shape[1] - c_real.shape[1]
            fill_tensor = torch.full((c_real.size(0), num_elements, 2), -1)
            c_real = torch.cat((c_real, fill_tensor), dim=1)

        if u_real.shape[1] != u_pred.shape[1]:
            padding_size = u_pred.shape[1] - u_real.shape[1]
            u_real = torch.nn.functional.pad(u_real, (0, padding_size), value=-1)

        d_hits = torch.round(d_pred).eq(d_real).sum().item()
        u_hits = torch.round(u_pred).eq(u_real).sum().item()

        if c_pred.shape[2] != c_real.shape[2]:
            num_elements = c_pred.shape[2] - c_real.shape[2]
            fill_tensor = torch.full((c_real.size(0), c_real.size(1), num_elements), -1)
            c_real = torch.cat((c_real, fill_tensor), dim=2)
        c_hits = torch.round(c_pred).eq(c_real).sum().item()

        return (
            d_hits,
            c_hits,
            u_hits,
            (length_batch * d_pred.size(1)),
            (length_batch * c_pred.size(1) * c_pred.size(2)),
        )


class ParserLossSumTF(nn.Module):
    """
    Loss function for SumTF
    """

    def __init__(self, max_c_len):
        """
        we're using L1Loss for calcualting the distances for each tuple element in the predictions

        Args:
            max_c_len (_type_): maximum number of cs in the dataset

        """
        super(ParserLossSumTF, self).__init__()
        self.ds = nn.L1Loss()
        self.cs = nn.L1Loss()
        self.us = nn.L1Loss()
        self.max_c_len = max_c_len

    def forward(self, d_pred, c_pred, u_pred, d_real, c_real, u_real, length_batch):
        """
        forward function for calculating the loss

        Args:
            d_pred (_type_): ds predicted by the model
            c_pred (_type_): cs predicted by the model
            u_pred (_type_): us predicted by the model
            d_real (_type_): the real d to be used as ground truth
            c_real (_type_): the real c to be used as ground truth
            u_real (_type_): the real u to be used as ground truth
            #!length_batch (_type_): i don't think this is used anymore. to be checked later

        Returns:
            the accumulated loss of d,c,u
        """
        # unlike ASTNN, there is a need to match the lengths of d_pred and d_real
        if d_real.shape[1] != d_pred.shape[1]:
            num_elements = d_pred.shape[1] - d_real.shape[1]
            fill_tensor = torch.full((d_real.size(0), num_elements, 2), -1)
            d_real = torch.cat((d_real, fill_tensor), dim=1)
        # same goes for cs
        if c_pred.shape[1] != c_real.shape[1]:
            padding_size = c_pred.shape[1] - c_real.shape[1]
            c_real = torch.nn.functional.pad(c_real, (0, padding_size), value=-1)
        # same goes for us
        if u_real.shape[1] != u_pred.shape[1]:
            padding_size = u_pred.shape[1] - u_real.shape[1]
            u_real = torch.nn.functional.pad(u_real, (0, padding_size), value=-1)

        loss_d = self.ds(d_pred, d_real)
        loss_c = self.cs(c_pred, c_real)
        loss_u = self.us(u_pred, u_real)

        return loss_d + loss_c + loss_u

    @staticmethod
    def calculate_hits(d_pred, c_pred, u_pred, d_real, c_real, u_real, length_batch):
        """
        calculates the number of correct predictions of the probe by comparing the predictions with the real values.
        to calculate the number of hits I'm using round function. if the predicted value is 1.1 and the real value is 1, torch.round(1.1) will return 1.
        Basically, torch.round the closest integer greater than or equal to each element.

        Args:
            d_pred (_type_): ds predicted by the model
            c_pred (_type_): cs predicted by the model
            u_pred (_type_): us predicted by the model
            d_real (_type_): the real d to be used as ground truth
            c_real (_type_): the real c to be used as ground truth
            u_real (_type_): the real u to be used as ground truth
            length_batch (_type_): _description_

        Returns:
            _type_: the number of correct predictions for d,c,u
        """
        if d_real.shape[1] != d_pred.shape[1]:
            padding_size = d_pred.shape[1] - d_real.shape[1]
            d_real = torch.nn.functional.pad(d_real, (0, padding_size), value=-1)

        if c_pred.shape[1] != c_real.shape[1]:
            num_elements = c_pred.shape[1] - c_real.shape[1]
            fill_tensor = torch.full((c_real.size(0), num_elements, 2), -1)
            c_real = torch.cat((c_real, fill_tensor), dim=1)

        if u_real.shape[1] != u_pred.shape[1]:
            padding_size = u_pred.shape[1] - u_real.shape[1]
            u_real = torch.nn.functional.pad(u_real, (0, padding_size), value=-1)

        d_hits = torch.round(d_pred).eq(d_real).sum().item()
        u_hits = torch.round(u_pred).eq(u_real).sum().item()

        if c_pred.shape[2] != c_real.shape[2]:
            num_elements = c_pred.shape[2] - c_real.shape[2]
            fill_tensor = torch.full((c_real.size(0), c_real.size(1), num_elements), -1)
            c_real = torch.cat((c_real, fill_tensor), dim=2)
        c_hits = torch.round(c_pred).eq(c_real).sum().item()

        return (
            d_hits,
            c_hits,
            u_hits,
            (length_batch * d_pred.size(1)),
            (length_batch * c_pred.size(1) * c_pred.size(2)),
        )

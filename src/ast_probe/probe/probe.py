import torch.nn as nn
import torch


class Probe(nn.Module):
    """
    each probe must have a bunch of projections

    Args:
        nn (_type_): _description_
    """

    pass


class ParserProbe(Probe):
    def __init__(
        self, probe_rank, hidden_dim, number_labels_d, number_labels_c, number_labels_u
    ):
        print("Constructing ParserProbe")
        super(ParserProbe, self).__init__()
        self.probe_rank = probe_rank
        self.hidden_dim = hidden_dim
        self.number_vectors_d = number_labels_d
        self.number_vectors_c = number_labels_c
        self.number_vectors_u = number_labels_u
        self.proj = nn.Parameter(data=torch.zeros(self.hidden_dim, self.probe_rank))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        #! add self.vectors_d for distance here later
        self.vectors_d = nn.Parameter(data=torch.zeros(self.probe_rank))
        nn.init.uniform_(self.vectors_d, -0.05, 0.05)
        self.vectors_c = nn.Parameter(
            data=torch.zeros(self.probe_rank, self.number_vectors_c)
        )
        nn.init.uniform_(self.vectors_c, -0.05, 0.05)
        self.vectors_u = nn.Parameter(data=torch.zeros(self.probe_rank))
        nn.init.uniform_(self.vectors_u, -0.05, 0.05)

    def forward(self, batch):
        """
        Args:
            batch: a batch of word representations of the shape
                    (batch_size, max_seq_len, representation_dim)

        Returns:
            d_pred: (batch_size, max_seq_len - 1)
            scores_c: (batch_size, max_seq_len - 1, number classes_c)
            scores_u: (batch_size, max_seq_len, number classes_u)
        """
        transformed = torch.matmul(batch, self.proj)

        ds_pred = torch.matmul(transformed, self.vectors_d)
        cs_pred = torch.matmul(transformed, self.vectors_c)
        us_pred = torch.matmul(transformed, self.vectors_u)

        return (ds_pred, cs_pred, us_pred)


class FuncGNNParserProbe(Probe):
    def __init__(
        self, probe_rank, hidden_dim, number_labels_d, number_labels_c, number_labels_u
    ):
        print("Constructing ParserProbe")
        super(FuncGNNParserProbe, self).__init__()
        self.probe_rank = probe_rank
        self.hidden_dim = hidden_dim
        self.number_vectors_d = number_labels_d
        self.number_vectors_c = number_labels_c
        self.number_vectors_u = number_labels_u
        self.proj = nn.Parameter(data=torch.zeros(self.hidden_dim, self.probe_rank))
        nn.init.uniform_(self.proj, -0.05, 0.05)
        #! add self.vectors_d for distance here later
        self.vectors_d = nn.Parameter(data=torch.zeros(self.probe_rank))
        nn.init.uniform_(self.vectors_d, -0.05, 0.05)
        self.vectors_c = nn.Parameter(data=torch.zeros(self.probe_rank, 1))
        nn.init.uniform_(self.vectors_c, -0.05, 0.05)
        self.vectors_c_edges = nn.Parameter(
            data=torch.zeros(self.number_vectors_d, self.number_vectors_c * 2)
        )
        nn.init.uniform_(self.vectors_c_edges, -0.05, 0.05)
        self.vectors_u = nn.Parameter(data=torch.zeros(self.probe_rank))
        nn.init.uniform_(self.vectors_u, -0.05, 0.05)

    def forward(self, batch):
        """
        Args:
            batch: a batch of word representations of the shape
                    (batch_size, max_seq_len, representation_dim)

        Returns:
            d_pred: (batch_size, max_seq_len - 1)
            scores_c: (batch_size, max_seq_len - 1, number classes_c)
            scores_u: (batch_size, max_seq_len, number classes_u)
        """
        transformed = torch.matmul(batch, self.proj)
        # j = transformed.unsqueeze(2)
        ds_pred = torch.matmul(transformed, self.vectors_d)
        cs_pred_proj = torch.matmul(transformed, self.vectors_c)
        # j = torch.squeeze(cs_pred_proj)
        cs_pred = torch.matmul(torch.squeeze(cs_pred_proj), self.vectors_c_edges).view(
            cs_pred_proj.shape[0], 64, 2
        )
        us_pred = torch.matmul(transformed, self.vectors_u)

        return (ds_pred, cs_pred, us_pred)

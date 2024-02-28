import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from layers import *


class AttentionDecoder(nn.Module):
    def __init__(self, dim_F, dim_rep, vocab_size, layer=1):
        super(AttentionDecoder, self).__init__()
        self.layer = layer
        self.dim_rep = dim_rep
        self.F = nn.Embedding(vocab_size, dim_F)
        self.layers = nn.ModuleList(
            [nn.LSTM(dim_rep, dim_rep, batch_first=True) for _ in range(layer)]
        )
        self.fc = nn.Linear(dim_rep, vocab_size)

        # used for attention
        self.W1 = nn.Linear(dim_rep, dim_rep)
        self.W2 = nn.Linear(dim_rep, dim_rep)
        self.V = nn.Linear(dim_rep, 1)

    @staticmethod
    def loss_function(real, pred):
        loss_ = F.cross_entropy(pred, real.long(), reduction="sum")
        return loss_

    def get_loss(self, enc_y, states, target, dropout=0.0):
        """
        enc_y: batch_size([seq_len, dim])
        states: ([batch, dim], [batch, dim])
        target: [batch, max_len] (padded with -1.)
        """
        mask = target != -1.0
        h, c = states
        enc_y, _ = pad_tensor(
            enc_y
        )  # Assuming you have a corresponding pad_tensor function
        enc_y = F.dropout(enc_y, 1.0 - dropout)
        dec_hidden = F.dropout(h, 1.0 - dropout)
        dec_cell = F.dropout(c, 1.0 - dropout)

        l_states = [(dec_hidden, dec_cell) for _ in range(self.layer)]
        target = F.relu(target)
        dec_input = target[:, 0]
        loss = 0
        for t in range(1, target.shape[1]):
            # passing enc_output to the decoder
            predictions, l_states, att = self(dec_input, l_states, enc_y)
            real = target[:, t].masked_select(mask[:, t])
            pred = predictions[mask[:, t]]
            loss += self.loss_function(real, pred)
            # using teacher forcing
            dec_input = target[:, t]

        return loss / torch.sum(mask.float())

    def translate(self, y_enc, states, max_length, start_token, end_token):
        """
        enc_y: [seq_len, dim]
        states: ([dim,], [dim,])
        """
        attention_plot = np.zeros((max_length, y_enc.shape[0]))

        l_states = [(state[0].unsqueeze(0), state[1].unsqueeze(0)) for state in states]
        dec_input = torch.full((1,), start_token, dtype=torch.int64)
        result = []

        for t in range(max_length):
            predictions, l_states, attention_weights = self(
                dec_input, l_states, y_enc.unsqueeze(0)
            )

            attention_weights = attention_weights.view(
                -1,
            )
            attention_plot[t] = attention_weights.detach().numpy()

            predicted_id = torch.argmax(predictions[0]).item()
            result.append(predicted_id)

            if predicted_id == end_token:
                return result[:-1], attention_plot[:t]

            # the predicted ID is fed back into the model
            dec_input = torch.tensor([predicted_id])

        return result, attention_plot

    def forward(self, x, l_states, enc_y):
        # hidden shape == (batch_size, hidden size)
        if l_states[0][0].shape[0] == 1:
            hidden_with_time_axis = l_states[-1][0].squeeze(0).unsqueeze(1)
        else:
            hidden_with_time_axis = l_states[-1][0].unsqueeze(1)

        # score shape == (batch_size, max_length, hidden_size)
        i_s = self.W1(enc_y)
        j_s = self.W2(hidden_with_time_axis)
        f = i_s + j_s
        score = torch.tanh(f)

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = F.softmax(self.V(score), dim=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_y
        context_vector = context_vector.sum(dim=1)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = x.long()
        x = x.unsqueeze(1)
        x = self.F(x)

        # passing the concatenated vector to the LSTM layers
        new_l_states = []
        for i, states in zip(range(self.layer), l_states):
            if i <= self.layer - 1:
                skip = x
                if len(states[0].shape) == 2:
                    hx = states[0].unsqueeze(0)
                    cx = states[1].unsqueeze(0)
                elif len(states[0].shape) == 3:
                    hx = states[0]
                    cx = states[1]
                x, (h, c) = self.layers[i](x, (hx, cx))
                x = x + skip
            else:
                x = torch.cat([context_vector.unsqueeze(1), x], dim=-1)
                hx = states[0].unsqueeze(1)
                cx = states[1].unsqueeze(1)
                x, (h, c) = self.layers[i](x, (hx, cx))
            n_states = (h, c)
            new_l_states.append(n_states)

        # output shape == (batch_size * 1, hidden_size)
        x = x.view(-1, x.shape[2])

        # output shape == (batch_size * 1, vocab)
        x = self.fc(x)

        return x, new_l_states, attention_weights


class BaseModel(nn.Module):
    def __init__(
        self, dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer=1, dropout=0.0, lr=1e-3
    ):
        super(BaseModel, self).__init__()
        self.dim_E = dim_E
        self.dim_F = dim_F
        self.dim_rep = dim_rep
        self.in_vocab = in_vocab
        self.out_vocab = out_vocab
        self.dropout = dropout
        self.decoder = AttentionDecoder(dim_F, dim_rep, out_vocab, layer)
        self.optimizer = Adam(self.parameters(), lr=lr)

    def encode(self, trees):
        """
        ys: list of [seq_len, dim]
        hx, cx: [batch, dim]
        return: ys, [hx, cx]
        """

    def train_on_batch(self, x, y):
        self.optimizer.zero_grad()

        y_enc, (c, h) = self.encode(x)
        loss = self.decoder.get_loss(y_enc, (c, h), y, dropout=self.dropout)
        loss.backward()
        self.optimizer.step()
        return loss

    def translate(self, x, nl_i2w, nl_w2i, max_length=100):
        res = []
        y_enc, (c, h) = self.encode(x)
        batch_size = len(y_enc)
        for i in range(batch_size):
            nl, _ = self.decoder.translate(
                y_enc[i], (c[i], h[i]), max_length, nl_w2i["<s>"], nl_w2i["</s>"]
            )
            res.append([nl_i2w[n] for n in nl])
        return res

    def evaluate_on_batch(self, x, y):
        y_enc, (c, h) = self.encode(x)
        loss = self.decoder.get_loss(y_enc, (c, h), y)
        return loss.item()


class CodennModel(BaseModel):
    def __init__(
        self, dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer=1, dropout=0.5, lr=1e-3
    ):
        super(CodennModel, self).__init__(
            dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer, dropout, lr
        )
        self.dropout = dropout
        self.E = SetEmbeddingLayer(dim_E, in_vocab)
        print(
            "I am CodeNNModel, dim is {} and {} layered".format(str(self.dim_rep), "0")
        )

    def encode(self, sets):
        sets = self.E(sets)
        # sets = [F.dropout(t, self.dropout) for t in sets]

        hx = torch.zeros([len(sets), self.dim_rep])
        cx = torch.zeros([len(sets), self.dim_rep])
        ys = sets

        return ys, [hx, cx]


class Seq2seqModel(BaseModel):
    def __init__(
        self, dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer=1, dropout=0.5, lr=1e-3
    ):
        super(Seq2seqModel, self).__init__(
            dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer, dropout, lr
        )
        self.layer = layer
        self.dropout = dropout
        self.E = nn.Embedding(in_vocab + 1, dim_E, padding_idx=0)
        self.lstm_layers = nn.ModuleList(
            [
                nn.LSTM(dim_rep if i == 0 else dim_E, dim_rep, batch_first=True)
                for i in range(layer)
            ]
        )
        print(
            "I am seq2seq model, dim is {} and {} layered".format(
                str(self.dim_rep), str(self.layer)
            )
        )

    def get_length(self, seq):
        return (seq != 0).sum(dim=1)

    def encode(self, seq):
        length = self.get_length(seq)
        tensor = self.E(seq + 1)
        # tensor = F.dropout(tensor, self.dropout)
        for i, lstm in enumerate(self.lstm_layers):
            tensor, (h, c) = lstm(tensor)
            if (
                i != len(self.lstm_layers) - 1
            ):  # No residual connection for the last layer
                tensor += skip

        hx = h[-1]  # Taking the last layer's hidden state
        cx = c[-1]  # Taking the last layer's cell state
        ys = [y[: i.item()] for y, i in zip(tensor, length)]

        return ys, [hx, cx]


class ChildsumModel(BaseModel):
    def __init__(
        self, dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer=1, dropout=0.5, lr=1e-4
    ):
        super(ChildsumModel, self).__init__(
            dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer, dropout, lr
        )
        self.layer = layer
        self.dropout = dropout
        self.E = TreeEmbeddingLayer(dim_E, in_vocab)
        self.child_sum_layers = nn.ModuleList(
            [ChildSumLSTMLayer(dim_E, dim_rep) for i in range(layer)]
        )
        print(
            "I am Child-sum model, dim is {} and {} layered".format(
                str(self.dim_rep), str(self.layer)
            )
        )

    def encode(self, x):
        tensor, indice, tree_num = x
        tensor = self.E(tensor)
        # tensor = [F.dropout(t, self.dropout) for t in tensor]
        for i, child_sum_layer in enumerate(self.child_sum_layers):
            skip = tensor
            tensor, c = child_sum_layer(tensor, indice)
            tensor = [t + s for t, s in zip(tensor, skip)]

        hx = tensor[-1]
        cx = c[-1]
        ys = []
        batch_size = tensor[-1].shape[0]
        tensor = torch.cat(tensor, 0)
        tree_num = torch.cat(tree_num, 0)
        for batch in range(batch_size):
            ys.append(tensor[tree_num == batch])
        return ys, [hx, cx]


class NaryModel(BaseModel):
    def __init__(
        self, dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer=1, dropout=0.5, lr=1e-4
    ):
        super(NaryModel, self).__init__(
            dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer, dropout, lr
        )
        self.layer = layer
        self.dropout = dropout
        self.E = TreeEmbeddingLayer(dim_E, in_vocab)
        self.nary_layers = nn.ModuleList(
            [NaryLSTMLayer(dim_E, dim_rep) for i in range(layer)]
        )
        print(
            "I am N-ary model, dim is {} and {} layered".format(
                str(self.dim_rep), str(self.layer)
            )
        )

    def encode(self, x):
        tensor, indice, tree_num = x
        tensor = self.E(tensor)
        # tensor = [F.dropout(t, self.dropout) for t in tensor]
        for i, nary_layer in enumerate(self.nary_layers):
            skip = tensor
            tensor, c = nary_layer(tensor, indice)
            tensor = [t + s for t, s in zip(tensor, skip)]

        hx = tensor[-1]
        cx = c[-1]
        ys = []
        batch_size = tensor[-1].shape[0]
        tensor = torch.cat(tensor, 0)
        tree_num = torch.cat(tree_num, 0)
        for batch in range(batch_size):
            ys.append(tensor[tree_num == batch])
        return ys, [hx, cx]


class MultiwayModel(BaseModel):
    def __init__(
        self, dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer=1, dropout=0.0, lr=1e-4
    ):
        super(MultiwayModel, self).__init__(
            dim_E, dim_F, dim_rep, in_vocab, out_vocab, layer, dropout, lr
        )
        self.layer = layer
        self.dropout = dropout
        self.E = TreeEmbeddingLayer(dim_E, in_vocab)
        self.multiway_layers = nn.ModuleList(
            [ShidoTreeLSTMLayer(dim_E, dim_rep) for i in range(layer)]
        )
        print(
            "I am Multi-way model, dim is {} and {} layered".format(
                str(self.dim_rep), str(self.layer)
            )
        )

    def encode(self, x):
        tensor, indice, tree_num = x
        tensor = self.E(tensor)
        # tensor = [F.dropout(t, self.dropout) for t in tensor]
        for i, multiway_layer in enumerate(self.multiway_layers):
            skip = tensor
            tensor, c = multiway_layer(tensor, indice)
            tensor = [t + s for t, s in zip(tensor, skip)]

        hx = tensor[-1]
        cx = c[-1]
        ys = []
        batch_size = tensor[-1].shape[0]
        tensor = torch.cat(tensor, 0)
        # tree_num = torch.cat(tree_num, 0)
        tree_num_tensors = [torch.from_numpy(arr) for arr in tree_num]
        tree_num = torch.cat(tree_num_tensors, dim=0)
        # tensors=[i.detach().numpy() for i in tensor]
        for batch in range(batch_size):
            # mask = (tensors == tree_num[batch])
            ys.append(tensor[tree_num == batch])
        return ys, [hx, cx]

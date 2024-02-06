import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
import random
from gensim.models.word2vec import Word2Vec
import os


class BatchTreeEncoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        encode_dim,
        batch_size,
        use_gpu,
        pretrained_weight=None,
    ):
        super(BatchTreeEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.W_c = nn.Linear(embedding_dim, encode_dim)
        self.activation = F.relu
        self.stop = -1
        self.batch_size = batch_size
        self.use_gpu = False
        self.node_list = []
        self.th = torch.cuda if use_gpu else torch
        self.batch_node = None
        self.max_index = vocab_size
        # pretrained  embedding
        if pretrained_weight is not None:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            # self.embedding.weight.requires_grad = False

    def create_tensor(self, tensor):
        if self.use_gpu:
            return tensor.cuda()
        return tensor

    def traverse_mul(self, node, batch_index):
        size = len(node)
        if not size:
            return None
        batch_current = self.create_tensor(
            Variable(torch.zeros(size, self.embedding_dim))
        )

        index, children_index = [], []
        current_node, children = [], []
        for i in range(size):
            # if node[i][0] is not -1:
            index.append(i)
            current_node.append(node[i][0])
            temp = node[i][1:]
            c_num = len(temp)
            for j in range(c_num):
                if temp[j][0] != -1:
                    if len(children_index) <= j:
                        children_index.append([i])
                        children.append([temp[j]])
                    else:
                        children_index[j].append(i)
                        children[j].append(temp[j])
        # else:
        #     batch_index[i] = -1

        batch_current = self.W_c(
            batch_current.index_copy(
                0,
                Variable(self.th.LongTensor(index)),
                self.embedding(Variable(self.th.LongTensor(current_node))),
            )
        )

        for c in range(len(children)):
            zeros = self.create_tensor(Variable(torch.zeros(size, self.encode_dim)))
            batch_children_index = [batch_index[i] for i in children_index[c]]
            tree = self.traverse_mul(children[c], batch_children_index)
            if tree is not None:
                batch_current += zeros.index_copy(
                    0, Variable(self.th.LongTensor(children_index[c])), tree
                )
        # batch_index = [i for i in batch_index if i is not -1]
        b_in = Variable(self.th.LongTensor(batch_index))
        self.node_list.append(self.batch_node.index_copy(0, b_in, batch_current))
        return batch_current

    def forward(self, x, bs):
        self.batch_size = bs
        self.batch_node = self.create_tensor(
            Variable(torch.zeros(self.batch_size, self.encode_dim))
        )
        self.node_list = []
        self.traverse_mul(x, list(range(self.batch_size)))
        self.node_list = torch.stack(self.node_list)
        return torch.max(self.node_list, 0)[0]


class BatchProgramCC(nn.Module):
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        vocab_size,
        encode_dim,
        label_size,
        batch_size,
        use_gpu=False,
        pretrained_weight=None,
        word2vec_path=None,
        language=None,
    ):
        super(BatchProgramCC, self).__init__()
        self.stop = [vocab_size - 1]
        self.hidden_dim = hidden_dim
        self.num_layers = 1
        self.gpu = use_gpu
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.encode_dim = encode_dim
        self.label_size = label_size
        self.encoder = BatchTreeEncoder(
            self.vocab_size,
            self.embedding_dim,
            self.encode_dim,
            self.batch_size,
            self.gpu,
            pretrained_weight,
        )
        self.root2label = nn.Linear(self.encode_dim, self.label_size)
        # gru
        self.bigru = nn.GRU(
            self.encode_dim,
            self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
        )
        # linear
        self.hidden2label = nn.Linear(self.hidden_dim * 2, self.label_size)
        # hidden
        self.hidden = self.init_hidden()
        self.dropout = nn.Dropout(0.2)

        # This is added by me to have the model remeber its embeddings when it used for probing
        word2vec = Word2Vec.load(word2vec_path).wv
        self.vocab = word2vec
        self.max_token = word2vec.vectors.shape[0]
        self.lang = language

    def process_input(self, input_batch):
        try:
            # get AST of input
            if self.lang == "c":
                from pycparser import c_parser
                from ast_nn.src.prepare_data import get_blocks as func

                code_asts = [c_parser.CParser().parse(i) for i in input_batch]
            elif self.lang == "java":
                import javalang
                from ast_nn.src.utils import get_blocks_v1 as func

                tokens = [javalang.tokenizer.tokenize(i) for i in input_batch]
                parsers = [javalang.parser.Parser(t) for t in tokens]
                code_asts = [p.parse_member_declaration() for p in parsers]

            # logger.info("Finished parsing single input")
        except Exception as e:
            # logger.exception("There was a problem in parsing the input: %s", e)
            raise e

        # convert AST to index representation
        def tree_to_index(node):
            token = node.token
            result = [
                (
                    self.vocab.key_to_index[token]
                    if token in self.vocab
                    else self.max_token
                )
            ]
            children = node.children

            for child in children:
                result.append(tree_to_index(child))

            return result

        def trans2seq(r):
            blocks = []
            func(r, blocks)
            tree = []
            for b in blocks:
                btree = tree_to_index(b)
                tree.append(btree)
            return tree

        try:
            code_trees = [trans2seq(code_ast) for code_ast in code_asts]
            x = [
                (
                    self.vocab.key_to_index[token]
                    if token in self.vocab
                    else self.max_token
                )
                for token in input_batch[0].split()
            ]
            # logger.info("Finished converting AST to index representation")
        except Exception as e:
            # logger.exception(
            #     "There was a problem in converting the AST to index representation: %s", e
            # )
            raise e

        return code_trees

    def init_hidden(self):
        if self.gpu is True:
            if isinstance(self.bigru, nn.LSTM):
                h0 = Variable(
                    torch.zeros(
                        self.num_layers * 2, self.batch_size, self.hidden_dim
                    ).cuda()
                )
                c0 = Variable(
                    torch.zeros(
                        self.num_layers * 2, self.batch_size, self.hidden_dim
                    ).cuda()
                )
                return h0, c0
            return Variable(
                torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)
            ).cuda()
        else:
            return Variable(
                torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)
            )

    def get_zeros(self, num):
        zeros = Variable(torch.zeros(num, self.encode_dim))
        if self.gpu:
            return zeros.cuda()
        return zeros

    def encode(self, x):
        #! This line should be turned off when training the original AST-NN model
        x = self.process_input(x)

        lens = [len(item) for item in x]
        max_len = max(lens)

        encodes = []
        for i in range(self.batch_size):
            for j in range(lens[i]):
                encodes.append(x[i][j])

        encodes = self.encoder(encodes, sum(lens))
        seq, start, end = [], 0, 0
        for i in range(self.batch_size):
            end += lens[i]
            seq.append(encodes[start:end])
            if max_len - lens[i]:
                seq.append(self.get_zeros(max_len - lens[i]))
            start = end

        encodes = torch.cat(seq)
        encodes = encodes.view(self.batch_size, max_len, -1)
        # since enforce_sorted is not supported in this pytorch version, we need to do it manually
        lengths, perm_idx = torch.LongTensor(lens).sort(0, descending=True)
        encodes = encodes[perm_idx]
        _, unperm_idx = perm_idx.sort(0, descending=False)

        encodes = nn.utils.rnn.pack_padded_sequence(encodes, lengths, True)

        # return encodes

        gru_out, _ = self.bigru(encodes, self.hidden)
        gru_out_hidden, _ = nn.utils.rnn.pad_packed_sequence(
            gru_out, batch_first=True, padding_value=-1e9
        )
        gru_out_hidden = gru_out_hidden[unperm_idx]
        gru_out_hidden = torch.transpose(gru_out_hidden, 1, 2)
        # pooling
        gru_out = F.max_pool1d(gru_out_hidden, gru_out_hidden.size(2)).squeeze(2)
        # gru_out = gru_out[:,-1]

        return gru_out, gru_out_hidden

    def forward(self, x1, x2):
        lvec, rvec = self.encode(x1)[0], self.encode(x2)[0]

        abs_dist = torch.abs(torch.add(lvec, -rvec))

        y = torch.sigmoid(self.hidden2label(abs_dist))
        return y

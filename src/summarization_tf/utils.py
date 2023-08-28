"""Utilities"""
import torch
import torch.nn.init as init
import numpy as np
import math
from collections import defaultdict
import pickle
from prefetch_generator import BackgroundGenerator


def get_nums(roots):
    """convert roots to indices"""
    res = [[x.num for x in n.children] if n.children != [] else [0] for n in roots]
    max_len = max([len(x) for x in res])

    # Pad sequences with a padding value of -1
    padded_res = [torch.tensor(seq + [-1] * (max_len - len(seq))) for seq in res]

    # Stack the sequences along the first dimension to form a tensor
    tensor_res = torch.stack(padded_res)

    return tensor_res


def tree2binary(trees):
    def helper(root):
        if len(root.children) > 2:
            tmp = root.children[0]
            for child in root.children[1:]:
                tmp.children += [child]
                tmp = child
            root.children = root.children[0:1]
        for child in root.children:
            helper(child)
        return root

    return [helper(x) for x in trees]


def tree2tensor(trees):
    """
    indice:
        this has structure data.
        0 represent init state,
        1<n represent children's number (1-indexed)
    depthes:
        these are labels of nodes at each depth.
    tree_num:
        explain number of tree that each node was conteined.
    """
    res = defaultdict(list)
    tree_num = defaultdict(list)
    for e, root in enumerate(trees):
        for k, v in depth_split(root).items():
            res[k] += v
            tree_num[k] += [e] * len(v)

    for k, v in res.items():
        for e, n in enumerate(v):
            n.num = e + 1
    depthes = [x[1] for x in sorted(res.items(), key=lambda x: -x[0])]
    indices = [get_nums(nodes) for nodes in depthes]
    depthes = [np.array([n.label for n in nn], np.int32) for nn in depthes]
    tree_num = [
        np.array(x[1], np.int32) for x in sorted(tree_num.items(), key=lambda x: -x[0])
    ]
    return depthes, indices, tree_num


class Node:
    def __init__(self, label="", parent=None, children=[], num=0):
        self.label = label
        self.parent = parent
        self.children = children
        self.num = num


class TreeLSTMNode:
    def __init__(self, h=None, c=None, parent=None, children=[], num=0):
        self.label = None
        self.h = h
        self.c = c
        self.parent = parent  # TreeLSTMNode
        self.children = children  # list of TreeLSTMNode
        self.num = num


def remove_identifier(root, mark='"identifier=', replacement="$ID"):
    """remove identifier of all nodes"""
    if mark in root.label:
        root.label = replacement
    for child in root.children:
        remove_identifier(child)
    return root


def print_traverse(root, indent=0):
    """print tree structure"""
    print(" " * indent + str(root.label))
    for child in root.children:
        print_traverse(child, indent + 2)


def print_num_traverse(root, indent=0):
    """print tree structure"""
    print(" " * indent + str(root.num))
    for child in root.children:
        print_num_traverse(child, indent + 2)


def traverse(root):
    """traverse all nodes"""
    res = [root]
    for child in root.children:
        res = res + traverse(child)
    return res


def traverse_leaf(root):
    """traverse all leafs"""
    res = []
    for node in traverse(root):
        if node.children == []:
            res.append(node)
    return res


def traverse_label(root):
    """return list of tokens"""
    li = [root.label]
    for child in root.children:
        li += traverse_label(child)
    return li


def traverse_leaf_label(root):
    """traverse all leafs"""
    res = []
    for node in traverse(root):
        if node.children == []:
            res.append(node.label)
    return res


def partial_traverse(root, kernel_depth, depth=0, children=[], depthes=[], left=[]):
    """indice start from 0 and counts do from 1"""
    children.append(root.num)
    depthes.append(depth)
    if root.parent is None:
        left.append(1.0)
    else:
        num_sibs = len(root.parent.children)
        if num_sibs == 1:
            left.append(1.0)
        else:
            left.append(1 - (root.parent.children.index(root) / (num_sibs - 1)))

    if depth < kernel_depth - 1:
        for child in root.children:
            res = partial_traverse(
                child, kernel_depth, depth + 1, children, depthes, left
            )
            children, depthes, left = res

    return (children, depthes, left)


def read_pickle(path):
    return pickle.load(open(path, "rb"))


def consult_tree(root, dic):
    nodes = traverse(root)
    for n in nodes:
        n.label = dic[n.label]
    return nodes[0]


def depth_split(root, depth=0):
    """
    root: Node
    return: dict
    """
    res = defaultdict(list)
    res[depth].append(root)
    for child in root.children:
        for k, v in depth_split(child, depth + 1).items():
            res[k] += v
    return res


def depth_split_batch(roots):
    """
    roots: list of Node
    return: dict
    """
    res = defaultdict(list)
    for root in roots:
        for k, v in depth_split(root).items():
            res[k] += v
    return res


def sequence_apply(func, xs):
    """
    xs: list of [any, dim]
    return: list of func([any, dim])
    """
    x_len = [x.shape[0] for x in xs]
    ex = func(torch.cat(xs, dim=0))
    exs = torch.split(ex, x_len, dim=0)
    return exs


def he_normal():
    return init.kaiming_normal_


def orthogonal():
    return init.orthogonal_


def get_sequence_mask(xs):
    x_len = torch.tensor([x.shape[0] for x in xs], dtype=torch.int32)
    mask = (
        torch.arange(0, torch.max(x_len), dtype=torch.int32)
        .unsqueeze(0)
        .expand(x_len.shape[0], -1)
    )
    mask = mask < x_len.unsqueeze(1)
    return mask


def pad_tensor(ys):
    length = [y.shape[0] for y in ys]
    max_length = max(length)
    ys = torch.stack(
        [torch.nn.functional.pad(y, (0, 0, 0, max_length - y.shape[0])) for y in ys]
    )
    mask = torch.arange(max_length).expand(len(length), -1)
    mask = mask < torch.tensor(length).unsqueeze(1)
    return ys, mask


def depth_split_batch2(roots):
    """
    roots: list of Node
    return: dict
    """
    res = defaultdict(list)
    for root in roots:
        for k, v in depth_split(root).items():
            res[k] += v
    for k, v in res.items():
        for e, n in enumerate(v):
            n.num = e + 1
    return res


class GeneratorLen(object):
    def __init__(self, gen, length):
        self.gen = gen
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        return self.gen


def ngram(words, n):
    return list(zip(*(words[i:] for i in range(n))))


def bleu4(true, pred):
    c = len(pred)
    r = len(true)
    bp = 1.0 if c > r else np.exp(1 - r / (c + 1e-10))
    score = 0
    for i in range(1, 5):
        true_ngram = set(ngram(true, i))
        pred_ngram = ngram(pred, i)
        length = float(len(pred_ngram)) + 1e-10
        count = sum([1.0 if t in true_ngram else 0.0 for t in pred_ngram])
        score += math.log(1e-10 + (count / length))
    score = math.exp(score * 0.25)
    bleu = bp * score
    return bleu


class Datagen_tree:
    def __init__(self, X, Y, batch_size, code_dic, nl_dic, train=True, binary=False):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.code_dic = code_dic
        self.nl_dic = nl_dic
        self.train = train
        self.binary = binary

    def __len__(self):
        return len(range(0, len(self.X), self.batch_size))

    def __call__(self, epoch=0):
        return GeneratorLen(BackgroundGenerator(self.gen(epoch), 1), len(self))

    def gen(self, epoch):
        if self.train:
            np.random.seed(epoch)
            newindex = list(np.random.permutation(len(self.X)))
            X = [self.X[i] for i in newindex]
            Y = [self.Y[i] for i in newindex]
        else:
            X = [x for x in self.X]
            Y = [y for y in self.Y]
        for i in range(0, len(self.X), self.batch_size):
            x = X[i : i + self.batch_size]
            y = Y[i : i + self.batch_size]
            x_raw = [read_pickle(n) for n in x]
            if self.binary:
                x_raw = tree2binary(x_raw)
            y_raw = [[self.nl_dic[t] for t in s] for s in y]
            x = [consult_tree(n, self.code_dic) for n in x_raw]
            x_raw = [traverse_label(n) for n in x_raw]
            padded_y = [
                torch.tensor(seq[:100] + [-1.0] * max(0, 100 - len(seq))) for seq in y
            ]
            y = torch.stack(padded_y)

            yield tree2tensor(x), y, x_raw, y_raw


class Datagen_binary(Datagen_tree):
    def __init__(self, X, Y, batch_size, code_dic, nl_dic, train=True, binary=True):
        super(Datagen_binary, self).__init__(
            X, Y, batch_size, code_dic, nl_dic, train=True, binary=True
        )


class Datagen_set:
    def __init__(self, X, Y, batch_size, code_dic, nl_dic, train=True):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.code_dic = code_dic
        self.nl_dic = nl_dic
        self.train = train

    def __len__(self):
        return len(range(0, len(self.X), self.batch_size))

    def __call__(self, epoch=0):
        return GeneratorLen(BackgroundGenerator(self.gen(epoch), 1), len(self))

    def gen(self, epoch):
        if self.train:
            np.random.seed(epoch)
            newindex = list(np.random.permutation(len(self.X)))
            X = [self.X[i] for i in newindex]
            Y = [self.Y[i] for i in newindex]
        else:
            X = [x for x in self.X]
            Y = [y for y in self.Y]
        for i in range(0, len(self.X), self.batch_size):
            x = X[i : i + self.batch_size]
            y = Y[i : i + self.batch_size]
            x_raw = [read_pickle(n) for n in x]
            y_raw = [[self.nl_dic[t] for t in s] for s in y]
            x = [traverse_label(n) for n in x_raw]
            x = [np.array([self.code_dic[t] for t in xx], "int32") for xx in x]
            x_raw = [traverse_label(n) for n in x_raw]
            padded_y = [
                torch.tensor(seq[:100] + [-1.0] * max(0, 100 - len(seq))) for seq in y
            ]
            y = torch.stack(padded_y)

            yield x, y, x_raw, y_raw


def sequencing(root):
    li = ["(", root.label]
    for child in root.children:
        li += sequencing(child)
    li += [")", root.label]
    return li


class Datagen_deepcom:
    def __init__(self, X, Y, batch_size, code_dic, nl_dic, train=True):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.code_dic = code_dic
        self.nl_dic = nl_dic
        self.train = train

    def __len__(self):
        return len(range(0, len(self.X), self.batch_size))

    def __call__(self, epoch=0):
        return GeneratorLen(BackgroundGenerator(self.gen(epoch), 1), len(self))

    def gen(self, epoch):
        def pad_sequence(sequence, max_length):
            return torch.tensor(sequence + [-1] * max(0, max_length - len(sequence)))[
                :max_length
            ]

        if self.train:
            np.random.seed(epoch)
            newindex = list(np.random.permutation(len(self.X)))
            X = [self.X[i] for i in newindex]
            Y = [self.Y[i] for i in newindex]
        else:
            X = [x for x in self.X]
            Y = [y for y in self.Y]
        for i in range(0, len(self.X), self.batch_size):
            x = X[i : i + self.batch_size]
            y = Y[i : i + self.batch_size]
            x_raw = [read_pickle(n) for n in x]
            y_raw = [[self.nl_dic[t] for t in s] for s in y]
            x = [sequencing(n) for n in x_raw]
            x = [np.array([self.code_dic[t] for t in xx], "int32") for xx in x]

            x = torch.stack([pad_sequence(seq, 400) for seq in x])
            x_raw = [traverse_label(n) for n in x_raw]
            y = torch.stack([pad_sequence(seq, 100) for seq in y])

            yield x, y, x_raw, y_raw


def get_length(tensor, pad_value=-1.0):
    """tensor: [batch, max_len]"""
    mask = tensor != pad_value
    return torch.sum(mask.int(), dim=1)

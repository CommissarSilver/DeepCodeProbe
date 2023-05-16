from pycparser import c_parser, c_ast
import pandas as pd
import os
import re
import sys
from gensim.models.word2vec import Word2Vec
import pickle
from tree import ASTNode, SingleNode
import numpy as np


def get_sequences(node, sequence):
    current = SingleNode(node)
    sequence.append(current.get_token())
    for _, child in node.children():
        get_sequences(child, sequence)
    if current.get_token().lower() == "compound":
        sequence.append("End")


def get_blocks(node, block_seq):
    children = node.children()
    name = node.__class__.__name__
    if name in ["FuncDef", "If", "For", "While", "DoWhile"]:
        block_seq.append(ASTNode(node))
        if name is not "For":
            skip = 1
        else:
            skip = len(children) - 1

        for i in range(skip, len(children)):
            child = children[i][1]
            if child.__class__.__name__ not in [
                "FuncDef",
                "If",
                "For",
                "While",
                "DoWhile",
                "Compound",
            ]:
                block_seq.append(ASTNode(child))
            get_blocks(child, block_seq)
    elif name is "Compound":
        block_seq.append(ASTNode(name))
        for _, child in node.children():
            if child.__class__.__name__ not in ["If", "For", "While", "DoWhile"]:
                block_seq.append(ASTNode(child))
            get_blocks(child, block_seq)
        block_seq.append(ASTNode("End"))
    else:
        for _, child in node.children():
            get_blocks(child, block_seq)


def generate_block_seqs(self):
    if self.language == "c":
        from prepare_data import get_blocks as func
    else:
        from utils import get_blocks_v1 as func
    from gensim.models.word2vec import Word2Vec

    word2vec = Word2Vec.load(
        self.root + "/" + self.language + "/train/embedding/node_w2v_" + str(self.size)
    ).wv
    vocab = word2vec.vocab
    max_token = word2vec.syn0.shape[0]

    def tree_to_index(node):
        token = node.token
        result = [vocab[token].index if token in vocab else max_token]
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

    trees = pd.DataFrame(self.sources, copy=True)
    trees["code"] = trees["code"].apply(trans2seq)
    if "label" in trees.columns:
        trees.drop("label", axis=1, inplace=True)
    self.blocks = trees


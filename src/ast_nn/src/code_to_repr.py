# Python Standard Libraries
import os

# Third-Party Libraries
import javalang
from javalang.ast import Node
import networkx as nx
import numpy as np
from gensim.models.word2vec import Word2Vec


def code_to_ast(code: str, language: str):
    if language == "c":
        from pycparser import c_parser

        parser = c_parser.CParser()
        tree = parser.parse(code)  # parse the code to generate AST

    elif language == "java":
        import javalang

        tokens = javalang.tokenizer.tokenize(code)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()  # parse the code to generate AST

    return tree


def ast_to_index(tree, language):
    if language == "c":
        try:
            from prepare_data import get_blocks as func
        except ImportError:
            from ast_nn.src.prepare_data import get_blocks as func

        word2vec = Word2Vec.load(
            os.path.join(
                os.getcwd(),
                "src",
                "ast_nn",
                "dataset",
                "c",
                "embeddings",
                "node_w2v_128",
            )
        ).wv
        vocab = word2vec
        max_token = word2vec.vectors.shape[0]
    elif language == "java":
        try:
            from utils import get_blocks_v1 as func
        except ImportError:
            from ast_nn.src.utils import get_blocks_v1 as func

        word2vec = Word2Vec.load(
                os.path.join(
                    os.getcwd(),
                    "src",
                    "ast_nn",
                    "dataset",
                    "java",
                    "embeddings",
                    "node_w2v_128",
                )
            ).wv
        vocab = word2vec
        max_token = word2vec.vectors.shape[0]

    def tree_to_index(node):
        token = node.token
        children = node.children

        result = [vocab.key_to_index[token] if token in vocab else max_token]
        for child in children:
            result.append(tree_to_index(child))
        return result

    def trans2seq(r):
        blocks = []
        func(r, blocks)
        tree = []
        tree_info = {}

        # for unpacking the list of children
        def unpack(l):
            for el in l:
                if isinstance(el, list):
                    yield from unpack(el)
                else:
                    yield el

        for i, b in enumerate(blocks):
            block_token = b.token
            block_position = i + 1

            btree = tree_to_index(b)
            tree_info[i] = {
                "d": block_position,
                "u": vocab.key_to_index[block_token]
                if block_token in vocab
                else max_token,
                "c": list(unpack(btree)),
            }
        # unpakc all ds into a list
        ds = [tree_info[i]["d"] for i in range(len(tree_info))]
        cs = [tree_info[i]["c"] for i in range(len(tree_info))]
        us = [tree_info[i]["u"] for i in range(len(tree_info))]

        return {"d": ds, "c": cs, "u": us}

    return trans2seq(tree)


def code_to_index(code, language):
    tree = code_to_ast(code, language)
    return ast_to_index(tree, language)


if __name__ == "__main__":
    # Define Test Cases
    c_code_sample = "int main() { return 0; }"
    java_code_sample = "public static void main(String[] args) { }"

    # Test 1: C Language
    c_output = code_to_index(c_code_sample, "c")
    assert c_output is not None, "Test 1 Failed: C language output is None"
    assert (
        "d" in c_output and "c" in c_output and "u" in c_output
    ), "Test 1 Failed: C language output lacks required keys"

    # Test 2: Java Language
    java_output = code_to_index(java_code_sample, "java")
    assert java_output is not None, "Test 2 Failed: Java language output is None"
    assert (
        "d" in java_output and "c" in java_output and "u" in java_output
    ), "Test 2 Failed: Java language output lacks required keys"

    print("\033[32m All tests passed.\033[0m")

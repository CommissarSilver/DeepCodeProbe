import networkx as nx
import javalang
from javalang.ast import Node
import numpy as np
from gensim.models.word2vec import Word2Vec

word2vec = Word2Vec.load(
    "/Users/ahura/Nexus/Leto/src/ast_nn/dataset/java/embeddings/node_w2v_128"
).wv
vocab = word2vec.vocab
max_token = word2vec.syn0.shape[0]


def code_to_ast(code: str, language: str):
    if language == "c":
        from pycparser import c_parser

        tree = c_parser.parse(code)  # parse the code to generate AST

    elif language == "java":
        import javalang

        tokens = javalang.tokenizer.tokenize(code)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()  # parse the code to generate AST

    return tree


def ast_to_index(tree, language):
    if language == "c":
        from prepare_data import get_blocks as func
    elif language == "java":
        try:
            from utils import get_blocks_v1 as func
        except ImportError:
            from ast_nn.src.utils import get_blocks_v1 as func

    def tree_to_index(node):
        token = node.token
        children = node.children

        result = [vocab[token].index if token in vocab else max_token]
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
                "u": vocab[block_token].index if block_token in vocab else max_token,
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
    # tree = code_to_ast(
    #     "public static void pipe(InputStream in, OutputStream out) throws IOException {\n        byte[] buf = new byte[4096];\n        int n;\n        while ((n = in.read(buf, 0, buf.length)) >= 0) out.write(buf, 0, n);\n    }\n",
    #     "java",
    # )
    # y = ast_to_index(
    #     "/Users/ahura/Nexus/Leto/src/ast_nn/dataset/java/embeddings/node_w2v_128",
    #     tree,
    #     "java",
    # )
    x = code_to_index(
        "public static void pipe(InputStream in, OutputStream out) throws IOException {\n        byte[] buf = new byte[4096];\n        int n;\n        while ((n = in.read(buf, 0, buf.length)) >= 0) out.write(buf, 0, n);\n    }\n",
        "java",
    )
    print("i")

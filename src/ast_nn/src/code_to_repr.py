import networkx as nx
import javalang
from javalang.ast import Node


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


def ast_to_index(word2vec_path, tree, language):
    from gensim.models.word2vec import Word2Vec

    if language == "c":
        from prepare_data import get_blocks as func
    elif language == "java":
        from utils import get_blocks_v1 as func

    word2vec = Word2Vec.load(word2vec_path).wv
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

    return trans2seq(tree)



tree = code_to_ast(
    "public String getValue(String key) {\n        KeyValue kv = getKV(key);\n        return kv == null ? null : kv.getValue();\n    }",
    "java",
)
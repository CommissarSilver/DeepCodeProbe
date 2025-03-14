# Python Standard Libraries
import os
import pickle

# Third-Party Libraries
import javalang
from javalang.ast import Node
import networkx as nx
import numpy as np
from gensim.models.word2vec import Word2Vec
from pycparser import c_parser, c_ast
import javalang


class CVisitor(c_ast.NodeVisitor):
    def __init__(self):
        self.connections = {}
        self.node_list = []  # List to store nodes in the order they are visited
        self.node_types = []

    def get_node_id(self, node):
        if node.coord:
            return f"{type(node).__name__} at {node.coord}"
        else:
            return f"{type(node).__name__}"

    def update_connections(self, parent, child):
        parent_id = self.get_node_id(parent)
        child_id = self.get_node_id(child)

        # Add the parent node if not already in the list
        if parent_id not in self.connections:
            self.connections[parent_id] = []
            self.node_list.append(parent_id)

        # Add the child node if not already in the list
        if child_id not in self.connections:
            self.connections[child_id] = []
            self.node_list.append(child_id)

        # Record the connection
        self.connections[parent_id].append(child_id)

    def visit(self, node):
        for c_name, c in node.children():
            self.update_connections(node, c)
            self.visit(c)

    def get_connections(self):
        # Create a binary list indicating if a node has children
        binary_list = [
            1 if children else 0 for node_id, children in self.connections.items()
        ]
        self.get_node_types()

        return binary_list

    def get_node_types(self):
        self.node_types = [
            node_coord.split("at")[0].replace(" ", "") for node_coord in self.node_list
        ]


class JavaVisitor:
    def __init__(self):
        self.connections = {}
        self.node_list = []

    def get_node_id(self, node):
        # Using the type of the node as its identifier
        return type(node).__name__

    def update_connections(self, parent, child):
        parent_id = self.get_node_id(parent)
        child_id = self.get_node_id(child)

        if parent_id not in self.connections:
            self.connections[parent_id] = []
            self.node_list.append(parent_id)

        if child_id not in self.connections:
            self.connections[child_id] = []
            self.node_list.append(child_id)

        self.connections[parent_id].append(child_id)

    def visit(self, node):
        # Check if the node is an instance of the base Node class
        if not isinstance(node, javalang.tree.Node):
            return

        for child in node.children:
            if child:
                if isinstance(child, javalang.tree.Node):
                    self.update_connections(node, child)
                    self.visit(child)
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(item, javalang.tree.Node):
                            self.update_connections(node, item)
                            self.visit(item)

    def get_connections(self):
        binary_list = [
            1 if self.connections[node_id] else 0 for node_id in self.node_list
        ]
        return binary_list

    def parse_java(self, code):
        tokens = list(javalang.tokenizer.tokenize(code))
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()
        self.visit(tree)


class TokenIndexerJava:
    def __init__(self) -> None:
        self.tokens = set()
        self.indxes = {}

    def add_token(self, token):
        self.tokens.add(token)
        if token not in self.indxes:
            self.indxes[token] = len(self.indxes)

    def get_index(self, token):
        return self.indxes[token]

    def map_tokens_to_indices(self, token_list):
        # if the token is not in the dictionary, add it
        for token in token_list:
            if token not in self.tokens:
                self.add_token(token)
        # return the list of indices
        return [self.get_index(token) for token in token_list]


class TokenIndexerC:
    def __init__(self) -> None:
        self.tokens = set()
        self.indxes = {}

    def add_token(self, token):
        self.tokens.add(token.split("at")[0])
        if token not in self.indxes:
            self.indxes[token] = len(self.indxes)

    def get_index(self, token):
        return self.indxes[token]

    def map_tokens_to_indices(self, token_list):
        # if the token is not in the dictionary, add it
        for token in token_list:
            if token.split("at")[0].replace(" ", "") not in self.tokens:
                self.add_token(token.split("at")[0].replace(" ", ""))
        # return the list of indices
        return [
            self.get_index(token.split("at")[0].replace(" ", ""))
            for token in token_list
        ]


def code_to_ast(code: str, language: str):
    if language == "c":
        parser = c_parser.CParser()
        tree = parser.parse(code)  # parse the code to generate AST
        visitor = CVisitor()
        visitor.visit(tree)
        visitor.get_node_types()

    elif language == "java":
        tokens = javalang.tokenizer.tokenize(code)
        parser = javalang.parser.Parser(tokens)
        tree = parser.parse_member_declaration()  # parse the code to generate AST
        visitor = JavaVisitor()
        visitor.parse_java(code)

    return tree, visitor


def ast_to_index(tree, visitor, language, indexer):
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

    def trans2seq(r, v):
        blocks = []
        func(r, blocks)
        tree = []
        tree_info = {}

        ds = []
        cs = v.get_connections()
        us = v.node_list

        return {"d": ds, "c": cs, "u": indexer.map_tokens_to_indices(us)}

    return trans2seq(tree, visitor)


def code_to_index(code, language, indexer):
    try:
        tree, out = code_to_ast(code, language)
        return ast_to_index(tree, out, language, indexer)
    except:
        return {"d": [], "c": [], "u": []}


if __name__ == "__main__":
    c_code_sample = """int main()
{
        int a;
        int bai,wushi,ershi,shi,wu,yi;
        cin>>a;
        bai=a/100;
        a=a%100;
        wushi=a/50;
        a=a%50;
        ershi=a/20;
        a=a%20;
        shi=a/10;
        a=a%10;
        wu=a/5;
        a=a%5;
        yi=a;
        cout<<bai<<endl;
        cout<<wushi<<endl;
        cout<<ershi<<endl;
        cout<<shi<<endl;
        cout<<wu<<endl;
        cout<<yi<<endl;
        return 0;
}"""
    java_code_sample = """
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!"); 
    }
}
"""

    # # Test 1: C Language
    c_output = code_to_index(c_code_sample, "c", TokenIndexerC())
    assert c_output is not None, "Test 1 Failed: C language output is None"
    assert (
        "d" in c_output and "c" in c_output and "u" in c_output
    ), "Test 1 Failed: C language output lacks required keys"

    # Test 2: Java Language
    java_output = code_to_index(java_code_sample, "java", TokenIndexerJava())
    assert java_output is not None, "Test 2 Failed: Java language output is None"
    assert (
        "d" in java_output and "c" in java_output and "u" in java_output
    ), "Test 2 Failed: Java language output lacks required keys"

    print("\033[32m All tests passed.\033[0m")

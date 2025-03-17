# Python Standard Libraries
import os
import pickle

# Third-Party Libraries
from infercode.data_utils.ast_parser import ASTParser
from infercode.data_utils.ast_util import ASTUtil
from infercode.data_utils.tensor_util import TensorUtil


def code_to_ast(code: str, language: str):
    if language == "c":
        parser = ASTParser(language="c")
        tree = parser.parse(code)  # parse the code to generate AST

    elif language == "java":
        parser = ASTParser(language="java")
        tree = parser.parse(code)  # parse the code to generate AST
    return tree


def ast_to_index(tree, language, code_snippet):
    tensor_util = TensorUtil()

    if language == "c":
        node_type_vocab_model_prefix = "c_node_type_vocab"
        node_token_vocab_model_prefix = "c_node_token_vocab"
    elif language == "java":
        node_type_vocab_model_prefix = "java_node_type_vocab"
        node_token_vocab_model_prefix = "java_node_token_vocab"

    ast_util = ASTUtil(
        node_type_vocab_model_path=node_type_vocab_model_prefix + ".model",
        node_token_vocab_model_path=node_token_vocab_model_prefix + ".model",
    )

    tree_representation, _ = ast_util.simplify_ast(
        tree,
        code_snippet.decode("utf-8"),
    )

    tree_indexes = tensor_util.transform_tree_to_index(tree_representation)

    return {
        "d": tree_indexes["node_index"],
        "c": tree_indexes["children_index"],
        "u": tree_indexes["node_type_id"],
    }


def code_to_index(code, language):
    try:
        tree = code_to_ast(code, language)
        return ast_to_index(tree, language, code)
    except Exception as e:
        print(e)
        return {"d": [], "c": [], "u": []}


if __name__ == "__main__":
    c_code_sample = b"""int main()
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

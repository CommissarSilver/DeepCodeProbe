import os, logging

from collections import Counter
from tree_sitter import Language, Parser

from .binary_tree import ast2binary, tree_to_distance
from .code2ast import code2ast, get_tokens_ast


logger = logging.getLogger("data")

PY_LANGUAGE = Language(
    os.path.join(os.getcwd(), "src", "ast_probe", "grammars", "languages.so"), "python"
)
JAVA_LANGUAGE = Language(
    os.path.join(os.getcwd(), "src", "ast_probe", "grammars", "languages.so"), "java"
)

PY_PARSER = Parser()
PY_PARSER.set_language(PY_LANGUAGE)
JAVA_PARSER = Parser()
JAVA_PARSER.set_language(JAVA_LANGUAGE)


def convert_sample_to_features(code, parser, lang):
    try:
        G, pre_code = code2ast(code, parser, lang)
        binary_ast = ast2binary(G)
        d, c, _, u = tree_to_distance(binary_ast, 0)
        code_tokens = get_tokens_ast(G, pre_code)

        logger.info(f"Successfully converted sample to features")
        logger.debug(
            f"Input code language: ( {lang} ) - code: ( {code} ) - (d,c,u) = ( {d}, {c}, {u} )"
        )
    except Exception as e:
        logger.exception(f"Error while converting sample to features: {e}")
        pass

    return {
        "d": d,
        "c": c,
        "u": u,
        "num_tokens": len(code_tokens),
        "code_tokens": code_tokens,
    }


def get_non_terminals_labels(train_set_labels, valid_set_labels, test_set_labels):
    all_labels = (
        [label for seq in train_set_labels for label in seq]
        + [label for seq in valid_set_labels for label in seq]
        + [label for seq in test_set_labels for label in seq]
    )
    # use a Counter to constantly get the same order in the labels
    ct = Counter(all_labels)
    labels_to_ids = {}
    for i, label in enumerate(ct):
        labels_to_ids[label] = i
    return labels_to_ids


def convert_to_ids(c, column_name, labels_to_ids):
    labels_ids = []
    for label in c:
        labels_ids.append(labels_to_ids[label])
    return {column_name: labels_ids}

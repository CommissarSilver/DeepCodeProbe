import ast, re

import ast

u_index = {}
u_index_counter = 0


def extract_node_info(node, parent=None, position=0):
    """
    Recursively extract node categories from the AST.
    """
    node_name = node.__class__.__name__
    node_position = position

    # Gather information about the node
    node_info = {"name": node_name, "position": node_position}

    # Add parent information if available
    if parent is not None:
        node_info["parent"] = parent.__class__.__name__

    # Recursively process child nodes
    for child_name, child_node in ast.iter_fields(node):
        if isinstance(child_node, list):
            for idx, child in enumerate(child_node):
                if isinstance(child, ast.AST):
                    child_info = extract_node_info(child, node, idx)
                    if "children" not in node_info:
                        node_info["children"] = []
                    node_info["children"].append(child_info)
        elif isinstance(child_node, ast.AST):
            child_info = extract_node_info(child_node, node, 0)
            if "children" not in node_info:
                node_info["children"] = []
            node_info["children"].append(child_info)

    return node_info


def extract_lists(ast_info):
    """
    Extract lists D, C, and U from AST information.
    """
    D = []
    C = []
    U = []

    # Helper function to recursively process AST information
    def process_node(node_info):
        D.append(node_info["position"])
        U.append(node_info["name"])

        if "children" in node_info:
            children = node_info["children"]
            C.append(len(children))
            for child in children:
                process_node(child)
        else:
            C.append(0)

    # Process the AST information
    process_node(ast_info)

    return D, C, U


def code_to_ast(code: str):
    code = code.replace(" DCNL DCSP ", "\n\t")
    code = code.replace(" DCNL  DCSP ", "\n\t")
    code = code.replace(" DCNL ", "\n")
    code = code.replace(" DCSP ", "\t")

    # create the ast
    try:
        tree = ast.parse(code)
        node_info = extract_node_info(tree)
        D, C, U = extract_lists(node_info)

        return D, C, U
    except Exception as e:
        return [], [], []


def ast_to_index(
    D: list,
    C: list,
    U: list,
):
    global u_index, u_index_counter
    for label in U:
        if label not in u_index.keys():
            u_index[label] = u_index_counter
            u_index_counter += 1

    final_d = D
    final_c = C
    final_u = [u_index[label] for label in U]

    return final_d, final_c, final_u


def code_to_index(code: str):
    D, C, U = code_to_ast(code)
    if (D, C, U) == ([], [], []):
        return {"d": [], "c": [], "u": []}
    else:
        D, C, U = ast_to_index(D, C, U)
        return {"d": D, "c": C, "u": U}


if __name__ == "__main__":
    path_to_code = (
        "/Users/ahura/Nexus/Leto/src/code_sum_drl/dataset/train/train0.60.20.2.code"
    )
    codes = open(path_to_code).readlines()
    errors = 0
    no_errors = 0
    for code in codes:
        # ds, cs, us = code_to_ast(code)
        # ds, cs, us = ast_to_index(ds, cs, us)
        test = code_to_index(code)
        if (test["d"], test["c"], test["u"]) == ([], [], []):
            errors += 1
        else:
            no_errors += 1

    print("errors: ", errors)
    print("no_errors: ", no_errors)

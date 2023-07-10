import ast, re

import ast


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
    Extract lists D, c, and u from AST information.
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


def code_to_ast(path_to_code: str):
    Ds = {}
    Cs = {}
    Us = {}

    codes = open(path_to_code).readlines()

    codes = [code.replace(" DCNL DCSP ", "\n\t") for code in codes]
    codes = [code.replace(" DCNL  DCSP ", "\n\t") for code in codes]
    codes = [code.replace(" DCNL ", "\n") for code in codes]
    codes = [code.replace(" DCSP ", "\t") for code in codes]

    fault_counter = 0
    for code_index, code in enumerate(codes):
        # create the ast
        try:
            tree = ast.parse(code)
            j = extract_node_info(tree)
            D, C, U = extract_lists(j)

            Ds[code_index] = D
            Cs[code_index] = C
            Us[code_index] = U
        except Exception as e:
            fault_counter += 1
            continue
    print("total codes: ", len(codes))
    print("faulty codes: ", fault_counter)
    return Ds, Cs, Us


def ast_to_index(
    all_d: dict,
    all_c: dict,
    all_u: dict,
):
    final_d = []
    final_c = []
    final_u = []
    u_index = {}
    u_index_counter = 0

    for code_index, code_ast_labels in all_u.items():
        for label in code_ast_labels:
            if label not in u_index.keys():
                u_index[label] = u_index_counter
                u_index_counter += 1
    for d, c, u in zip(all_d.values(), all_c.values(), all_u.values()):
        final_d.append(d)
        final_c.append(c)
        final_u.append([u_index[label] for label in u])

    return final_d, final_c, final_u, u_index


if __name__ == "__main__":
    ds, cs, us = code_to_ast(
        "/Users/ahura/Nexus/Leto/src/code_sum_drl/dataset/train/train0.60.20.2.code"
    )
    s = ast_to_index(ds, cs, us)

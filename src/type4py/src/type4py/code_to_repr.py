import ast
import json

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
            C.append(1 if len(children) > 1 else 0)
            for child in children:
                process_node(child)
        else:
            C.append(0)

    # Process the AST information
    process_node(ast_info)

    return D, C, U


def code_to_ast(code: str):
    # create the ast
    try:
        code = code.replace("[EOL] ", "\n")

        # Simple approach specifically for this dataset format
        lines = code.split("\n")
        indented_lines = []
        indent_stack = [0]  # Stack to track indentation levels

        for line in lines:
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                indented_lines.append("")
                continue

            # Check for block ending statements
            if (
                stripped.startswith("return")
                or stripped.startswith("break")
                or stripped.startswith("continue")
            ):
                indented_lines.append("    " * indent_stack[-1] + stripped)
                if len(indent_stack) > 1:  # Don't remove the base level
                    indent_stack.pop()
                continue

            # Check if this is a block starter
            if (
                stripped.startswith("def ")
                or stripped.startswith("if ")
                or stripped.startswith("for ")
                or stripped.startswith("while ")
                or stripped.startswith("class ")
                or stripped == "else:"
                or stripped.startswith("elif ")
            ) and stripped.endswith(":"):

                indented_lines.append("    " * indent_stack[-1] + stripped)
                # Push a new indentation level
                indent_stack.append(indent_stack[-1] + 1)
            else:
                # Regular line within current block
                indented_lines.append("    " * indent_stack[-1] + stripped)

        code = "\n".join(indented_lines)

        # Apply formatting replacements (but keep them for function definitions)
        for i, line in enumerate(indented_lines):
            if not line.startswith(("def ", "    def ")):
                indented_lines[i] = (
                    line.replace(" = ", "=")
                    .replace(" . ", ".")
                    .replace(" , ", ",")
                    .replace(" (", "(")
                    .replace("( ", "(")
                    .replace(" )", ")")
                )

        code = (
            "\n".join(indented_lines)
            .replace("[EOL]", "")
            .replace(" [EOL]", "")
            .replace(" [EOL]", "")
        )
        print(code)
        # Parse with ast
        tree = ast.parse(code)
        node_info = extract_node_info(tree)
        D, C, U = extract_lists(node_info)

        return D, C, U
    except Exception as e:
        # Fallback to a more primitive approach
        try:
            # Indentation based on colons
            lines = code.replace("[EOL] ", "\n").split("\n")
            result = []
            indent = 0

            for line in lines:
                if not line.strip():
                    result.append("")
                    continue

                if ":" in line and line.rstrip().endswith(":"):
                    result.append("    " * indent + line.strip())
                    indent += 1
                else:
                    result.append("    " * indent + line.strip())

            fixed_code = "\n".join(result)
            tree = ast.parse(fixed_code)
            node_info = extract_node_info(tree)
            D, C, U = extract_lists(node_info)
            return D, C, U
        except:
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
    path_to_code = "/type4py/dataset/ManyTypes4PyDataset-v0.7/processed_projects_complete/0hoosnowball.json"
    codes = json.load(open(path_to_code))
    repo_code = codes["0hoo/snowball"]["src_files"][
        "repos/0hoo/snowball/core_sample.py"
    ]["untyped_seq"]

    errors = 0
    no_errors = 0
    for code in codes:
        # ds, cs, us = code_to_ast(code)
        # ds, cs, us = ast_to_index(ds, cs, us)
        test = code_to_index(repo_code)
        print(test)

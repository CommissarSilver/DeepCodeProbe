import javalang, json

u_index = {}
u_index_counter = 0


def code_to_ast(code_json: str):
    num_faulty_codes = 0
    D = []
    C = []
    U = []

    json_data = json.loads(code_json)["code"]
    try:
        tokens = javalang.tokenizer.tokenize(json_data)
        parser = javalang.parser.Parser(tokens)

        tree = parser.parse_member_declaration()

        node_names = [node.__class__.__name__ for path, node in tree]
        node_children = [len(node.children) for path, node in tree]
        node_positions = [
            (node.position.line, node.position.column)
            for path, node in tree
            if node.position is not None
        ]
        D = node_positions
        C = node_children
        U = node_names
    except Exception as e:
        pass

    return D, C, U


def ast_to_index(D, C, U):
    final_d = []
    final_c = []
    final_u = []
    global u_index, u_index_counter

    for label in U:
        if label not in u_index.keys():
            u_index[label] = u_index_counter
            u_index_counter += 1
    final_d = D
    final_c = C
    final_u = [u_index[label] for label in U]

    return final_d, final_c, final_u


if __name__ == "__main__":
    path_to_code = "/Users/ahura/Nexus/Leto/src/summarization_tf/dataset/valid.json"
    codes = open(path_to_code).readlines()
    for code in codes:
        ds, cs, us = code_to_ast(code)
        ds, cs, us = ast_to_index(ds, cs, us)

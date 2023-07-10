import javalang, json


def code_to_ast(path_to_code: str):
    all = open(path_to_code).readlines()
    num_faulty_codes = 0
    for i in all:
        json_data = json.loads(i)["code"]
        try:
            tokens = javalang.tokenizer.tokenize(json_data)
            parser = javalang.parser.Parser(tokens)

            tree = parser.parse_member_declaration()

            node_names = [node.__class__.__name__ for path, node in tree]
            node_children = [len(node.children) for path, node in tree]
            node_positions = [node.position for path, node in tree]
        except Exception as e:
            print(e)
            num_faulty_codes += 1
            continue
    print(num_faulty_codes)


def ast_to_index():
    pass


if __name__ == "__main__":
    code_to_ast("/Users/ahura/Nexus/Leto/src/summarization_tf/dataset/test.json")

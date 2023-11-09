import javalang, json, os
from summarization_tf.src.utils import Datagen_tree, read_pickle


u_index = {}
u_index_counter = 0
dataset_path = "/store/travail/vamaj/Leto/src/summarization_tf/dataset"
trn_data = read_pickle(f"{dataset_path}/nl/train.pkl")
code_indexes = list(trn_data.keys())
code_indexes = [int(i.split("/")[-1]) for i in code_indexes]
vld_data = read_pickle(f"{dataset_path}/nl/valid.pkl")
tst_data = read_pickle(f"{dataset_path}/nl/test.pkl")
code_i2w = read_pickle(f"{dataset_path}/code_i2w.pkl")
code_w2i = read_pickle(f"{dataset_path}/code_w2i.pkl")
nl_i2w = read_pickle(f"{dataset_path}/nl_i2w.pkl")
nl_w2i = read_pickle(f"{dataset_path}/nl_w2i.pkl")

trn_x, trn_y_raw = zip(*sorted(trn_data.items()))
vld_x, vld_y_raw = zip(*sorted(vld_data.items()))
tst_x, tst_y_raw = zip(*sorted(tst_data.items()))

# trn_y = [
#     [nl_w2i[t] if t in nl_w2i.keys() else nl_w2i["<UNK>"] for t in l] for l in trn_y_raw
# ]
# vld_y = [
#     [nl_w2i[t] if t in nl_w2i.keys() else nl_w2i["<UNK>"] for t in l] for l in vld_y_raw
# ]
# tst_y = [
#     [nl_w2i[t] if t in nl_w2i.keys() else nl_w2i["<UNK>"] for t in l] for l in tst_y_raw
# ]


def code_to_ast(code: str):
    num_faulty_codes = 0
    D = []
    C = []
    U = []

    # json_data = json.loads(code_json)["code"]
    try:
        tokens = javalang.tokenizer.tokenize(code)
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


def code_to_index(code: str, nl: str, code_index: int):
    if code_index in code_indexes:
        if os.path.exists(
            f"/store/travail/vamaj/Leto/src/summarization_tf/dataset/tree/train/{code_index+1}"
        ):
            trn_y_code = [
                nl_w2i[t] if t in nl_w2i.keys() else nl_w2i["<UNK>"] for t in nl.split()
            ]

            trn_gen = Datagen_tree(
                [
                    f"/store/travail/vamaj/Leto/src/summarization_tf/dataset/tree/train/{code_index+1}"
                ],
                [trn_y_code],
                1,
                code_w2i,
                nl_i2w,
                train=True,
            )

            generator_object = trn_gen(epoch=code_index)
            a = list(generator_object)
            tree_tensor, y, x_raw, y_raw = a[0][0], a[0][1], a[0][2], a[0][3]

            D, C, U = code_to_ast(code)
            D, C, U = ast_to_index(D, C, U)

            return {
                "d": D,
                "c": C,
                "u": U,
                "tree_tensor": tree_tensor,
                "y_raw": y_raw,
            }
        else:
            return None


if __name__ == "__main__":
    path_to_code = "/Users/ahura/Nexus/Leto/src/summarization_tf/dataset/valid.json"
    codes = open(path_to_code).readlines()
    for code in codes:
        ds, cs, us = code_to_ast(code)
        ds, cs, us = ast_to_index(ds, cs, us)

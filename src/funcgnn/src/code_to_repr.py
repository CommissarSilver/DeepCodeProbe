import json, glob, torch, os
import numpy as np

src_path = os.path.join(os.getcwd(), "src", "funcgnn", "dataset")


def process_pair(path):
    """
    Reading a json file with a pair of graphs.
    :param path: Path to a JSON file.
    :return data: Dictionary with data.
    """
    data = json.load(open(path))
    return data


training_graphs = glob.glob(os.path.join(src_path, "train") + "/*.json")
testing_graphs = glob.glob(os.path.join(src_path, "test") + "/*.json")
random_graphs = glob.glob(os.path.join(src_path, "test") + "/*.json")

graph_pairs = training_graphs + testing_graphs
global_labels = set()

for graph_pair in graph_pairs:
    data = process_pair(graph_pair)
    global_labels = global_labels.union(set(data["labels_1"]))
    global_labels = global_labels.union(set(data["labels_2"]))
# generate the tokens for the statements in the code. THIS COMES FROM FUNCGNN AND IS NOT MINE
global_labels = list(global_labels)
global_labels = {val: index for index, val in enumerate(global_labels)}
number_of_labels = len(global_labels)


def code_to_graph(data: dict):
    """
    generates the d,c,u tuple for the graph repr of the code
    """
    new_data = dict()

    edges_1 = data["graph_1"] + [[y, x] for x, y in data["graph_1"]]
    # this sohould be preset
    edges_2 = data["graph_2"] + [[y, x] for x, y in data["graph_2"]]

    edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
    edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)

    d = [i + 1 for i in range(len(data["labels_1"]))]
    c = data["graph_1"] + [[y, x] for x, y in data["graph_1"]]
    u = [global_labels[n] for n in data["labels_1"]]

    return {"d": d, "c": c, "u": u}


def code_to_index(data: dict):
    return code_to_graph(data)


if __name__ == "__main__":
    j = code_to_index(
        json.loads(
            open(
                os.path.join(
                    src_path, "test", "addTwoArrays_DC_EQ_m3___addTwoArrays_L_EQ_m4.json"
                )
            ).read()
        )
    )

    assert (
        "d" and "c" and "u" in j.keys()
    ), "Problem. This needs to be inspected manually. Sorry."

    print("\033[32m All tests passed.\033[0m")

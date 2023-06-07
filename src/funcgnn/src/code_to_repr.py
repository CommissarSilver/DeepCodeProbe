import json, glob, torch
import numpy as np


def process_pair(path):
    """
    Reading a json file with a pair of graphs.
    :param path: Path to a JSON file.
    :return data: Dictionary with data.
    """
    data = json.load(open(path))
    return data


training_graphs = glob.glob(
    "/Users/ahura/Nexus/Leto/src/funcgnn/dataset/train/" + "*.json"
)
testing_graphs = glob.glob("/Users/ahura/Nexus/Leto/src/funcgnn/dataset/test/" + "*.json")
random_graphs = glob.glob("/Users/ahura/Nexus/Leto/src/funcgnn/dataset/test/" + "*.json")

graph_pairs = training_graphs + testing_graphs
global_labels = set()

for graph_pair in graph_pairs:
    data = process_pair(graph_pair)
    global_labels = global_labels.union(set(data["labels_1"]))
    global_labels = global_labels.union(set(data["labels_2"]))
global_labels = list(global_labels)
global_labels = {val: index for index, val in enumerate(global_labels)}
number_of_labels = len(global_labels)


def code_to_graph(data: dict):
    new_data = dict()

    edges_1 = data["graph_1"] + [[y, x] for x, y in data["graph_1"]]
    # this sohould be preset
    edges_2 = data["graph_2"] + [[y, x] for x, y in data["graph_2"]]

    edges_1 = torch.from_numpy(np.array(edges_1, dtype=np.int64).T).type(torch.long)
    edges_2 = torch.from_numpy(np.array(edges_2, dtype=np.int64).T).type(torch.long)

    features_1, features_2 = [], []
    d = [i + 1 for i in range(len(data["labels_1"]))]
    c = data["graph_1"] + [[y, x] for x, y in data["graph_1"]]
    u = [global_labels[n] for n in data["labels_1"]]
    # for n in data["labels_1"]:
    #     features_1.append(
    #         [1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()]
    #     )

    # for n in data["labels_2"]:
    #     features_2.append(
    #         [1.0 if global_labels[n] == i else 0.0 for i in global_labels.values()]
    #     )

    # features_1 = torch.FloatTensor(np.array(features_1))
    # features_2 = torch.FloatTensor(np.array(features_2))

    # new_data["edge_index_1"] = edges_1
    # new_data["edge_index_2"] = edges_2

    # new_data["features_1"] = features_1
    # new_data["features_2"] = features_2

    return {"d": d, "c": c, "u": u}


def code_to_index(data: dict):
    return code_to_graph(data)


if __name__ == "__main__":
    j = code_to_index(
        "/Users/ahura/Nexus/Leto/src/funcgnn/dataset/test/addTwoArrays_DC_EQ_m3::::addTwoArrays_L_EQ_m4.json"
    )
    print(j)

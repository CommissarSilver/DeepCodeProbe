import json,glob
def process_pair(path):
    """
    Reading a json file with a pair of graphs.
    :param path: Path to a JSON file.
    :return data: Dictionary with data.
    """
    data = json.load(open(path))
    return data

training_graphs = glob.glob("./dataset/train/" + "*.json")
testing_graphs = glob.glob("./dataset/test/" + "*.json")
random_graphs = glob.glob("./dataset/test/" + "*.json")
graph_pairs = training_graphs + testing_graphs
global_labels = set()
for graph_pair in graph_pairs:
    data = process_pair(graph_pair)
    global_labels = global_labels.union(set(data["labels_1"]))
    global_labels = global_labels.union(set(data["labels_2"]))
global_labels = list(global_labels)
for val,index in enumerate(global_labels):
    j={val:index}
    print('h')
global_labels = {val: index for index, val in enumerate(global_labels)}
number_of_labels = len(global_labels)
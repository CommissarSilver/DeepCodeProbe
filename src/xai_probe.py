import argparse
import json
import logging
import logging.config
import os
import pickle
import warnings

import numpy as np
import pandas as pd
import pickle5 as pickle
import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import probe_utils

warnings.filterwarnings("ignore")
with open(os.path.join(os.getcwd(), "src", "logging_config.yaml"), "r") as f:
    config = yaml.safe_load(f.read())

logging.config.dictConfig(config)

parser = argparse.ArgumentParser(
    prog="Validate Probe",
    description="Validate Probe",
)
#### Arguemnt Parser ####
parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Whether to use CPU or GPU for training and evaluating the probe",
)
parser.add_argument(
    "--model",
    type=str,
    help="Model to probe",
    choices=["ast_nn", "funcgnn", "summarization_tf", "code_sum_drl", "cscg_dual"],
    default="funcgnn",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    help="Path to the dataset - Path follows the format /model_name/dataset",
    default=os.path.join(os.getcwd(), "src", "funcgnn", "dataset"),
)
parser.add_argument(
    "--language",
    type=str,
    help="Language of the dataset. Only for AST-NN",
    choices=["java", "c"],
    default="java",
)
parser.add_argument(
    "--train_epochs",
    type=int,
    default=100,
    help="Number of epochs to train the probe",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=32,
    help="Batch size for training the probe",
)
parser.add_argument(
    "--patience",
    type=int,
    default=5,
    help="Patience for early stopping",
)
parser.add_argument(
    "--probe_rank",
    type=int,
    default=128,
    choices=[128, 512],
    help="Rank of the probe. 128 for AST-NN and FuncGNN, 512 for SumTF and CodeSumDRL",
)
parser.add_argument(
    "--probe_hidden_dim",
    type=int,
    default=64,
    choices=[200, 64, 512],
    help="Hidden dimension of the probe. 200 for AST-NN. 64 for FuncGnn, 512 for SumTF, 512 for CodeSumDRL",
)
args = parser.parse_args()
#### Arguemnt Parser ####
device = args.device

if args.model == "ast_nn":
    pass
elif args.model == "funcgnn":
    import json

    import pandas as pd

    from funcgnn.src.code_to_repr import code_to_index, code_to_index_single
    from funcgnn.src.funcgnn import funcGNNTrainer
    from funcgnn.src.param_parser import parameter_parser

    # functions for calculating D,C,U similarity
    def pad_list(short_list: list, target_length: int) -> list:
        """
        Pads a given list with -1 values to reach the target length.

        Args:
            short_list (list): The list to be padded.
            target_length (int): The desired length of the padded list.

        Returns:
            list: The padded list.
        """
        return short_list + [-1] * (target_length - len(short_list))

    def cosine_similarity(list1: list, list2: list) -> float:
        """
        Compute the cosine similarity between two lists.

        Args:
            list1 (list): The first list.
            list2 (list): The second list.

        Returns:
            similarity (float): The cosine similarity between the two lists.
        """
        # Pad the shorter list
        if len(list1) > len(list2):
            list2 = pad_list(list2, len(list1))
        elif len(list2) > len(list1):
            list1 = pad_list(list1, len(list2))

        # Convert lists to numpy arrays
        vec1 = np.array(list1)
        vec2 = np.array(list2)

        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        similarity = dot_product / (norm_vec1 * norm_vec2)

        return similarity

    def average_cosine_similarity(C1: list, C2: list) -> float:
        """
        Calculate the average cosine similarity between two lists of lists.

        Args:
            C1 (list): The first list of lists.
            C2 (list): The second list of lists.

        Returns:
            avg_similarity (float): The average cosine similarity between the two lists of lists.
        """
        # Determine the length of the longest list
        max_length = max(len(C1), len(C2))

        # Pad the shorter list with zero vectors
        while len(C1) < max_length:
            C1.append([-1] * len(C1[0]))
        while len(C2) < max_length:
            C2.append([-1] * len(C2[0]))

        similarities = [cosine_similarity(vec1, vec2) for vec1, vec2 in zip(C1, C2)]

        # Calculate average similarity
        avg_similarity = sum(similarities) / len(similarities)
        return avg_similarity

    def calculate_similarity(programs: list) -> tuple:
        """
        Calculate the similarity between programs based on their 'd', 'c', and 'u' attributes.

        Args:
            programs (list): A list of dictionaries representing programs, where each dictionary
                             contains 'd', 'c', and 'u' attributes.

        Returns:
            tuple: A tuple containing the average cosine similarity between 'd' attributes,
                   the average cosine similarity between 'c' attributes, and the average cosine
                   similarity between 'u' attributes.
        """
        ds_similarities = []
        cs_similarities = []
        us_similarities = []

        for i in range(len(programs)):
            for j in range(i + 1, len(programs)):
                program_1 = programs[i]
                program_2 = programs[j]
                ds_similarities.append(
                    cosine_similarity(program_1["d"], program_2["d"])
                )
                cs_similarities.append(
                    average_cosine_similarity(program_1["c"], program_2["c"])
                )
                us_similarities.append(
                    cosine_similarity(program_1["u"], program_2["u"])
                )

        return (
            sum(ds_similarities) / len(ds_similarities),
            sum(cs_similarities) / len(cs_similarities),
            sum(us_similarities) / len(us_similarities),
        )

    def get_embeddings(graphs: dict, data: dict or list) -> tuple:
        """
        Get embeddings for the given graphs and data.

        Args:
            graphs (dict): A dictionary containing the graphs.
            data (dict or list): The data to be processed.

        Returns:
            tuple: A tuple containing two dictionaries. The first dictionary contains the embeddings
            for the trained model, and the second dictionary contains the embeddings for the untrained model.
        """
        funcgnn_param_parser = parameter_parser()
        funcgnn_trainer = funcGNNTrainer(funcgnn_param_parser)

        model_to_probe_trained = funcgnn_trainer.model

        model_to_probe_trained.load_state_dict(
            torch.load(
                os.path.join(
                    os.getcwd(), "src", args.model, "models", "model_state.pth"
                )
            )
        )

        embeddings_trained_all = {}
        for k, v in tqdm(
            data.items() if type(data) == dict else enumerate(data),
            total=len(data),
            desc="Validating probe trained/untrained",
        ):

            input_data = {
                    "graph_1": graphs[v[0]]["graph"],
                    "graph_2": graphs[v[1]]["graph"],
                    "labels_1": graphs[v[0]]["labels"],
                    "labels_2": graphs[v[1]]["labels"],
                    "ged": 0,
                }
                # embeddings_untrained = model_to_probe_untrained.encode(input_data)
            feature_gradients = (
                    model_to_probe_trained.get_gradient_from_output_to_input(input_data)
                )

        return feature_gradients

    def compare_embeddings(data: list or dict, embeddings: dict) -> float:
        """
        Compare the embeddings of data pairs and calculate the average cosine similarity.

        Args:
            data (list or dict): The data pairs to compare. If it's a list, each element should be a pair of indices.
                                 If it's a dictionary, the values should be lists of indices.
            embeddings (dict): A dictionary of embeddings, where the keys are the indices and the values are the embeddings.

        Returns:
            float: The average cosine similarity of the embeddings.

        """
        similarities = []

        # for similars, the
        if type(data) == list:
            try:
                for v in data:
                    embedding_x = embeddings[v[0]].reshape(-1)
                    embedding_y = embeddings[v[1]].reshape(-1)

                    # the embeddings are padded with zeros to make them the same length
                    if len(embedding_x) < len(embedding_y):
                        embedding_x = torch.cat(
                            (
                                embedding_x,
                                torch.zeros(len(embedding_y) - len(embedding_x)),
                            )
                        )
                    elif len(embedding_y) < len(embedding_x):
                        embedding_y = torch.cat(
                            (
                                embedding_y,
                                torch.zeros(len(embedding_x) - len(embedding_y)),
                            )
                        )
                    embedding_x = torch.nn.functional.normalize(embedding_x, dim=0)
                    embedding_y = torch.nn.functional.normalize(embedding_y, dim=0)
                    # Calculate the cosine similarity
                    cosine_similarity = (
                        torch.dot(embedding_x, embedding_y)
                        / (
                            (
                                torch.linalg.norm(embedding_x)
                                * torch.linalg.norm(embedding_y)
                            )
                        )
                    ).item()

                    similarities.append(cosine_similarity)
            except:
                pass

        # for dissimilars, we have multiple dissimilars, comparison is required between each pair
        elif type(data) == dict:
            for _, similars in data.items():
                for i in range(len(similars)):
                    for j in range(i + 1, len(similars)):
                        try:
                            embedding_x = embeddings[similars[i]].reshape(-1)
                            embedding_y = embeddings[similars[j]].reshape(-1)

                            if len(embedding_x) < len(embedding_y):
                                embedding_x = torch.cat(
                                    (
                                        embedding_x,
                                        torch.zeros(
                                            len(embedding_y) - len(embedding_x)
                                        ),
                                    )
                                )
                            elif len(embedding_y) < len(embedding_x):
                                embedding_y = torch.cat(
                                    (
                                        embedding_y,
                                        torch.zeros(
                                            len(embedding_x) - len(embedding_y)
                                        ),
                                    )
                                )

                            embedding_x = torch.nn.functional.normalize(
                                embedding_x, dim=0
                            )
                            embedding_y = torch.nn.functional.normalize(
                                embedding_y, dim=0
                            )
                            # Calculate the cosine similarity
                            cosine_similarity = (
                                torch.dot(embedding_x, embedding_y)
                                / (
                                    (
                                        torch.linalg.norm(embedding_x)
                                        * torch.linalg.norm(embedding_y)
                                    )
                                )
                            ).item()

                            similarities.append(cosine_similarity)
                        except:
                            pass

        return sum(similarities) / len(similarities)

    def get_graphs(data_files: dict) -> dict:
        """
        Retrieves graphs from the given data files.

        Args:
            data_files (dict): A dictionary containing the paths to the data files.

        Returns:
            dict: A dictionary containing the retrieved graphs, with file names as keys and graph information as values.
        """
        graphs = {}

        for file in data_files["train"]:
            file_names = (
                file.replace(f"{args.dataset_path}/train/", "")
                .replace(".json", "")
                .split("::::")
            )
            data = json.load(open(file))
            if file_names[0] not in graphs.keys():
                graphs[file_names[0]] = {
                    "labels": data["labels_1"],
                    "graph": data["graph_1"],
                }
            if file_names[1] not in graphs.keys():
                graphs[file_names[1]] = {
                    "labels": data["labels_2"],
                    "graph": data["graph_2"],
                }
        for file in data_files["test"]:
            file_names = (
                file.replace(f"{args.dataset_path}/test/", "")
                .replace(".json", "")
                .split("::::")
            )
            data = json.load(open(file))
            if file_names[0] not in graphs.keys():
                graphs[file_names[0]] = {
                    "labels": data["labels_1"],
                    "graph": data["graph_1"],
                }
            if file_names[1] not in graphs.keys():
                graphs[file_names[1]] = {
                    "labels": data["labels_2"],
                    "graph": data["graph_2"],
                }

        return graphs

    def get_similar_graphs(data_files: dict) -> dict:
        """
        Get similar graphs based on the given data files.

        Args:
            data_files (dict): A dictionary containing the data files.

        Returns:
            dict: A dictionary containing the similar graphs.
        """
        similars = {}
        file_names = [
            (
                file_name.replace(f"{args.dataset_path}/train/", "")
                .replace(".json", "")
                .split("::::")
            )
            for file_name in data_files["train"]
        ]
        u_file_names = set([f_n for f_N in file_names for f_n in f_N])

        for file_name in u_file_names:
            file_name_prefix = file_name.split("_")[0]
            if file_name_prefix not in similars.keys():
                similars[file_name_prefix] = []
            else:
                similars[file_name_prefix].append(file_name)

        return similars

    def get_dissimilar_graphs(data_files: dict) -> list:
        """
        Retrieves the dissimilar graphs from the given data files.

        Args:
            data_files (dict): A dictionary containing the data files.

        Returns:
            list: A list of dissimilar graph file names.
        """
        dissimlars = []
        for file in data_files["train"]:
            file_names = (
                file.replace(f"{args.dataset_path}/train/", "")
                .replace(".json", "")
                .split("::::")
            )
            dissimlars.append(file_names)
        return dissimlars

    # the data_files come from the original dataset of FungGNN
    data_files = {
        "train": [
            os.path.join(args.dataset_path, "train", i)
            for i in os.listdir(os.path.join(args.dataset_path, "train"))
            if i.endswith(".json")
        ],
        "test": [
            os.path.join(args.dataset_path, "test", i)
            for i in os.listdir(os.path.join(args.dataset_path, "test"))
        ],
    }
    graphs = get_graphs(data_files)

    similars = get_similar_graphs(data_files)
    dissimilars = get_dissimilar_graphs(data_files)

    embeddings_trained_similar, embeddings_untrained_similar = get_embeddings(
        graphs, dissimilars
    )
    emebeddings_trained_similar_similarity = compare_embeddings(
        similars, embeddings_trained_similar
    )
    emebeddings_untrained_similar_similarity = compare_embeddings(
        similars, embeddings_untrained_similar
    )
    print(
        "Cosine similarity for similar embeddings, trained: ",
        emebeddings_trained_similar_similarity,
    )
    print(
        "Cosine similarity for similar embeddings, untrained: ",
        emebeddings_untrained_similar_similarity,
    )

    embeddings_trained_dissimilar, embeddings_untrained_dissimilar = get_embeddings(
        graphs, dissimilars
    )
    embeddings_trained_dissimilar_similarity = compare_embeddings(
        dissimilars, embeddings_trained_dissimilar
    )
    embeddings_untrained_dissimilar_similarity = compare_embeddings(
        dissimilars, embeddings_untrained_dissimilar
    )
    print(
        "Cosine similarity for dissimilar embeddings, trained: ",
        embeddings_trained_dissimilar_similarity,
    )
    print(
        "Cosine similarity for dissimilar embeddings, untrained: ",
        embeddings_untrained_dissimilar_similarity,
    )

    dcu_similar = []
    for k, v in similars.items():
        dcu_similar.append(
            calculate_similarity([code_to_index_single(graphs[i]) for i in v])
        )

    dcu_similar = np.array(dcu_similar)
    print("Average of Ds - similars: ", dcu_similar[:, 0].mean())
    print("Average of Cs - similars: ", dcu_similar[:, 1].mean())
    print("Average of Us - similars: ", dcu_similar[:, 2].mean())

    dcu_disimilar = []
    for i in dissimilars:
        dcu_disimilar.append(
            calculate_similarity(
                [code_to_index_single(graphs[i[0]]), code_to_index_single(graphs[i[1]])]
            )
        )
    dcu_disimilar = np.array(dcu_disimilar)
    print("Average of Ds - dissimilars: ", dcu_disimilar[:, 0].mean())
    print("Average of Cs - dissimilars: ", dcu_disimilar[:, 1].mean())
    print("Average of Us - dissimilars: ", dcu_disimilar[:, 2].mean())

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
    default="ast_nn",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    help="Path to the dataset - Path follows the format /model_name/dataset",
    default=os.path.join(os.getcwd(), "src", "ast_nn", "dataset"),
)
parser.add_argument(
    "--language",
    type=str,
    help="Language of the dataset. Only for AST-NN",
    choices=["java", "c"],
    default="c",
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
    default=200,
    choices=[200, 64, 512],
    help="Hidden dimension of the probe. 200 for AST-NN. 64 for FuncGnn, 512 for SumTF, 512 for CodeSumDRL",
)
args = parser.parse_args()
#### Arguemnt Parser ####
device = args.device

if args.model == "ast_nn":
    from gensim.models.word2vec import Word2Vec

    from ast_nn.src.code_to_repr import code_to_index
    from ast_nn.src.data_pipeline import process_input
    from ast_nn.src.model import BatchProgramCC
    from ast_probe.probe import (
        ParserLoss,
        ParserProbe,
        collator_fn_astnn,
        get_embeddings_astnn,
    )

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

    def get_initial_data():
        clone_ids = pickle.load(
            open(
                os.path.join(args.dataset_path, args.language, "clone_ids.pkl"),
                "rb",
            )
        )
        programs = (
            pickle.load(
                open(
                    os.path.join(args.dataset_path, args.language, "programs.pkl"),
                    "rb",
                )
            )
            if args.language == "c"
            else pd.read_csv(
                os.path.join(args.dataset_path, args.language, "programs.tsv"),
                delimiter="\t",
            )
        )
        programs.columns = (
            ["id", "code", "label"] if args.language == "c" else ["id", "code"]
        )
        if args.language == "c":
            programs.drop(columns=["label"], inplace=True)
        clone_ids["id1"] = clone_ids["id1"].astype(int)
        clone_ids["id2"] = clone_ids["id2"].astype(int)

        merged_data = pd.merge(
            clone_ids, programs, how="left", left_on="id1", right_on="id"
        )
        merged_data = pd.merge(
            merged_data, programs, how="left", left_on="id2", right_on="id"
        )

        merged_data.drop(["id_x", "id_y"], axis=1, inplace=True)
        merged_data.dropna(inplace=True)
        merged_data.reset_index(drop=True, inplace=True)
        merged_data_similar = merged_data[merged_data["label"] == 1]
        merged_data_dissimilar = merged_data[merged_data["label"] != 1]

        return merged_data_similar, merged_data_dissimilar

    # comapre the similarity of trained and untrained embeddings for siimilar and dissimilar pairs
    def get_attention_map(model, data):
        # input_shapes = set(i.shape for i in embeddings)

        similarities = []
        for row_index, row in tqdm(data.iterrows(), total=len(data)):
            try:
                code_x = row["code_x"]
                code_y = row["code_y"]
                hook1 = code_x.register_hook(hook_fn)
                hook2 = code_y.register_hook(hook_fn)
                code_x_out, code_y_out = model.forward_att([code_x], [code_y])

                grad_output = torch.ones_like(code_x_out)

                code_x_out.backward(grad_output)
                code_y_out.backward(grad_output)

                # Unregister hooks to avoid memory leaks
                hook1.remove()
                hook2.remove()

                # Calculate attention scores (magnitude of gradients)
                attention_scores_x1 = torch.abs(gradients_x1)
                attention_scores_x2 = torch.abs(gradients_x2)
                if code_x_out is None or code_y_out is None:
                    print("hi")
            except Exception as e:
                print(e)
                continue

        return sum(similarities) / len(similarities)

    def hook_fn(grad):
        global gradients_x1, gradients_x2
        gradients_x1 = grad[0]
        gradients_x2 = grad[1]

    word2vec = Word2Vec.load(
        os.path.join(
            os.path.join(os.getcwd(), "src", "ast_nn", "dataset"),
            args.language,
            "embeddings",
            "node_w2v_128",
        )
    ).wv
    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]

    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[: word2vec.vectors.shape[0]] = word2vec.vectors

    model = BatchProgramCC(
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=100,
        vocab_size=MAX_TOKENS + 1,
        encode_dim=128,
        label_size=1,
        batch_size=1,
        use_gpu=False,
        pretrained_weight=embeddings,
        word2vec_path=os.path.join(
            os.path.join(
                os.path.join(os.getcwd(), "src", "ast_nn", "dataset"),
                args.language,
                "embeddings",
                "node_w2v_128",
            )
        ),
        language=args.language,
    )

    model.load_state_dict(
        torch.load(
            os.path.join(
                os.getcwd(),
                "src",
                args.model,
                "models",
                "astnn_C_1.pkl" if args.language == "c" else "astnn_JAVA_5.pkl",
            )
        )
    )
    # get DCU similarity for the similar and dissimilar pairs
    merged_data_similar, merged_data_dissimilar = get_initial_data()

    gradients_x1 = None
    gradients_x2 = None

    get_attention_map(model, merged_data_similar)

import argparse
import logging
import logging.config
import os
import pickle
import warnings
import json
import numpy as np
import pandas as pd
import pickle5 as pickle
import torch
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

    def pad_list(short_list, target_length):
        return short_list + [-1] * (target_length - len(short_list))

    def cosine_similarity(list1, list2):
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

    def average_cosine_similarity(C1, C2):
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

    def calculate_similarity(row):
        tests1 = code_to_index(row["code_x"], args.language)
        tests2 = code_to_index(row["code_y"], args.language)
        return (
            cosine_similarity(tests1["d"], tests2["d"]),
            cosine_similarity(tests1["u"], tests2["u"]),
            average_cosine_similarity(tests1["c"], tests2["c"]),
        )

    def get_similarity_from_asts(merged_data):
        merged_data_ds = []
        merged_data_us = []
        merged_data_cs = []
        for index, row in tqdm(
            merged_data.iterrows(),
            total=len(merged_data),
        ):
            try:
                similarity_ds, similarity_us, similarity_cs = calculate_similarity(row)
                merged_data_ds.append(similarity_ds)
                merged_data_us.append(similarity_us)
                merged_data_cs.append(similarity_cs)
            except:
                merged_data_ds.append(0)
                merged_data_us.append(0)
                merged_data_cs.append(0)

        return merged_data_ds, merged_data_us, merged_data_cs

    def get_embeddings(merged_data):
        probe_model = ParserProbe(
            probe_rank=128,
            hidden_dim=200,
            number_labels_d=264,
            number_labels_c=445,
            number_labels_u=264,
        )

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

        model_to_probe_untrained = BatchProgramCC(
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=100,
            vocab_size=MAX_TOKENS + 1,
            encode_dim=128,
            label_size=1,
            batch_size=2,
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

        model_to_probe_trained = BatchProgramCC(
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=100,
            vocab_size=MAX_TOKENS + 1,
            encode_dim=128,
            label_size=1,
            batch_size=2,
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

        model_to_probe_trained.load_state_dict(
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

        embeddings_trained_all, embeddings_untrained_all = [], []
        for index, row in tqdm(
            merged_data.iterrows(),
            total=len(merged_data),
            desc="Validating probe trained/untrained",
        ):
            try:
                embeddings_untrained = get_embeddings_astnn(
                    [row["code_x"], row["code_y"]], model_to_probe_untrained
                )
                embeddings_trained = get_embeddings_astnn(
                    [row["code_x"], row["code_y"]], model_to_probe_trained
                )
                embeddings_trained_all.append(embeddings_trained)
                embeddings_untrained_all.append(embeddings_untrained)
            except:
                embeddings_trained_all.append(None)
                embeddings_untrained_all.append(None)

        return embeddings_trained_all, embeddings_untrained_all

    def get_dcu_similarity(data: pd.DataFrame):
        (
            merged_data_ds,
            merged_data_us,
            merged_data_cs,
        ) = get_similarity_from_asts(data)

        print(f"Average of Ds: {sum(merged_data_ds)/len(merged_data_ds)}")
        print(f"Average of Cs: {sum(merged_data_cs)/len(merged_data_cs)}")
        print(f"Average of Us: {sum(merged_data_us)/len(merged_data_us)}")

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

    (
        embeddings_trained_all,
        embeddings_untrained_all,
    ) = get_embeddings(merged_data_similar)

    pickle.dump(
        embeddings_trained_all,
        open(
            os.path.join(
                os.getcwd(),
                "src",
                args.model,
                "models",
                f"astnn_embeddings_trained_{args.language}_similar.pkl",
            ),
            "wb",
        ),
    )
    pickle.dump(
        embeddings_untrained_all,
        open(
            os.path.join(
                os.getcwd(),
                "src",
                args.model,
                "models",
                f"astnn_embeddings_untrained_{args.language}_similar.pkl",
            ),
            "wb",
        ),
    )

    (
        embeddings_trained_all,
        embeddings_untrained_all,
    ) = get_embeddings(merged_data_dissimilar)

    pickle.dump(
        embeddings_trained_all,
        open(
            os.path.join(
                os.getcwd(),
                "src",
                args.model,
                "models",
                f"astnn_embeddings_trained_{args.language}_dissimilar.pkl",
            ),
            "wb",
        ),
    )
    pickle.dump(
        embeddings_untrained_all,
        open(
            os.path.join(
                os.getcwd(),
                "src",
                args.model,
                "models",
                f"astnn_embeddings_untrained_{args.language}_dissimilar.pkl",
            ),
            "wb",
        ),
    )

    print("Getting D,C,U similarity for similar")
    get_dcu_similarity(merged_data_similar)

    print("Getting D,C,U similarity for dissimilar")
    get_dcu_similarity(merged_data_dissimilar)
    
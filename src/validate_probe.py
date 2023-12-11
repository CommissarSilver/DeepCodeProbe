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
import torch.nn.functional as F
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
    default="code_sum_drl",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    help="Path to the dataset - Path follows the format /model_name/dataset",
    default=os.path.join(os.getcwd(), "src", "summarization_tf", "dataset"),
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

    def calculate_similarity(row: pd.Series) -> tuple:
        """
        1 - Code is converted to AST-NN's required representation.
        2 - The cosine similarity between the D, C, U representations of the two codes is calculated.
        3 - The C tuple is a list of tensors therefore a different function is used to calculate the average cosine similarity.
        """
        tests1 = code_to_index(row["code_x"], args.language)
        tests2 = code_to_index(row["code_y"], args.language)
        return (
            cosine_similarity(tests1["d"], tests2["d"]),
            cosine_similarity(tests1["u"], tests2["u"]),
            average_cosine_similarity(tests1["c"], tests2["c"]),
        )

    def get_similarity_from_asts(merged_data: pd.DataFrame) -> tuple:
        """
        Calculate the D,C,U similarity for the given dataset.
        """
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

    def get_dcu_similarity(data: pd.DataFrame):
        (
            merged_data_ds,
            merged_data_us,
            merged_data_cs,
        ) = get_similarity_from_asts(data)

        print(f"Average of Ds: {sum(merged_data_ds)/len(merged_data_ds)}")
        print(f"Average of Cs: {sum(merged_data_cs)/len(merged_data_cs)}")
        print(f"Average of Us: {sum(merged_data_us)/len(merged_data_us)}")

    # function for getting embeddings of the trained/untrained model
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

    # function for kick-starting the initial phase of getting embeddings and D,C,U similarity
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

    # comapre the similarity of trained and untrained embeddings for siimilar and dissimilar pairs
    def compare_embeddings(model: str = "trained", mode: str = "similar"):
        embeddings = pickle.load(
            open(
                os.path.join(
                    os.getcwd(),
                    "src",
                    args.model,
                    "models",
                    f"astnn_embeddings_{model}_{args.language}_{mode}.pkl",
                ),
                "rb",
            )
        )
        # input_shapes = set(i.shape for i in embeddings)

        similarities = []
        for row in tqdm(embeddings, total=len(embeddings)):
            try:
                embedding_x = row[0].reshape(-1)
                embedding_y = row[1].reshape(-1)

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

        return sum(similarities) / len(similarities)

    # get DCU similarity for the similar and dissimilar pairs
    get_initial_data()
    # get emebdding similarity for the similar and dissimilar pairs
    cosine_sim_similar_trained = compare_embeddings(
        model="untrained",
        mode="similar",
    )
    consine_sim_dissimilar_trained = compare_embeddings(
        model="untrained",
        mode="dissimilar",
    )

    print(
        f"Average cosine similarity for similar trained: {cosine_sim_similar_trained}"
    )
    print(
        f"Average cosine similarity for dissimilar trained: {consine_sim_dissimilar_trained}"
    )

elif args.model == "funcgnn":
    import json

    import pandas as pd

    from ast_probe.probe import (
        FuncGNNParserProbe,
        ParserLossFuncGNN,
        collator_fn_funcgnn,
        get_embeddings_funcgnn,
    )
    from funcgnn.src.code_to_repr import code_to_index, code_to_index_single
    from funcgnn.src.funcgnn import funcGNNTrainer
    from funcgnn.src.param_parser import parameter_parser

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

    def calculate_similarity(programs):
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

    def get_embeddings(graphs, data):
        funcgnn_param_parser = parameter_parser()
        funcgnn_trainer_1 = funcGNNTrainer(funcgnn_param_parser)
        funcgnn_trainer_2 = funcGNNTrainer(funcgnn_param_parser)
        model_to_probe_trained = funcgnn_trainer_1.model
        model_to_probe_untrained = funcgnn_trainer_2.model

        model_to_probe_trained.load_state_dict(
            torch.load(
                os.path.join(
                    os.getcwd(), "src", args.model, "models", "model_state.pth"
                )
            )
        )

        embeddings_trained_all, embeddings_untrained_all = {}, {}
        for k, v in tqdm(
            data.items() if type(data) == dict else enumerate(data),
            total=len(data),
            desc="Validating probe trained/untrained",
        ):
            for value in v:
                try:
                    input_data = {
                        "graph_1": graphs[value]["graph"],
                        "graph_2": graphs[value]["graph"],
                        "labels_1": graphs[value]["labels"],
                        "labels_2": graphs[value]["labels"],
                        "ged": 0,
                    }
                    embeddings_untrained = model_to_probe_untrained.encode(input_data)
                    embeddings_trained = model_to_probe_trained.encode(input_data)

                    embeddings_trained_all[value] = embeddings_trained
                    embeddings_untrained_all[value] = embeddings_untrained
                except:
                    embeddings_trained_all[value] = None
                    embeddings_untrained_all[value] = None

        return embeddings_trained_all, embeddings_untrained_all

    def compare_embeddings(data, embeddings):
        # if type(data)
        similarities = []

        if type(data) == list:
            try:
                for v in data:
                    embedding_x = embeddings[v[0]].reshape(-1)
                    embedding_y = embeddings[v[1]].reshape(-1)

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

    def get_graphs(data_files: dict):
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

    def get_similar_graphs(data_files: dict):
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

        # similar_graphs = {}
        # for k, v in similars.items():
        #     similar_graphs[k] = [graphs[i] for i in v]

        return similars

    def get_dissimilar_graphs(data_files: dict):
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
        graphs, similars
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

elif args.model == "summarization_tf":
    import pandas as pd
    from torch.utils.data import Dataset

    from ast_probe.probe import (
        ParserLossSumTF,
        SumTFParserProbe,
        collator_fn_sum_tf,
        get_embeddings_sum_tf,
    )
    from summarization_tf.src.code_to_repr import (
        code_to_ast,
        ast_to_index,
        code_to_index,
    )
    from summarization_tf.src.models import MultiwayModel
    from summarization_tf.src.utils import read_pickle

    # functions for calculating D,C,U similarity
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
        tests1 = ast_to_index(*code_to_ast(row["code_x"]))
        tests2 = ast_to_index(*code_to_ast(row["code_y"]))
        return (
            average_cosine_similarity(tests1[0], tests2[0]),
            cosine_similarity(tests1[1], tests2[1]),
            cosine_similarity(tests1[2], tests2[2]),
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

    # we use the java dataset from AST-NN
    def get_initial_data_from_astnn():
        """
        We're using the dataset from AST-NN since Summarization-TF is designed to work with JAVA files.
        We also need to compare the D,C,U similarity of what we have designed.
        Since that dataset has codes that are similar and dissimilar, we simply use that.
        """
        clone_ids = pickle.load(
            open(
                os.path.join(
                    os.getcwd(), "src", "ast_nn", "dataset", "java", "clone_ids.pkl"
                ),
                "rb",
            )
        )
        programs = pd.read_csv(
            os.path.join(
                os.getcwd(), "src", "ast_nn", "dataset", "java", "programs.tsv"
            ),
            delimiter="\t",
        )

        programs.columns = ["id", "code"]
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

    def get_model_processed_data(merged_data, index_map):
        pairs = []

        def find_key_by_value(json_data, target_value):
            for key, value in json_data.items():
                if target_value in value:
                    return key

        for index, row in tqdm(merged_data.iterrows(), total=len(merged_data)):
            code_x = row["code_x"]
            code_y = row["code_y"]
            index_x = find_key_by_value(index_map, row["id1"])
            index_y = find_key_by_value(index_map, row["id2"])
            if "train" in index_x and index_y:
                repr_x = code_to_index(
                    code_x, "useless_drivel", int(index_x.split("_")[1])
                )
                repr_y = code_to_index(
                    code_y, "useless_drivel", int(index_y.split("_")[1])
                )
                if repr_x and repr_y is not None:
                    pairs.append((repr_x, repr_y))
            elif "valid" in index_x and index_y:
                repr_x = code_to_index(
                    code_x, "useless_drivel", int(index_x.split("_")[1]), "valid"
                )
                repr_y = code_to_index(
                    code_y, "useless_drivel", int(index_y.split("_")[1]), "valid"
                )
                if repr_x and repr_y is not None:
                    pairs.append((repr_x, repr_y))
            elif "test" in index_x and index_y:
                repr_x = code_to_index(
                    code_x, "useless_drivel", int(index_x.split("_")[1]), "test"
                )
                repr_y = code_to_index(
                    code_y, "useless_drivel", int(index_y.split("_")[1]), "test"
                )
                if repr_x and repr_y is not None:
                    pairs.append((repr_x, repr_y))

        return pairs

    def get_embeddings(pairs):
        code_w2i = read_pickle(
            f"/store/travail/vamaj/Leto/src/summarization_tf/dataset_original/code_w2i.pkl"
        )
        nl_w2i = read_pickle(
            f"/store/travail/vamaj/Leto/src/summarization_tf/dataset_original/nl_w2i.pkl"
        )

        model_to_probe_untrained = MultiwayModel(
            512,
            512,
            512,
            len(code_w2i),
            len(nl_w2i),
            dropout=0.5,
            lr=0.001,
            layer=1,
        )

        model_to_probe_trained = MultiwayModel(
            512,
            512,
            512,
            len(code_w2i),
            len(nl_w2i),
            dropout=0.5,
            lr=0.001,
            layer=1,
        )

        model_to_probe_trained.load_state_dict(
            torch.load(
                os.path.join(
                    os.getcwd(),
                    "src",
                    args.model,
                    "models",
                    "model_state.pth",
                )
            )
        )

        embeddings_trained_all, embeddings_untrained_all = [], []
        for index, row in tqdm(
            enumerate(pairs),
            total=len(pairs),
            desc="Validating probe trained/untrained",
        ):
            embeddings_trained_all.append(
                get_embeddings_sum_tf(
                    (row[0]["tree_tensor"], row[1]["tree_tensor"]),
                    model_to_probe_trained,
                )
            )
            embeddings_untrained_all.append(
                get_embeddings_sum_tf(
                    (row[0]["tree_tensor"], row[1]["tree_tensor"]),
                    model_to_probe_untrained,
                )
            )

        return embeddings_trained_all, embeddings_untrained_all

    def compare_embeddings(embeddings):
        similarities = []
        for row in tqdm(embeddings, total=len(embeddings)):
            try:
                embeddings_x, embeddings_y = torch.split(row, 1, dim=0)
                embedding_x = embeddings_x.reshape(-1)
                embedding_y = embeddings_y.reshape(-1)

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

        return sum(similarities) / len(similarities)

    index_map = json.load(
        open(
            os.path.join(
                os.getcwd(), "src", "summarization_tf", "sum_tf_index_map.json"
            ),
            "r",
        )
    )
    merged_data_similar, merged_data_dissimilar = get_initial_data_from_astnn()

    pairs_similar = get_model_processed_data(merged_data_similar, index_map)
    embeddings_similar_trained, embeddings_similar_untrained = get_embeddings(
        pairs_similar
    )
    embeddings_trained_cosine_sim = compare_embeddings(embeddings_similar_trained)
    embeddings_untrained_cosine_sim = compare_embeddings(embeddings_similar_untrained)
    print(
        f"Average cosine similarity for similar trained: {embeddings_trained_cosine_sim}"
    )
    print(
        f"Average cosine similarity for similar untrained: {embeddings_untrained_cosine_sim}"
    )

    pairs_dissimilar = get_model_processed_data(merged_data_dissimilar, index_map)
    embeddings_dissimilar_trained, embeddings_dissimilar_untrained = get_embeddings(
        pairs_dissimilar
    )
    embeddings_trained_cosine_disim = compare_embeddings(embeddings_dissimilar_trained)
    embeddings_untrained_cosine_disim = compare_embeddings(
        embeddings_dissimilar_untrained
    )
    print(
        f"Average cosine similarity for dissimilar trained: {embeddings_trained_cosine_disim}"
    )
    print(
        f"Average cosine similarity for dissimilar untrained: {embeddings_untrained_cosine_disim}"
    )

    ds, cs, us = get_similarity_from_asts(merged_data_similar)
    print(f"Average of Ds: {sum(ds)/len(ds)}")
    print(f"Averagen of Cs: {sum(cs)/len(cs)}")
    print(f"Average of Us: {sum(us)/len(us)}")

    ds, cs, us = get_similarity_from_asts(merged_data_dissimilar)
    print(f"Average of Ds - dissimilar: {sum(ds)/len(ds)}")
    print(f"Averagen of Cs - dissimilar: {sum(cs)/len(cs)}")
    print(f"Average of Us - dissimilar: {sum(us)/len(us)}")

elif args.model == "code_sum_drl":
    import sys

    import pandas as pd
    from torch.utils.data import Dataset
    from datasets import load_dataset
    from code_sum_drl.src.code_to_repr import Dataset as CodeSumDataset
    from code_sum_drl.src.code_to_repr import code_to_index

    sys.path.append("/store/travail/vamaj/Leto/src/code_sum_drl/src")
    import lib
    from lib.data.Tree import *

    from ast_probe.probe import (
        CodeSumDRLarserProbe,
        ParserLossCodeSumDRL,
        get_embeddings_code_sum_drl,
    )

    def get_opt():
        opt = argparse.ArgumentParser()

        opt.add_argument(
            "-rnn_size", type=int, default=512, help="Size of LSTM hidden states,"
        )
        opt.add_argument(
            "-word_vec_size",
            type=int,
            default=512,
            help="Word embedding sizes",
        )
        opt.add_argument(
            "-brnn",
            action="store_true",
            help="Use a bidirectional encoder",
        )
        opt.add_argument(
            "-gpus",
            default=[],
            nargs="+",
            type=int,
            help="Use CUDA on the listed devices.",
        )
        opt.add_argument(
            "-layers",
            type=int,
            default=1,
            help="Number of layers in the LSTM encoder/decoder",
        )
        opt.add_argument(
            "-dropout",
            type=float,
            default=0.3,
            help="Dropout probability; applied between LSTM stacks.",
        )
        opt.add_argument(
            "-input_feed",
            type=int,
            default=1,
            help="""Feed the context vector at each time step as
                            additional input (via concatenation with the word embeddings) to the decoder.""",
        )
        opt.add_argument(
            "-has_attn",
            type=int,
            default=1,
            help="""attn model or not""",
        )
        opt.add_argument(
            "-cuda",
            type=bool,
            default=False,
            help="""attn model or not""",
        )
        # opt.cuda = False
        opt = opt.parse_args()
        return opt

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
        tests1 = code_to_index(row["code_x"])
        tests2 = code_to_index(row["code_y"])
        if tests1["d"] == [] or tests1["d"] == []:
            return (0, 0, 0)
        return (
            cosine_similarity(tests1["d"], tests2["d"]),
            cosine_similarity(tests1["u"], tests2["u"]),
            cosine_similarity(tests1["c"], tests2["c"]),
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
                if not any(np.isnan([similarity_ds, similarity_us, similarity_cs])):
                    merged_data_ds.append(similarity_ds)
                    merged_data_us.append(similarity_us)
                    merged_data_cs.append(similarity_cs)
            except:
                merged_data_ds.append(0)
                merged_data_us.append(0)
                merged_data_cs.append(0)

        return merged_data_ds, merged_data_us, merged_data_cs

    def create_dataset_for_code_sum(merged_data, mode):
        data_dir = "/store/travail/vamaj/Leto/src/code_sum_drl/dataset_clones/"
        original_path = data_dir + "original/"
        processed_path = data_dir + "processed/"
        train_path = data_dir + "train/"
        for index, row in tqdm(merged_data.iterrows(), total=len(merged_data)):
            code_x = row["code_x"].replace("\n", " DCNL DCSP ")
            code_y = row["code_y"].replace("\n", " DCNL DCSP ")
            with open(processed_path + f"all_x_{mode}.code", "a") as code_x_file:
                with open(
                    processed_path + f"all_x_{mode}.comment", "a"
                ) as comment_x_file:
                    code_x_file.write(code_x + "\n")
                    comment_x_file.write("useless drivel needed" + "\n")
            with open(processed_path + f"all_y_{mode}.code", "a") as code_y_file:
                with open(
                    processed_path + f"all_y_{mode}.comment", "a"
                ) as comment_y_file:
                    code_y_file.write(code_y + "\n")
                    comment_y_file.write("useless drivel needed" + "\n")

    def get_similar_codesum_ds():
        dataset_similar_x = torch.load(
            "/store/travail/vamaj/Leto/src/code_sum_drl/dataset_clones/train/processed_all_x_similar.train.pt"
        )
        dataset_similar_y = torch.load(
            "/store/travail/vamaj/Leto/src/code_sum_drl/dataset_clones/train/processed_all_y_similar.train.pt"
        )
        indexes_x = {
            v: i for i, v in enumerate(dataset_similar_x["train_xe"]["indexes"])
        }
        indexes_y = {
            v: i for i, v in enumerate(dataset_similar_y["train_xe"]["indexes"])
        }
        shared_indexes_file_nums = set(indexes_x.keys()).intersection(
            set(indexes_y.keys())
        )
        shared_indexes_x = [indexes_x[v] for v in shared_indexes_file_nums]
        shared_indexes_y = [indexes_y[v] for v in shared_indexes_file_nums]
        # shared_indexes = sorted(list(shared_indexes), reverse=True)
        not_shared_indexes_x = [
            i
            for i in range(len(dataset_similar_x["train_xe"]["src"]))
            if i not in shared_indexes_x
        ]
        not_shared_indexes_y = [
            i
            for i in range(len(dataset_similar_y["train_xe"]["src"]))
            if i not in shared_indexes_y
        ]
        not_shared_indexes_x = sorted(not_shared_indexes_x, reverse=True)
        not_shared_indexes_y = sorted(not_shared_indexes_y, reverse=True)
        for index in not_shared_indexes_x:
            del dataset_similar_x["train_xe"]["src"][index]
            del dataset_similar_x["train_xe"]["tgt"][index]
            del dataset_similar_x["train_xe"]["trees"][index]
            del dataset_similar_x["train_xe"]["original_codes"][index]
            del dataset_similar_x["train_xe"]["original_comments"][index]
            del dataset_similar_x["train_pg"]["src"][index]
            del dataset_similar_x["train_pg"]["tgt"][index]
            del dataset_similar_x["train_pg"]["trees"][index]
            del dataset_similar_x["train_pg"]["original_codes"][index]
            del dataset_similar_x["train_pg"]["original_comments"][index]

        for index in not_shared_indexes_y:
            del dataset_similar_y["train_xe"]["src"][index]
            del dataset_similar_y["train_xe"]["tgt"][index]
            del dataset_similar_y["train_xe"]["trees"][index]
            del dataset_similar_y["train_xe"]["original_codes"][index]
            del dataset_similar_y["train_xe"]["original_comments"][index]
            del dataset_similar_y["train_pg"]["src"][index]
            del dataset_similar_y["train_pg"]["tgt"][index]
            del dataset_similar_y["train_pg"]["trees"][index]
            del dataset_similar_y["train_pg"]["original_codes"][index]
            del dataset_similar_y["train_pg"]["original_comments"][index]

        return dataset_similar_x, dataset_similar_y

    def pre_process_for_code_sum(dataset_x, dataset_y):
        def get_data_trees(trees):
            data_trees = []
            for t_json in trees:
                for k, node in t_json.items():
                    if node["parent"] == None:
                        root_idx = k
                tree = json2tree_binary(t_json, Tree(), root_idx)
                data_trees.append(tree)

            return data_trees

        def get_data_leafs(trees, srcDicts):
            leafs = []
            for tree in trees:
                leaf_contents = tree.leaf_contents()

                leafs.append(srcDicts.convertToIdx(leaf_contents, Constants.UNK_WORD))
            return leafs

        dicts_x = dataset_x["dicts"]
        dicts_y = dataset_y["dicts"]

        dataset_x["train_xe"]["trees"] = get_data_trees(dataset_x["train_xe"]["trees"])
        dataset_y["train_xe"]["trees"] = get_data_trees(dataset_y["train_xe"]["trees"])

        # dataset_x["train_pg"]["trees"] = get_data_trees(dataset_x["train_pg"]["trees"])
        # dataset_y["train_pg"]["trees"] = get_data_trees(dataset_y["train_pg"]["trees"])

        # dataset_x["valid"]["trees"] = get_data_trees(dataset_x["valid"]["trees"])
        # dataset_y["valid"]["trees"] = get_data_trees(dataset_y["valid"]["trees"])

        # dataset_x["test"]["trees"] = get_data_trees(dataset_x["test"]["trees"])
        # dataset_y["test"]["trees"] = get_data_trees(dataset_y["test"]["trees"])

        dataset_x["train_xe"]["leafs"] = get_data_leafs(
            dataset_x["train_xe"]["trees"],
            dicts_x["src"],
        )
        dataset_y["train_xe"]["leafs"] = get_data_leafs(
            dataset_y["train_xe"]["trees"],
            dicts_y["src"],
        )

        # dataset_x["train_pg"]["leafs"] = get_data_leafs(
        #     dataset_x["train_pg"]["trees"],
        #     dicts_x["src"],
        # )
        # dataset_y["train_pg"]["leafs"] = get_data_leafs(
        #     dataset_y["train_pg"]["trees"],
        #     dicts_y["src"],
        # )

        # dataset_x["valid"]["leafs"] = get_data_leafs(
        #     dataset_x["valid"]["trees"],
        #     dicts_x["src"],
        # )
        # dataset_y["valid"]["leafs"] = get_data_leafs(
        #     dataset_y["valid"]["trees"],
        #     dicts_y["src"],
        # )

        # dataset_x["test"]["leafs"] = get_data_leafs(
        #     dataset_x["test"]["trees"],
        #     dicts_x["src"],
        # )
        # dataset_y["test"]["leafs"] = get_data_leafs(
        #     dataset_y["test"]["trees"],
        #     dicts_y["src"],
        # )

        train_set_x = CodeSumDataset(
            dataset_x["train_xe"],
            1,
            False,
            eval=False,
        )
        train_set_y = CodeSumDataset(
            dataset_y["train_xe"],
            1,
            False,
            eval=False,
        )

        return train_set_x, train_set_y

    def get_model_embeddings(
        opt, x_dicts, y_dicts, train_set_x, train_set_y, mode="trained"
    ):
        embs_x_all, embs_y_all = [], []

        code_encoder = lib.TreeEncoder(opt, x_dicts["src"])
        text_encoder = lib.Encoder(opt, x_dicts["src"])
        decoder = lib.HybridDecoder(opt, x_dicts["tgt"])
        generator = lib.BaseGenerator(
            torch.nn.Linear(opt.rnn_size, x_dicts["tgt"].size()), opt
        )
        model = lib.Hybrid2SeqModel(
            code_encoder,
            text_encoder,
            decoder,
            generator,
            opt,
        )

        model = lib.Hybrid2SeqModel(
            code_encoder,
            text_encoder,
            decoder,
            generator,
            opt,
        )
        if mode == "trained":
            checkpoint = torch.load(
                os.path.join(
                    os.getcwd(),
                    "src",
                    args.model,
                    "models",
                    "model_state.pt",
                ),
                map_location=torch.device("cpu"),
            )
            model = checkpoint["model"]

        model.opt.cuda = False
        model.opt.cuda = False

        try:
            for batches in tqdm(zip(train_set_x, train_set_y), total=len(train_set_x)):
                try:
                    batch_x = batches[0]
                    batch_y = batches[1]

                    embds_x = get_embeddings_code_sum_drl(
                        (
                            batch_x[0],
                            batch_x[1],
                            batch_x[2],
                            batch_x[3],
                        ),
                        model,
                    )
                    embs_y = get_embeddings_code_sum_drl(
                        (
                            batch_y[0],
                            batch_y[1],
                            batch_y[2],
                            batch_y[3],
                        ),
                        model,
                    )
                    embs_x_all.append(embds_x)
                    embs_y_all.append(embs_y)
                except:
                    pass
        except AssertionError:
            pass

        return embs_x_all, embs_y_all

    def compare_embeddings(embeddings_x, embeddings_y):
        similarities = []
        for embedding_x, embedding_y in tqdm(
            zip(embeddings_x, embeddings_y), total=len(embeddings_x)
        ):
            try:
                embedding_x = embedding_x.reshape(-1)
                embedding_y = embedding_y.reshape(-1)

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

        return sum(similarities) / len(similarities)

    opt = get_opt()

    #! load the clone dataset and save it as a csv. not needed a second time
    # dataset = load_dataset("PoolC/1-fold-clone-detection-600k-5fold")
    # dataset['train'].to_csv("/store/travail/vamaj/Leto/src/code_sum_drl/dataset/clones.csv", index=False)

    # dataset = pd.read_csv(
    #     "/store/travail/vamaj/Leto/src/code_sum_drl/dataset/clones.csv"
    # )
    # dataset.drop(
    #     columns=["pair_id", "question_pair_id", "code1_group", "code2_group"],
    #     inplace=True,
    # )
    # dataset.columns = ["code_x", "code_y", "label"]
    # merged_data_similar = dataset[dataset["label"] == 1]
    # merged_data_dissimilar = dataset[dataset["label"] == 0]

    # create_dataset_for_code_sum(merged_data_similar, "similar")
    # create_dataset_for_code_sum(merged_data_dissimilar, "dissimilar")
    # ds_sim, us_sim, cs_sim = get_similarity_from_asts(
    #     merged_data_similar.sample(n=10000)
    # )
    # print(f"Average of Ds - similar: {sum(ds_sim)/len(ds_sim)}")
    # print(f"Averagen of Cs - similar: {sum(cs_sim)/len(cs_sim)}")
    # print(f"Average of Us - similar: {sum(us_sim)/len(us_sim)}")

    # ds_sim, us_sim, cs_sim = get_similarity_from_asts(
    #     merged_data_dissimilar.sample(n=10000)
    # )
    # print(f"Average of Ds - dissimilar: {sum(ds_sim)/len(ds_sim)}")
    # print(f"Averagen of Cs - dissimilar: {sum(cs_sim)/len(cs_sim)}")
    # print(f"Average of Us - dissimilar: {sum(us_sim)/len(us_sim)}")

    dataset_similar_x, dataset_similar_y = get_similar_codesum_ds()
    x_dicts, y_dicts = dataset_similar_x["dicts"], dataset_similar_y["dicts"]
    train_set_x, train_set_y = pre_process_for_code_sum(
        dataset_similar_x, dataset_similar_y
    )
    # for i in range(len(train_set_x)):
    #     batch = train_set_x[i]
    #     ds = [dcu["d"] for dcu in batch[6]]
    #     cs = [dcu["c"] for dcu in batch[6]]
    #     us = [dcu["u"] for dcu in batch[6]]

    #     max_d = max(max_d, max([len(d) for d in ds]))
    #     max_c = max(max_c, max([len(c) for c in cs]))
    #     max_u = max(max_u, max([len(u) for u in us]))

    embs_x_similar_trained, embs_y_similar_trained = get_model_embeddings(
        opt, x_dicts, y_dicts, train_set_x, train_set_y, "trained"
    )
    compare_embeddings(embs_x_similar_trained, embs_y_similar_trained)
    embs_x_similar_untrained, embs_y_similar_untrained = get_model_embeddings(
        opt, x_dicts, y_dicts, train_set_x, train_set_y, "untrained"
    )
    print("hi")

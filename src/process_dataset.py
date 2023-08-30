import os
import pickle
import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    type=str,
    help="Model to probe",
    choices=["ast_nn", "funcgnn", "code_sum_drl", "sum_tf", "cscg_dual"],
    default="ast_nn",
)
parser.add_argument(
    "--language",
    type=str,
    help="Language of the dataset",
    choices=["java", "c"],
    default="c",
)
args = parser.parse_args()

if args.model == "ast_nn":
    if args.language == "c":
        dataset = pd.read_pickle(
            os.path.join(
                os.getcwd(),
                "src",
                args.model,
                "dataset",
                args.language,
                "programs.pkl",
            )
        )
    elif args.language == "java":
        dataset = pd.read_csv(
            os.path.join(
                os.getcwd(),
                "src",
                args.model,
                "dataset",
                args.language,
                "programs.tsv",
            ),
            delimiter="\t",
            header=None,
        )
    dataset.rename(columns={dataset.columns[1]: "original_string"}, inplace=True)
    dataset["code_tokens"] = dataset["original_string"].apply(lambda x: x.split())
    dataset.drop(
        columns=[
            col
            for col in dataset.columns
            if col not in ["original_string", "code_tokens"]
        ],
        inplace=True,
    )
    # split the dataset into train, validation and test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = int(0.1 * len(dataset))

    train_data = dataset.iloc[:train_size]
    val_data = dataset.iloc[train_size : train_size + val_size]
    test_data = dataset.iloc[train_size + val_size :]
    # save as jsonl
    train_data.to_json(
        os.path.join(
            os.getcwd(),
            "src",
            args.model,
            "dataset",
            args.language,
            "train.jsonl",
        ),
        orient="records",
        lines=True,
    )
    val_data.to_json(
        os.path.join(
            os.getcwd(),
            "src",
            args.model,
            "dataset",
            args.language,
            "valid.jsonl",
        ),
        orient="records",
        lines=True,
    )
    test_data.to_json(
        os.path.join(
            os.getcwd(),
            "src",
            args.model,
            "dataset",
            args.language,
            "test.jsonl",
        ),
        orient="records",
        lines=True,
    )


# x = pd.read_pickle("/Users/ahura/Nexus/Leto/src/ast_nn/dataset/c/programs.pkl")
# print("hi")

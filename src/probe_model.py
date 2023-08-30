import os, torch, yaml, logging, logging.config, warnings, argparse
import numpy as np

from datasets import load_dataset
from utils import probe_utils
from ast_probe.probe.utils import collator_fn_astnn, collator_fn_funcgnn
from ast_probe.probe import (
    ParserProbe,
    ParserLossFuncGNN,
    ParserLoss,
    get_embeddings_astnn,
    get_embeddings_funcgnn,
    FuncGNNParserProbe,
)

warnings.filterwarnings("ignore")
with open(os.path.join(os.getcwd(), "src", "logging_config.yaml"), "r") as f:
    config = yaml.safe_load(f.read())

logging.config.dictConfig(config)

parser = argparse.ArgumentParser(
    prog="LFC Probe",
    description="LFC Probe",
)
parser.add_argument(
    "--device",
    type=str,
    default="cpu" if not torch.cuda.is_available() else "gpu",
    help="Whether to use CPU or GPU for training and evaluating the probe",
)
parser.add_argument(
    "--model",
    type=str,
    help="Model to probe",
    choices=["ast_nn", "funcgnn", "code_sum_drl", "sum_tf", "cscg_dual"],
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
    help="Language of the dataset",
    choices=["java", "c"],
    default="c",
)
parser.add_argument(
    "--train_epochs",
    type=int,
    default=20,
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
    help="Rank of the probe",
)
parser.add_argument(
    "--probe_hidden_dim",
    type=int,
    default=200,
    help="Hidden dimension of the probe",
)

args = parser.parse_args()

device = args.device

if args.model == "ast_nn":
    from ast_nn.src.code_to_repr import code_to_index
    from ast_nn.src.data_pipeline import process_input
    from ast_nn.src.model import BatchProgramCC
    from gensim.models.word2vec import Word2Vec
    from ast_probe.probe import get_embeddings_astnn, collator_fn_astnn

    data_files = {
        "train": os.path.join(args.dataset_path, args.language, "train.jsonl"),
        "valid": os.path.join(args.dataset_path, args.language, "valid.jsonl"),
        "test": os.path.join(args.dataset_path, args.language, "test.jsonl"),
    }

    train_set = load_dataset("json", data_files=data_files, split="train")
    valid_set = load_dataset("json", data_files=data_files, split="valid")
    test_set = load_dataset("json", data_files=data_files, split="test")

    train_set = train_set.map(
        lambda e: code_to_index(
            e["original_string"],
            args.language,
        )
    )
    valid_set = valid_set.map(
        lambda e: code_to_index(
            e["original_string"],
            args.language,
        )
    )
    test_set = test_set.map(
        lambda e: code_to_index(
            e["original_string"],
            args.language,
        )
    )

    max_d_len_train = max([len(x) for x in train_set["d"]])
    max_c_len_train = max([len(j) for x in train_set["c"] for j in x])
    max_u_len_train = max([len(x) for x in train_set["u"]])

    max_d_len_valid = max([len(x) for x in valid_set["d"]])
    max_c_len_valid = max([len(j) for x in valid_set["c"] for j in x])
    max_u_len_valid = max([len(x) for x in valid_set["u"]])

    max_d_len_test = max([len(x) for x in test_set["d"]])
    max_c_len_test = max([len(j) for x in test_set["c"] for j in x])
    max_u_len_test = max([len(x) for x in test_set["u"]])

    max_d_len, max_c_len, max_u_len = (
        max(
            max_d_len_train,
            max_d_len_valid,
            max_d_len_test,
        ),
        max(
            max_c_len_train,
            max_c_len_valid,
            max_c_len_test,
        ),
        max(
            max_u_len_train,
            max_u_len_valid,
            max_u_len_test,
        ),
    )

    word2vec = Word2Vec.load(
        os.path.join(
            args.dataset_path,
            args.language,
            "embeddings",
            "node_w2v_128",
        )
    ).wv

    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]

    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[: word2vec.vectors.shape[0]] = word2vec.vectors

    model_to_probe = BatchProgramCC(
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=100,
        vocab_size=MAX_TOKENS + 1,
        encode_dim=128,
        label_size=1,
        batch_size=32,
        use_gpu=False,
        pretrained_weight=embeddings,
        word2vec_path=os.path.join(
            os.path.join(
                args.dataset_path,
                args.language,
                "embeddings",
                "node_w2v_128",
            )
        ),
        language=args.language,
    )

    model_to_probe.load_state_dict(
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

    probe_model = ParserProbe(
        probe_rank=args.probe_rank,
        hidden_dim=args.probe_hidden_dim,
        number_labels_d=max_d_len,
        number_labels_c=max_c_len,
        number_labels_u=max_u_len,
    ).to(device)

    probe_utils.train_probe(
        embedding_func=get_embeddings_astnn,
        collator_fn=collator_fn_astnn,
        train_dataset=train_set,
        valid_dataset=valid_set,
        batch_size=args.batch_size,
        patience=args.patience,
        probe_model=probe_model,
        probe_loss=ParserLoss(max_c_len=max_c_len),
        model_under_probe=model_to_probe,
        train_epochs=args.train_epochs,
        output_path=os.path.join(os.getcwd(), "results", "astnn"),
    )

elif args.model == "funcgnn":
    from funcgnn.src.code_to_repr import code_to_index
    from funcgnn.src.funcgnn import funcGNNTrainer
    from funcgnn.src.param_parser import parameter_parser
    import json, pandas as pd

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

    train_set = load_dataset("json", data_files=data_files, split="train")
    test_set = load_dataset("json", data_files=data_files, split="test")

    train_set = train_set.map(lambda e: code_to_index(e))
    test_set = test_set.map(lambda e: code_to_index(e))

    max_d_len_train = max([len(x) for x in train_set["d"]])
    max_d_len_test = max([len(x) for x in test_set["d"]])

    max_c_len_train = max([len(x) for x in train_set["c"]])
    max_c_len_test = max([len(x) for x in test_set["c"]])

    max_u_len_train = max([len(x) for x in train_set["u"]])
    max_u_len_test = max([len(x) for x in test_set["u"]])

    funcgnn_param_parser = parameter_parser()
    funcgnn_trainer = funcGNNTrainer(funcgnn_param_parser)
    model_to_probe = funcgnn_trainer.model

    # the embeddings of FuncGNN are of shape 64, probe_rank is still 128
    probe_model = FuncGNNParserProbe(
        probe_rank=args.probe_rank,
        hidden_dim=64,
        number_labels_d=max_d_len_train,
        number_labels_c=max_c_len_train,
        number_labels_u=max_u_len_train,
    ).to(device)

    probe_utils.train_probe(
        embedding_func=get_embeddings_funcgnn,
        collator_fn=collator_fn_funcgnn,
        train_dataset=train_set,
        valid_dataset=test_set,
        batch_size=args.batch_size,
        patience=args.patience,
        probe_model=probe_model,
        probe_loss=ParserLossFuncGNN(max_c_len=max_c_len_train),
        model_under_probe=model_to_probe,
        train_epochs=args.train_epochs,
        output_path=os.path.join(os.getcwd(), "results", "funcgnn"),
    )
    print("hi")

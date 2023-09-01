import os, torch, yaml, logging, logging.config, warnings, argparse, numpy as np
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
#### Arguemnt Parser ####
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
    default="sum_tf",
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
#### Arguemnt Parser ####
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

elif args.model == "sum_tf":
    from summarization_tf.src.code_to_repr import code_to_index
    from summarization_tf.src.models import MultiwayModel
    import pandas as pd
    from summarization_tf.src.utils import read_pickle
    from ast_probe.probe import get_embeddings_sum_tf, collator_fn_sum_tf
    from torch.utils.data import Dataset

    class CustomDataset(Dataset):
        def __init__(self, dataframe):
            self.dataframe = dataframe

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, index):
            return self.dataframe.iloc[index]

    data_files = {
        "train": os.path.join(args.dataset_path, "train.json"),
        "valid": os.path.join(args.dataset_path, "valid.json"),
        "test": os.path.join(args.dataset_path, "test.json"),
    }

    train_set = load_dataset("json", data_files=data_files, split="train[:64]")
    valid_set = load_dataset("json", data_files=data_files, split="valid[:64]")
    test_set = load_dataset("json", data_files=data_files, split="test[:64]")

    train_set_processed = [
        code_to_index(i["code"], i["nl"], idx) for idx, i in enumerate(train_set)
    ]
    valid_set_processed = [
        code_to_index(i["code"], i["nl"], idx) for idx, i in enumerate(valid_set)
    ]
    test_set_processed = [
        code_to_index(i["code"], i["nl"], idx) for idx, i in enumerate(test_set)
    ]

    train_set_processed = [i for i in train_set_processed if i is not None]
    test_set_processed = [i for i in test_set_processed if i is not None]
    valid_set_processed = [i for i in valid_set_processed if i is not None]
    train_set = pd.DataFrame(train_set_processed)
    valid_set = pd.DataFrame(valid_set_processed)
    test_set = pd.DataFrame(test_set_processed)

    max_d_len_train = max([len(x) for x in train_set["d"]])
    max_c_len_train = max([len(x) for x in train_set["c"]])
    max_u_len_train = max([len(x) for x in train_set["u"]])

    max_d_len_valid = max([len(x) for x in valid_set["d"]])
    max_c_len_valid = max([len(x) for x in valid_set["c"]])
    max_u_len_valid = max([len(x) for x in valid_set["u"]])

    max_d_len_test = max([len(x) for x in test_set["d"]])
    max_c_len_test = max([len(x) for x in test_set["c"]])
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

    code_w2i = read_pickle(f"{args.dataset_path}/code_w2i.pkl")
    nl_w2i = read_pickle(f"{args.dataset_path}/nl_w2i.pkl")
    model = MultiwayModel(
        512,
        512,
        512,
        len(code_w2i),
        len(nl_w2i),
        dropout=0.5,
        lr=0.001,
        layer=1,
    )

    train_set = CustomDataset(train_set)
    test_set = CustomDataset(test_set)
    valid_set = CustomDataset(valid_set)
    probe_model = FuncGNNParserProbe(
        probe_rank=args.probe_rank,
        hidden_dim=64,
        number_labels_d=max_d_len_train,
        number_labels_c=max_c_len_train,
        number_labels_u=max_u_len_train,
    ).to(device)

    probe_utils.train_probe(
        embedding_func=get_embeddings_sum_tf,
        collator_fn=collator_fn_sum_tf,
        train_dataset=train_set,
        valid_dataset=test_set,
        batch_size=args.batch_size,
        patience=args.patience,
        probe_model=probe_model,
        probe_loss=ParserLossFuncGNN(max_c_len=max_c_len_train),
        model_under_probe=model,
        train_epochs=args.train_epochs,
        output_path=os.path.join(os.getcwd(), "results", "funcgnn"),
    )
    # first, we need to call each code instance and get the original repr of it to feed to the model
    # second, for each of these call the encode function of the model to get the embeddings
    # send them to the probe to be investigated

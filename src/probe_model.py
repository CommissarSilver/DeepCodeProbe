import os, torch, yaml, logging, logging.config, warnings, argparse, numpy as np
from datasets import load_dataset
from utils import probe_utils


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
    choices=["ast_nn", "funcgnn", "summarization_tf", "code_sum_drl", "cscg_dual"],
    default="code_sum_drl",
)
parser.add_argument(
    "--dataset_path",
    type=str,
    help="Path to the dataset - Path follows the format /model_name/dataset",
    default=os.path.join(os.getcwd(), "src", "code_sum_drl", "dataset"),
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
    default=512,
    choices=[128, 512],
    help="Rank of the probe. 128 for AST-NN and FuncGNN, 512 for SumTF and CodeSumDRL",
)
parser.add_argument(
    "--probe_hidden_dim",
    type=int,
    default=512,
    choices=[200, 512],
    help="Hidden dimension of the probe. 200 for FuncGnn, 512 for SumTF, 512 for CodeSumDRL",
)
args = parser.parse_args()
#### Arguemnt Parser ####
device = args.device

if args.model == "ast_nn":
    from ast_nn.src.code_to_repr import code_to_index
    from ast_nn.src.data_pipeline import process_input
    from ast_nn.src.model import BatchProgramCC
    from gensim.models.word2vec import Word2Vec
    from ast_probe.probe import (
        ParserProbe,
        ParserLoss,
        get_embeddings_astnn,
        collator_fn_astnn,
    )

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
        test_dataset=test_set,
        batch_size=args.batch_size,
        patience=args.patience,
        probe_model=probe_model,
        probe_loss=ParserLoss(max_c_len=max_c_len),
        model_under_probe=model_to_probe,
        train_epochs=args.train_epochs,
        output_path=os.path.join(os.getcwd(), "src", "probe_models", args.model),
    )

elif args.model == "funcgnn":
    from funcgnn.src.code_to_repr import code_to_index
    from funcgnn.src.funcgnn import funcGNNTrainer
    from funcgnn.src.param_parser import parameter_parser

    from ast_probe.probe import (
        FuncGNNParserProbe,
        ParserLossFuncGNN,
        get_embeddings_funcgnn,
        collator_fn_funcgnn,
    )

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

    #!-- this part was required to rename the original files of FuncGNN as they are separated by :::: which is not a valid character in file names
    #! it should only be run once
    # for file in list(data_files["train"]):
    #     new_name = file.replace("::::", "___")
    #     os.rename(file, new_name)
    # for file in list(data_files["test"]):
    #     new_name = file.replace("::::", "___")
    #     os.rename(file, new_name)
    #!-- this part was required to rename the original files of FuncGNN as they are separated by :::: which is not a valid character in file names

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
    model_to_probe.load_state_dict(
        torch.load(
            os.path.join(os.getcwd(), "src", args.model, "models", "model_state.pth")
        )
    )

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
        test_dataset=test_set,
        batch_size=args.batch_size,
        patience=args.patience,
        probe_model=probe_model,
        probe_loss=ParserLossFuncGNN(max_c_len=max_c_len_train),
        model_under_probe=model_to_probe,
        train_epochs=args.train_epochs,
        output_path=os.path.join(os.getcwd(), "src", "probe_models", args.model),
    )

elif args.model == "summarization_tf":
    import pandas as pd
    from torch.utils.data import Dataset

    from summarization_tf.src.code_to_repr import code_to_index
    from summarization_tf.src.models import MultiwayModel
    from summarization_tf.src.utils import read_pickle

    from ast_probe.probe import (
        SumTFParserProbe,
        ParserLossSumTF,
        get_embeddings_sum_tf,
        collator_fn_sum_tf,
    )

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

    train_set = load_dataset("json", data_files=data_files, split="train")
    valid_set = load_dataset("json", data_files=data_files, split="valid")
    test_set = load_dataset("json", data_files=data_files, split="test")

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
    model.load_state_dict(
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

    train_set = CustomDataset(train_set)
    test_set = CustomDataset(test_set)
    valid_set = CustomDataset(valid_set)

    probe_model = SumTFParserProbe(
        probe_rank=args.probe_rank,
        hidden_dim=args.probe_hidden_dim,
        number_labels_d=max_d_len,
        number_labels_c=max_c_len,
        number_labels_u=max_u_len,
    ).to(device)

    probe_utils.train_probe(
        embedding_func=get_embeddings_sum_tf,
        collator_fn=collator_fn_sum_tf,
        train_dataset=train_set,
        valid_dataset=valid_set,
        test_dataset=test_set,
        batch_size=args.batch_size,
        patience=args.patience,
        probe_model=probe_model,
        probe_loss=ParserLossSumTF(max_c_len=max_c_len_train),
        model_under_probe=model,
        train_epochs=args.train_epochs,
        output_path=os.path.join(os.getcwd(), "src", "probe_models", args.model),
    )

elif args.model == "code_sum_drl":
    import pandas as pd

    from torch.utils.data import Dataset
    from code_sum_drl.src.code_to_repr import code_to_index
    from code_sum_drl.src.code_to_repr import Dataset as CodeSumDataset
    import sys

    sys.path.append("/Users/ahura/Nexus/Leto/src/code_sum_drl/src")
    import lib
    from lib.data.Tree import *

    from ast_probe.probe import (
        CodeSumDRLarserProbe,
        ParserLossCodeSumDRL,
        get_embeddings_code_sum_drl,
    )

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

    dataset = torch.load(
        "/Users/ahura/Nexus/Leto/src/code_sum_drl/dataset/train/processed_all_new.train.pt"
    )

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

    dicts = dataset["dicts"]
    dataset["train_xe"]["trees"] = get_data_trees(dataset["train_xe"]["trees"])
    dataset["train_pg"]["trees"] = get_data_trees(dataset["train_pg"]["trees"])
    dataset["valid"]["trees"] = get_data_trees(dataset["valid"]["trees"])
    dataset["test"]["trees"] = get_data_trees(dataset["test"]["trees"])

    dataset["train_xe"]["leafs"] = get_data_leafs(
        dataset["train_xe"]["trees"],
        dicts["src"],
    )
    dataset["train_pg"]["leafs"] = get_data_leafs(
        dataset["train_pg"]["trees"],
        dicts["src"],
    )
    dataset["valid"]["leafs"] = get_data_leafs(
        dataset["valid"]["trees"],
        dicts["src"],
    )
    dataset["test"]["leafs"] = get_data_leafs(
        dataset["test"]["trees"],
        dicts["src"],
    )

    train_set = CodeSumDataset(
        dataset["train_xe"],
        args.batch_size,
        False,
        eval=False,
    )
    valid_set = CodeSumDataset(
        dataset["valid"],
        args.batch_size,
        False,
        eval=True,
    )
    test_set = CodeSumDataset(
        dataset["test"],
        args.batch_size,
        False,
        eval=True,
    )
    max_d, max_c, max_u = 0, 0, 0

    for i in range(len(train_set)):
        batch = train_set[i]
        ds = [dcu["d"] for dcu in batch[6]]
        cs = [dcu["c"] for dcu in batch[6]]
        us = [dcu["u"] for dcu in batch[6]]

        max_d = max(max_d, max([len(d) for d in ds]))
        max_c = max(max_c, max([len(c) for c in cs]))
        max_u = max(max_u, max([len(u) for u in us]))

    #! No longer required
    # print("mac D: ", max_d)
    # print("mac C: ", max_c)
    # print("mac U: ", max_u)

    code_encoder = lib.TreeEncoder(opt, dicts["src"])
    text_encoder = lib.Encoder(opt, dicts["src"])
    decoder = lib.HybridDecoder(opt, dicts["tgt"])
    generator = lib.BaseGenerator(torch.nn.Linear(opt.rnn_size, dicts["tgt"].size()), opt)
    model = lib.Hybrid2SeqModel(code_encoder, text_encoder, decoder, generator, opt)

    # checkpoint = torch.load(
    #     os.path.join(
    #         os.getcwd(),
    #         "src",
    #         args.model,
    #         "models",
    #         "model_state.pt",
    #     ),
    #     map_location=torch.device("cpu"),
    # )
    # model = checkpoint["model"]

    probe_model = CodeSumDRLarserProbe(
        probe_rank=args.probe_rank,
        hidden_dim=args.probe_hidden_dim,
        number_labels_d=max_d,
        number_labels_c=max_c,
        number_labels_u=max_u,
    ).to(device)

    probe_utils.train_probe_code_sum_drl(
        embedding_func=get_embeddings_code_sum_drl,
        train_dataset=train_set,
        valid_dataset=valid_set,
        test_dataset=test_set,
        batch_size=args.batch_size,
        patience=args.patience,
        probe_model=probe_model,
        probe_loss=ParserLossCodeSumDRL(),
        model_under_probe=model,
        train_epochs=args.train_epochs,
        output_path=os.path.join(os.getcwd(), "src", "probe_models", args.model),
    )

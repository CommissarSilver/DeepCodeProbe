import os, torch, yaml, logging, logging.config, warnings
import numpy as np

from datasets import load_dataset
from utils import probe_utils
from ast_probe.probe.utils import collator_fn
from ast_probe.probe import ParserProbe, ParserLoss, get_embeddings

warnings.filterwarnings("ignore")
with open(os.path.join(os.getcwd(), "src", "logging_config.yaml"), "r") as f:
    config = yaml.safe_load(f.read())

logging.config.dictConfig(config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "astnn"
language = "java"
dataset_path = "/Users/ahura/Nexus/Leto/dataset/ast-nn"

train_epochs = 20
batch_size = 32
patience = 5

probe_rank = 128
probe_hidden_dim = 200

if model_name == "astnn":
    from ast_nn.src.code_to_repr import code_to_index
    from ast_nn.src.data_pipeline import process_input
    from ast_nn.src.model import BatchProgramCC
    from gensim.models.word2vec import Word2Vec
    from ast_probe.probe import get_embeddings

    data_files = {
        "train": os.path.join(dataset_path, "train.jsonl"),
        "valid": os.path.join(dataset_path, "valid.jsonl"),
        "test": os.path.join(dataset_path, "test.jsonl"),
    }

    train_set = load_dataset("json", data_files=data_files, split="train[:32768]")
    valid_set = load_dataset("json", data_files=data_files, split="valid[:4096]")
    test_set = load_dataset("json", data_files=data_files, split="test[:2048]")

    train_set = train_set.map(lambda e: code_to_index(e["original_string"], language))
    valid_set = valid_set.map(lambda e: code_to_index(e["original_string"], language))
    test_set = test_set.map(lambda e: code_to_index(e["original_string"], language))

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
            os.getcwd(),
            "src",
            "ast_nn",
            "dataset",
            language,
            "embeddings",
            "node_w2v_128",
        )
    ).wv

    MAX_TOKENS = word2vec.syn0.shape[0]
    EMBEDDING_DIM = word2vec.syn0.shape[1]

    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[: word2vec.syn0.shape[0]] = word2vec.syn0

    model_to_probe = BatchProgramCC(
        embedding_dim=word2vec.syn0.shape[1],
        hidden_dim=100,
        vocab_size=word2vec.syn0.shape[0] + 1,
        encode_dim=128,
        label_size=1,
        batch_size=32,
        use_gpu=False,
        pretrained_weight=embeddings,
        word2vec_path=os.path.join(
            os.getcwd(),
            "src",
            "ast_nn",
            "dataset",
            language,
            "embeddings",
            "node_w2v_128",
        ),
        language=language,
    )

    if language == "c":
        model_to_probe.load_state_dict(
            torch.load(
                os.path.join(
                    os.getcwd(),
                    "src",
                    "ast_nn",
                    "trained_models",
                    "astnn_model_c.pkl",
                )
            )
        )
    elif language == "java":
        model_to_probe.load_state_dict(
            torch.load(
                os.path.join(
                    os.getcwd(),
                    "src",
                    "ast_nn",
                    "trained_models",
                    "astnn_model_java_category_4.pkl",
                )
            )
        )

    probe_model = ParserProbe(
        probe_rank=probe_rank,
        hidden_dim=probe_hidden_dim,
        number_labels_d=max_d_len,
        number_labels_c=max_c_len,
        number_labels_u=max_u_len,
    ).to(device)

    probe_utils.train_probe(
        embedding_func=get_embeddings,
        train_dataset=train_set,
        valid_dataset=valid_set,
        batch_size=batch_size,
        patience=patience,
        probe_model=probe_model,
        probe_loss=ParserLoss(max_c_len=max_c_len),
        model_under_probe=model_to_probe,
        train_epochs=train_epochs,
        output_path=os.path.join(os.getcwd(), "results", "astnn"),
    )

    print("Done!")

import os, argparse, logging, pickle, torch
import numpy as np

from collections import defaultdict
from torch.utils.data import DataLoader
from datasets import load_dataset
from tree_sitter import Parser
from tqdm import tqdm

from ast_nn.src.code_to_repr import code_to_index

from ast_probe.data import (
    convert_sample_to_features,
    PY_LANGUAGE,
    JAVA_LANGUAGE,
)
from ast_probe.probe import ParserProbe, ParserLoss, get_embeddings
from ast_probe.data.utils import (
    match_tokenized_to_untokenized_roberta,
    remove_comments_and_docstrings_java_js,
    remove_comments_and_docstrings_python,
)
from ast_probe.probe.utils import collator_fn
from ast_probe.data.data_loading import get_non_terminals_labels, convert_to_ids
from ast_probe.data.binary_tree import (
    distance_to_tree,
    remove_empty_nodes,
    extend_complex_nodes,
    get_precision_recall_f1,
    add_unary,
    get_recall_non_terminal,
)

logger = logging.getLogger(__name__)


def run_probing_train(
    args: argparse.Namespace,
    language="java",
    dataset_path="/Users/ahura/Nexus/Leto/dataset/ast-nn",
    batch_size=32,
    model_type="astnn",
    probe_rank=128,
    layer=None,
):
    logger.info("Loading dataset from local file.")
    data_files = {
        "train": os.path.join(dataset_path, "train.jsonl"),
        "valid": os.path.join(dataset_path, "valid.jsonl"),
        "test": os.path.join(dataset_path, "test.jsonl"),
    }

    train_set = load_dataset("json", data_files=data_files, split="train[:128]")
    valid_set = load_dataset("json", data_files=data_files, split="valid[:128]")
    test_set = load_dataset("json", data_files=data_files, split="test[:128]")

    train_set = train_set.map(lambda e: code_to_index(e["original_string"], language))
    valid_set = valid_set.map(lambda e: code_to_index(e["original_string"], language))
    test_set = test_set.map(lambda e: code_to_index(e["original_string"], language))

    max_d_len = max([len(x) for x in train_set["d"]])
    max_c_len = max([len(j) for x in train_set["c"] for j in x])
    max_u_len = max([len(x) for x in train_set["u"]])

    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collator_fn(batch),
        num_workers=0,
    )
    valid_dataloader = DataLoader(
        dataset=valid_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collator_fn(batch),
        num_workers=0,
    )

    if model_type == "astnn":
        from ast_nn.src.data_pipeline import process_input
        from ast_nn.src.model import BatchProgramCC
        from gensim.models.word2vec import Word2Vec

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

        lmodel = BatchProgramCC(
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
            lmodel.load_state_dict(
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
            lmodel.load_state_dict(
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

        logger.info(f"Loaded ASTNN_{language} model.")

    device = "cpu"
    lmodel = lmodel.to(device)

    probe_model = ParserProbe(
        probe_rank=probe_rank,
        hidden_dim=200,
        number_labels_d=max_d_len,
        number_labels_c=max_c_len,
        number_labels_u=max_u_len,
    ).to(device)

    optimizer = torch.optim.Adam(probe_model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=0
    )
    criterion = ParserLoss(max_c_len=max_c_len)

    probe_model.train()
    lmodel.eval()
    best_eval_loss = float("inf")
    metrics = {
        "training_loss": [],
        "validation_loss": [],
        "test_precision": None,
        "test_recall": None,
        "test_f1": None,
    }
    patience_count = 0
    for epoch in range(1, 20 + 1):
        training_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            ds, cs, us, batch_len_tokens, original_code_strings = batch

            embds = get_embeddings(original_code_strings, "astnn", lmodel)

            d_pred, c_pred, u_pred = probe_model(embds.to(device))

            loss = criterion(
                d_pred=d_pred.to(device),
                c_pred=c_pred.to(device),
                u_pred=u_pred.to(device),
                d_real=torch.tensor(ds).to(device),
                c_real=torch.tensor(cs).to(device),
                u_real=torch.tensor(us).to(device),
                length_batch=batch_len_tokens.to(device),
            )

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            training_loss += loss.item()

        training_loss = training_loss / len(train_dataloader)
        eval_loss, _, _, _ = run_probing_eval(
            valid_dataloader, probe_model, lmodel, criterion, args
        )
        scheduler.step(eval_loss)
        logger.info(
            f"[epoch {epoch}] train loss: {round(training_loss, 4)}, validation loss: {round(eval_loss, 4)}"
        )

        metrics["training_loss"].append(round(training_loss, 4))
        metrics["validation_loss"].append(round(eval_loss, 4))

        if eval_loss < best_eval_loss:
            logger.info("-" * 100)
            logger.info("Saving model checkpoint")
            logger.info("-" * 100)
            output_path = os.path.join(args.output_path, f"pytorch_model.bin")
            torch.save(probe_model.state_dict(), output_path)
            logger.info(f"Probe model saved: {output_path}")
            patience_count = 0
            best_eval_loss = eval_loss
        else:
            patience_count += 1
        if patience_count == args.patience:
            logger.info("Stopping training loop (out of patience).")
            break

    logger.info("Loading test set.")
    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collator_fn(batch),
        num_workers=8,
    )

    logger.info("Loading best model.")
    checkpoint = torch.load(os.path.join(args.output_path, "pytorch_model.bin"))
    probe_model.load_state_dict(checkpoint)

    logger.info("Evaluating probing on test set.")
    eval_precision, eval_recall, eval_f1_score = run_probing_eval_f1(
        test_dataloader, probe_model, lmodel, ids_to_labels_c, ids_to_labels_u, args
    )
    metrics["test_precision"] = round(eval_precision, 4)
    metrics["test_recall"] = round(eval_recall, 4)
    metrics["test_f1"] = round(eval_f1_score, 4)
    logger.info(
        f"test precision: {round(eval_precision, 4)} | test recall: {round(eval_recall, 4)} "
        f"| test F1 score: {round(eval_f1_score, 4)}"
    )

    logger.info("-" * 100)
    logger.info("Saving metrics.")
    with open(os.path.join(args.output_path, "metrics.log"), "wb") as f:
        pickle.dump(metrics, f)


def run_probing_eval(valid_dataloader, probe_model, lmodel, criterions):
    probe_model.eval()
    eval_loss = 0.0
    
    with torch.no_grad():
        for step, batch in enumerate(
            tqdm(
                valid_dataloader,
                desc="[test batch]",
                bar_format="{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}",
            )
        ):
            ds, cs, us, batch_len_tokens, original_code_strings = batch

            ds = ds.to(args.device)
            cs = cs.to(args.device)
            us = us.to(args.device)

            embds = get_embeddings(original_code_strings, "astnn", lmodel, 0)

            d_pred, c_pred, u_pred = probe_model(embds.to(device))

            loss = criterion(
                d_pred=d_pred.to(device),
                c_pred=c_pred.to(device),
                u_pred=u_pred.to(device),
                d_real=torch.tensor(ds).to(device),
                c_real=torch.tensor(cs).to(device),
                u_real=torch.tensor(us).to(device),
                length_batch=batch_len_tokens.to(device),
            )

            eval_loss += loss.item()

            d_acc, c_acc, u_acc = ParserLoss.calculate_accuracy(
                d_pred, c_pred, u_pred, ds, cs, us, batch_len_tokens
            )
        return (eval_loss / len(valid_dataloader)), d_acc, c_acc, u_acc


if __name__ == "__main__":
    args = argparse.Namespace()
    run_probing_train(args)


# def run_probing_eval_f1(
#     test_dataloader, probe_model, lmodel, ids_to_labels_c, ids_to_labels_u, args
# ):
#     probe_model.eval()
#     precisions, recalls, f1_scores = [], [], []
#     with torch.no_grad():
#         for step, batch in enumerate(
#             tqdm(
#                 test_dataloader,
#                 desc="[test batch]",
#                 bar_format="{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}",
#             )
#         ):
#             all_inputs, all_attentions, ds, cs, us, batch_len_tokens, alignment = batch

#             embds = get_embeddings(
#                 all_inputs.to(args.device),
#                 all_attentions.to(args.device),
#                 lmodel,
#                 args.layer,
#                 args.model_type,
#             )
#             embds = align_function(embds.to(args.device), alignment.to(args.device))

#             d_pred, scores_c, scores_u = probe_model(embds.to(args.device))
#             scores_c = torch.argmax(scores_c, dim=2)
#             scores_u = torch.argmax(scores_u, dim=2)

#             for i, len_tokens in enumerate(batch_len_tokens):
#                 len_tokens = len_tokens.item()
#                 d_pred_current = d_pred[i, 0 : len_tokens - 1].tolist()
#                 score_c_current = scores_c[i, 0 : len_tokens - 1].tolist()
#                 score_u_current = scores_u[i, 0:len_tokens].tolist()
#                 ds_current = ds[i, 0 : len_tokens - 1].tolist()
#                 cs_current = cs[i, 0 : len_tokens - 1].tolist()
#                 us_current = us[i, 0:len_tokens].tolist()

#                 cs_labels = [ids_to_labels_c[c] for c in cs_current]
#                 us_labels = [ids_to_labels_u[c] for c in us_current]
#                 scores_c_labels = [ids_to_labels_c[s] for s in score_c_current]
#                 scores_u_labels = [ids_to_labels_u[s] for s in score_u_current]

#                 ground_truth_tree = distance_to_tree(
#                     ds_current, cs_labels, us_labels, [str(i) for i in range(len_tokens)]
#                 )
#                 ground_truth_tree = extend_complex_nodes(
#                     add_unary(remove_empty_nodes(ground_truth_tree))
#                 )

#                 pred_tree = distance_to_tree(
#                     d_pred_current,
#                     scores_c_labels,
#                     scores_u_labels,
#                     [str(i) for i in range(len_tokens)],
#                 )
#                 pred_tree = extend_complex_nodes(add_unary(remove_empty_nodes(pred_tree)))

#                 p, r, f1_score = get_precision_recall_f1(ground_truth_tree, pred_tree)
#                 f1_scores.append(f1_score)
#                 precisions.append(p)
#                 recalls.append(r)

#     return np.mean(precisions), np.mean(recalls), np.mean(f1_scores)


# def run_probing_eval_recall_non_terminal(
#     test_dataloader, probe_model, lmodel, ids_to_labels_c, ids_to_labels_u, args
# ):
#     probe_model.eval()
#     recall_scores = defaultdict(list)
#     with torch.no_grad():
#         for step, batch in enumerate(
#             tqdm(
#                 test_dataloader,
#                 desc="[test batch]",
#                 bar_format="{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}",
#             )
#         ):
#             all_inputs, all_attentions, ds, cs, us, batch_len_tokens, alignment = batch

#             embds = get_embeddings(
#                 all_inputs.to(args.device),
#                 all_attentions.to(args.device),
#                 lmodel,
#                 args.layer,
#                 args.model_type,
#             )
#             embds = align_function(embds.to(args.device), alignment.to(args.device))

#             d_pred, scores_c, scores_u = probe_model(embds.to(args.device))
#             scores_c = torch.argmax(scores_c, dim=2)
#             scores_u = torch.argmax(scores_u, dim=2)

#             for i, len_tokens in enumerate(batch_len_tokens):
#                 len_tokens = len_tokens.item()
#                 d_pred_current = d_pred[i, 0 : len_tokens - 1].tolist()
#                 score_c_current = scores_c[i, 0 : len_tokens - 1].tolist()
#                 score_u_current = scores_u[i, 0:len_tokens].tolist()
#                 ds_current = ds[i, 0 : len_tokens - 1].tolist()
#                 cs_current = cs[i, 0 : len_tokens - 1].tolist()
#                 us_current = us[i, 0:len_tokens].tolist()

#                 cs_labels = [ids_to_labels_c[c] for c in cs_current]
#                 us_labels = [ids_to_labels_u[c] for c in us_current]
#                 scores_c_labels = [ids_to_labels_c[s] for s in score_c_current]
#                 scores_u_labels = [ids_to_labels_u[s] for s in score_u_current]

#                 ground_truth_tree = distance_to_tree(
#                     ds_current, cs_labels, us_labels, [str(i) for i in range(len_tokens)]
#                 )
#                 ground_truth_tree = extend_complex_nodes(
#                     add_unary(remove_empty_nodes(ground_truth_tree))
#                 )

#                 pred_tree = distance_to_tree(
#                     d_pred_current,
#                     scores_c_labels,
#                     scores_u_labels,
#                     [str(i) for i in range(len_tokens)],
#                 )
#                 pred_tree = extend_complex_nodes(add_unary(remove_empty_nodes(pred_tree)))

#                 recall_score = get_recall_non_terminal(ground_truth_tree, pred_tree)
#                 for k, s in recall_score.items():
#                     recall_scores[k].append(s)

#     return {k: np.mean(v) for k, v in recall_scores.items()}


# def run_probing_test(args):
#     logger.info("-" * 100)
#     logger.info("Running probing test.")
#     logger.info("-" * 100)

#     # select the parser
#     parser = Parser()
#     if args.lang == "python":
#         parser.set_language(PY_LANGUAGE)
#     elif args.lang == "javascript":
#         parser.set_language(JS_LANGUAGE)
#     elif args.lang == "go":
#         parser.set_language(GO_LANGUAGE)
#     elif args.lang == "java":
#         parser.set_language(JAVA_LANGUAGE)
#     elif args.lang == "php":
#         parser.set_language(PHP_LANGUAGE)
#     elif args.lang == "ruby":
#         parser.set_language(RUBY_LANGUAGE)

#     logger.info("Loading tokenizer")
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"
#     tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

#     logger.info("Loading dataset from local file.")
#     data_files = {"test": os.path.join(args.dataset_name_or_path, "test.jsonl")}

#     # get class labels-ids mapping for c and u
#     labels_file_path = os.path.join(args.dataset_name_or_path, "labels.pkl")
#     with open(labels_file_path, "rb") as f:
#         data = pickle.load(f)
#         labels_to_ids_c = data["labels_to_ids_c"]
#         ids_to_labels_c = data["ids_to_labels_c"]
#         labels_to_ids_u = data["labels_to_ids_u"]
#         ids_to_labels_u = data["ids_to_labels_u"]

#     test_set = load_dataset("json", data_files=data_files, split="test")

#     # get d and c for each sample
#     test_set = test_set.map(
#         lambda e: convert_sample_to_features(e["original_string"], parser, args.lang)
#     )
#     test_set = test_set.map(lambda e: convert_to_ids(e["c"], "c", labels_to_ids_c))
#     test_set = test_set.map(lambda e: convert_to_ids(e["u"], "u", labels_to_ids_u))

#     test_dataloader = DataLoader(
#         dataset=test_set,
#         batch_size=args.batch_size,
#         shuffle=False,
#         collate_fn=lambda batch: collator_fn(batch, tokenizer),
#         num_workers=8,
#     )

#     logger.info("Loading models.")
#     if args.model_type == "t5":
#         lmodel = T5EncoderModel.from_pretrained(
#             args.pretrained_model_name_or_path, output_hidden_states=True
#         )
#     else:
#         lmodel = AutoModel.from_pretrained(
#             args.pretrained_model_name_or_path, output_hidden_states=True
#         )
#         if "-baseline" in args.run_name:
#             lmodel = generate_baseline(lmodel)
#     lmodel = lmodel.to(args.device)

#     probe_model = ParserProbe(
#         probe_rank=args.rank,
#         hidden_dim=args.hidden,
#         number_labels_c=len(labels_to_ids_c),
#         number_labels_u=len(labels_to_ids_u),
#     ).to(args.device)
#     criterion = ParserLoss(loss="rank")

#     if args.model_checkpoint:
#         logger.info("Restoring model checkpoint.")
#         checkpoint = torch.load(os.path.join(args.model_checkpoint, "pytorch_model.bin"))
#         probe_model.load_state_dict(checkpoint)

#     probe_model.eval()
#     lmodel.eval()
#     eval_loss, acc_c, acc_u, acc_d = run_probing_eval(
#         test_dataloader, probe_model, lmodel, criterion, args
#     )
#     logger.info(
#         f"test loss: {eval_loss} | acc_c: {acc_c} | acc_u: {acc_u} | acc_d: {acc_d}"
#     )

#     recall_dict = run_probing_eval_recall_non_terminal(
#         test_dataloader, probe_model, lmodel, ids_to_labels_c, ids_to_labels_u, args
#     )

#     for v, k in sorted(((v, k) for k, v in recall_dict.items()), reverse=True):
#         logger.info(f"Non-terminal {k} | recall {v}")


# def run_probing_direct_transfer_train(args):
#     logger.info("-" * 100)
#     logger.info("Running probing training.")
#     logger.info("-" * 100)

#     # select the parser
#     parser = Parser()
#     if args.lang == "python":
#         parser.set_language(PY_LANGUAGE)
#     elif args.lang == "javascript":
#         parser.set_language(JS_LANGUAGE)
#     elif args.lang == "go":
#         parser.set_language(GO_LANGUAGE)
#     elif args.lang == "java":
#         parser.set_language(JAVA_LANGUAGE)
#     elif args.lang == "php":
#         parser.set_language(PHP_LANGUAGE)
#     elif args.lang == "ruby":
#         parser.set_language(RUBY_LANGUAGE)

#     logger.info("Loading tokenizer")
#     tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)

#     logger.info("Loading dataset from local file.")
#     data_files = {
#         "train": os.path.join(args.dataset_name_or_path, "train.jsonl"),
#         "valid": os.path.join(args.dataset_name_or_path, "valid.jsonl"),
#         "test": os.path.join(args.dataset_name_or_path, "test.jsonl"),
#     }

#     train_set = load_dataset("json", data_files=data_files, split="train")
#     valid_set = load_dataset("json", data_files=data_files, split="valid")
#     test_set = load_dataset("json", data_files=data_files, split="test")

#     # get d and c for each sample
#     train_set = train_set.map(
#         lambda e: convert_sample_to_features(e["original_string"], parser, args.lang)
#     )
#     valid_set = valid_set.map(
#         lambda e: convert_sample_to_features(e["original_string"], parser, args.lang)
#     )
#     test_set = test_set.map(
#         lambda e: convert_sample_to_features(e["original_string"], parser, args.lang)
#     )

#     # get class labels-ids mapping for c and u
#     labels_file_path = os.path.join(args.dataset_name_or_path, "labels.pkl")
#     if not os.path.exists(labels_file_path):
#         # convert each non-terminal labels to its id
#         labels_to_ids_c = get_non_terminals_labels(
#             train_set["c"], valid_set["c"], test_set["c"]
#         )
#         ids_to_labels_c = {x: y for y, x in labels_to_ids_c.items()}
#         labels_to_ids_u = get_non_terminals_labels(
#             train_set["u"], valid_set["u"], test_set["u"]
#         )
#         ids_to_labels_u = {x: y for y, x in labels_to_ids_u.items()}
#         with open(labels_file_path, "wb") as f:
#             pickle.dump(
#                 {
#                     "labels_to_ids_c": labels_to_ids_c,
#                     "ids_to_labels_c": ids_to_labels_c,
#                     "labels_to_ids_u": labels_to_ids_u,
#                     "ids_to_labels_u": ids_to_labels_u,
#                 },
#                 f,
#             )
#     else:
#         with open(labels_file_path, "rb") as f:
#             data = pickle.load(f)
#             labels_to_ids_c = data["labels_to_ids_c"]
#             ids_to_labels_c = data["ids_to_labels_c"]
#             labels_to_ids_u = data["labels_to_ids_u"]
#             ids_to_labels_u = data["ids_to_labels_u"]

#     train_set = train_set.map(lambda e: convert_to_ids(e["c"], "c", labels_to_ids_c))
#     valid_set = valid_set.map(lambda e: convert_to_ids(e["c"], "c", labels_to_ids_c))
#     test_set = test_set.map(lambda e: convert_to_ids(e["c"], "c", labels_to_ids_c))

#     train_set = train_set.map(lambda e: convert_to_ids(e["u"], "u", labels_to_ids_u))
#     valid_set = valid_set.map(lambda e: convert_to_ids(e["u"], "u", labels_to_ids_u))
#     test_set = test_set.map(lambda e: convert_to_ids(e["u"], "u", labels_to_ids_u))

#     train_dataloader = DataLoader(
#         dataset=train_set,
#         batch_size=args.batch_size,
#         shuffle=True,
#         collate_fn=lambda batch: collator_fn(batch, tokenizer),
#         num_workers=8,
#     )
#     valid_dataloader = DataLoader(
#         dataset=valid_set,
#         batch_size=args.batch_size,
#         shuffle=False,
#         collate_fn=lambda batch: collator_fn(batch, tokenizer),
#         num_workers=8,
#     )

#     logger.info("Loading models.")
#     if args.model_type == "t5":
#         lmodel = T5EncoderModel.from_pretrained(
#             args.pretrained_model_name_or_path, output_hidden_states=True
#         )
#     else:
#         lmodel = AutoModel.from_pretrained(
#             args.pretrained_model_name_or_path, output_hidden_states=True
#         )
#         if "-baseline" in args.run_name:
#             lmodel = generate_baseline(lmodel)
#     lmodel = lmodel.to(args.device)

#     probe_model = ParserProbe(
#         probe_rank=args.rank,
#         hidden_dim=args.hidden,
#         number_labels_c=len(labels_to_ids_c),
#         number_labels_u=len(labels_to_ids_u),
#     ).to(args.device)

#     logger.info("Loading model checkpoint.")
#     checkpoint = torch.load(
#         os.path.join(args.model_source_checkpoint, "pytorch_model.bin")
#     )
#     probe_model.proj = torch.nn.Parameter(data=checkpoint["proj"])

#     optimizer = torch.optim.Adam(
#         [probe_model.vectors_c, probe_model.vectors_u], lr=args.lr
#     )
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", factor=0.1, patience=0
#     )
#     criterion = ParserLoss(loss="rank", pretrained=True)

#     probe_model.train()
#     lmodel.eval()
#     best_eval_loss = float("inf")
#     metrics = {
#         "training_loss": [],
#         "validation_loss": [],
#         "test_precision": None,
#         "test_recall": None,
#         "test_f1": None,
#     }
#     patience_count = 0
#     for epoch in range(1, args.epochs + 1):
#         training_loss = 0.0
#         for step, batch in enumerate(
#             tqdm(
#                 train_dataloader,
#                 desc="[training batch]",
#                 bar_format="{desc:<10}{percentage:3.0f}%|{bar:100}{r_bar}",
#             )
#         ):
#             all_inputs, all_attentions, ds, cs, us, batch_len_tokens, alignment = batch

#             embds = get_embeddings(
#                 all_inputs.to(args.device),
#                 all_attentions.to(args.device),
#                 lmodel,
#                 args.layer,
#                 args.model_type,
#             )
#             embds = align_function(embds.to(args.device), alignment.to(args.device))

#             d_pred, scores_c, scores_u = probe_model(embds.to(args.device))
#             loss = criterion(
#                 d_pred=d_pred.to(args.device),
#                 scores_c=scores_c.to(args.device),
#                 scores_u=scores_u.to(args.device),
#                 d_real=ds.to(args.device),
#                 c_real=cs.to(args.device),
#                 u_real=us.to(args.device),
#                 length_batch=batch_len_tokens.to(args.device),
#             )

#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#             training_loss += loss.item()

#         training_loss = training_loss / len(train_dataloader)
#         eval_loss, _, _, _ = run_probing_eval(
#             valid_dataloader, probe_model, lmodel, criterion, args
#         )
#         scheduler.step(eval_loss)
#         logger.info(
#             f"[epoch {epoch}] train loss: {round(training_loss, 4)}, validation loss: {round(eval_loss, 4)}"
#         )
#         metrics["training_loss"].append(round(training_loss, 4))
#         metrics["validation_loss"].append(round(eval_loss, 4))

#         if eval_loss < best_eval_loss:
#             logger.info("-" * 100)
#             logger.info("Saving model checkpoint")
#             logger.info("-" * 100)
#             output_path = os.path.join(args.output_path, f"pytorch_model.bin")
#             torch.save(probe_model.state_dict(), output_path)
#             logger.info(f"Probe model saved: {output_path}")
#             patience_count = 0
#             best_eval_loss = eval_loss
#         else:
#             patience_count += 1
#         if patience_count == args.patience:
#             logger.info("Stopping training loop (out of patience).")
#             break

#     logger.info("Loading test set.")
#     test_dataloader = DataLoader(
#         dataset=test_set,
#         batch_size=args.batch_size,
#         shuffle=False,
#         collate_fn=lambda batch: collator_fn(batch, tokenizer),
#         num_workers=8,
#     )

#     logger.info("Loading best model.")
#     checkpoint = torch.load(os.path.join(args.output_path, "pytorch_model.bin"))
#     probe_model.load_state_dict(checkpoint)

#     logger.info("Evaluating probing on test set.")
#     eval_precision, eval_recall, eval_f1_score = run_probing_eval_f1(
#         test_dataloader, probe_model, lmodel, ids_to_labels_c, ids_to_labels_u, args
#     )
#     metrics["test_precision"] = round(eval_precision, 4)
#     metrics["test_recall"] = round(eval_recall, 4)
#     metrics["test_f1"] = round(eval_f1_score, 4)
#     logger.info(
#         f"test precision: {round(eval_precision, 4)} | test recall: {round(eval_recall, 4)} "
#         f"| test F1 score: {round(eval_f1_score, 4)}"
#     )

#     logger.info("-" * 100)
#     logger.info("Saving metrics.")
#     with open(os.path.join(args.output_path, "metrics.log"), "wb") as f:
#         pickle.dump(metrics, f)

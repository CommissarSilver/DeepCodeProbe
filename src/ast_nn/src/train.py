import pandas as pd
import torch
import os
import time
import numpy as np
import warnings
from gensim.models.word2vec import Word2Vec
from model import BatchProgramCC
from torch.autograd import Variable
from sklearn.metrics import precision_recall_fscore_support

warnings.filterwarnings("ignore")


def get_batch(dataset, idx, bs):
    tmp = dataset.iloc[idx : idx + bs]
    x1, x2, labels = [], [], []
    for _, item in tmp.iterrows():
        x1.append(item["code_x"])
        x2.append(item["code_y"])
        labels.append([item["label"]])
    return x1, x2, torch.FloatTensor(labels)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Choose a dataset:[c|java]")
    parser.add_argument("--lang", default="java")
    args = parser.parse_args()
    if not args.lang:
        print("No specified dataset")
        exit(1)
    root = os.path.join(os.getcwd(), "src", "ast_nn", "dataset")
    lang = args.lang
    categories = 1
    if lang == "java":
        categories = 5
    print("Train for ", str.upper(lang))
    train_data = pd.read_pickle(os.path.join(root, lang, "train", "blocks.pkl")).sample(
        frac=1
    )
    test_data = pd.read_pickle(os.path.join(root, lang, "test", "blocks.pkl")).sample(
        frac=1
    )

    # word2vec = Word2Vec.load(root + lang + "/train/embedding/node_w2v_128").wv
    word2vec = Word2Vec.load(os.path.join(root, lang, "embeddings", "node_w2v_128")).wv

    MAX_TOKENS = word2vec.vectors.shape[0]
    EMBEDDING_DIM = word2vec.vectors.shape[1]
    embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
    embeddings[: word2vec.vectors.shape[0]] = word2vec.vectors

    HIDDEN_DIM = 512
    ENCODE_DIM = 512
    LABELS = 1
    EPOCHS = 7
    BATCH_SIZE = 32
    USE_GPU = False

    model = BatchProgramCC(
        EMBEDDING_DIM,
        HIDDEN_DIM,
        MAX_TOKENS + 1,
        ENCODE_DIM,
        LABELS,
        BATCH_SIZE,
        USE_GPU,
        embeddings,
        word2vec_path=os.path.join(root, lang, "embeddings", "node_w2v_128"),
    )
    if USE_GPU:
        model.cuda()

    parameters = model.parameters()
    optimizer = torch.optim.Adamax(parameters)
    loss_function = torch.nn.BCELoss()

    print(train_data)
    precision, recall, f1 = 0, 0, 0
    print("Start training...")
    for t in range(1, categories + 1):
        if lang == "java":
            train_data_t = train_data[train_data["label"].isin([t, 0])]
            train_data_t.loc[train_data_t["label"] > 0, "label"] = 1

            test_data_t = test_data[test_data["label"].isin([t, 0])]
            test_data_t.loc[test_data_t["label"] > 0, "label"] = 1
        else:
            train_data_t, test_data_t = train_data, test_data
        # training procedure
        for epoch in range(EPOCHS):
            start_time = time.time()
            # training epoch
            total_acc = 0.0
            total_loss = 0.0
            total = 0.0
            i = 0
            while i < len(train_data_t):
                batch = get_batch(train_data_t, i, BATCH_SIZE)
                i += BATCH_SIZE
                train1_inputs, train2_inputs, train_labels = batch
                if USE_GPU:
                    train1_inputs, train2_inputs, train_labels = (
                        train1_inputs,
                        train2_inputs,
                        train_labels.cuda(),
                    )

                model.zero_grad()
                model.batch_size = len(train_labels)
                model.hidden = model.init_hidden()
                output = model(train1_inputs, train2_inputs)

                loss = loss_function(output, Variable(train_labels))
                loss.backward()
                optimizer.step()

                if i % 50 == 0:
                    print(f"Epoch {epoch} - {i}: loss = {loss.item()}")
        # save the model
        if not os.path.exists(os.path.join(os.getcwd(), "src", "ast_nn", "models")):
            os.mkdir(os.path.join(os.getcwd(), "src", "ast_nn", "models"))
        torch.save(
            model.state_dict(),
            os.path.join(
                os.getcwd(),
                "src",
                "ast_nn",
                "models",
                "astnn_" + str.upper(lang) + "_" + str(t) + f"_{HIDDEN_DIM}" + ".pkl",
            ),
        )
        print("Testing-%d..." % t)
        # testing procedure
        predicts = []
        trues = []
        total_loss = 0.0
        total = 0.0
        i = 0
        while i < len(test_data_t):
            batch = get_batch(test_data_t, i, BATCH_SIZE)
            i += BATCH_SIZE
            test1_inputs, test2_inputs, test_labels = batch
            if USE_GPU:
                test_labels = test_labels.cuda()

            model.batch_size = len(test_labels)
            model.hidden = model.init_hidden()
            output = model(test1_inputs, test2_inputs)

            loss = loss_function(output, Variable(test_labels))

            # calc testing acc
            predicted = (output.data > 0.5).cpu().numpy()
            predicts.extend(predicted)
            trues.extend(test_labels.cpu().numpy())
            total += len(test_labels)
            total_loss += loss.item() * len(test_labels)
        if lang == "java":
            weights = [0, 0.005, 0.001, 0.002, 0.010, 0.982]
            p, r, f, _ = precision_recall_fscore_support(
                trues, predicts, average="binary"
            )
            precision += weights[t] * p
            recall += weights[t] * r
            f1 += weights[t] * f
            print("Type-" + str(t) + ": " + str(p) + " " + str(r) + " " + str(f))
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                trues, predicts, average="binary"
            )

    print("Total testing results(P,R,F1):%.3f, %.3f, %.3f" % (precision, recall, f1))

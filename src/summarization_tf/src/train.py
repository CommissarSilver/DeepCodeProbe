import argparse
from utils import (
    read_pickle,
    Datagen_set,
    Datagen_deepcom,
    Datagen_tree,
    Datagen_binary,
    bleu4,
)
from models import Seq2seqModel, CodennModel, ChildsumModel, MultiwayModel, NaryModel
import numpy as np
import os
import torch
from tqdm import tqdm
from joblib import delayed, Parallel
import json

# set torch to use GPU

parser = argparse.ArgumentParser(description="Source Code Generation")

parser.add_argument(
    "-m",
    "--method",
    type=str,
    nargs="?",
    required=False,
    choices=["seq2seq", "deepcom", "codenn", "childsum", "multiway", "nary"],
    default="multiway",
    help="Encoder method",
)
parser.add_argument(
    "-d",
    "--dim",
    type=int,
    nargs="?",
    required=False,
    default=1024,
    help="Representation dimension",
)
parser.add_argument(
    "--embed",
    type=int,
    nargs="?",
    required=False,
    default=256,
    help="Representation dimension",
)
parser.add_argument(
    "--drop",
    type=float,
    nargs="?",
    required=False,
    default=0.5,
    help="Dropout rate",
)
parser.add_argument(
    "-r",
    "--lr",
    type=float,
    nargs="?",
    required=False,
    default=0.001,
    help="Learning rate",
)
parser.add_argument(
    "-b",
    "--batch",
    type=int,
    nargs="?",
    required=False,
    default=128,
    help="Mini batch size",
)
parser.add_argument(
    "-e",
    "--epochs",
    type=int,
    nargs="?",
    required=False,
    default=10,
    help="Epoch number",
)
parser.add_argument(
    "-g",
    "--gpu",
    type=str,
    nargs="?",
    required=False,
    default="0",
    help="What GPU to use",
)
parser.add_argument(
    "-l",
    "--layer",
    type=int,
    nargs="?",
    required=False,
    default=1,
    help="Number of layers",
)
parser.add_argument(
    "--val",
    type=str,
    nargs="?",
    required=False,
    default="BLEU",
    help="Validation method",
)


args = parser.parse_args()

name = args.method + "_dim" + str(args.dim) + "_embed" + str(args.embed)
name = name + "_drop" + str(args.drop)
name = name + "_lr" + str(args.lr) + "_batch" + str(args.batch)
name = (
    name
    + "_epochs"
    + str(args.epochs)
    + "_layer"
    + str(args.layer)
    + "NEW_skip_size100"
)

checkpoint_dir = "./models/" + name


# load data
dataset_path = "/store/travail/vamaj/Leto/src/summarization_tf/dataset"
trn_data = read_pickle(f"{dataset_path}/nl/train.pkl")
vld_data = read_pickle(f"{dataset_path}/nl/valid.pkl")
tst_data = read_pickle(f"{dataset_path}/nl/test.pkl")
code_i2w = read_pickle(f"{dataset_path}/code_i2w.pkl")
code_w2i = read_pickle(f"{dataset_path}/code_w2i.pkl")
nl_i2w = read_pickle(f"{dataset_path}/nl_i2w.pkl")
nl_w2i = read_pickle(f"{dataset_path}/nl_w2i.pkl")

trn_x, trn_y_raw = zip(*sorted(trn_data.items()))
vld_x, vld_y_raw = zip(*sorted(vld_data.items()))
tst_x, tst_y_raw = zip(*sorted(tst_data.items()))

trn_y = [
    [nl_w2i[t] if t in nl_w2i.keys() else nl_w2i["<UNK>"] for t in l] for l in trn_y_raw
]
vld_y = [
    [nl_w2i[t] if t in nl_w2i.keys() else nl_w2i["<UNK>"] for t in l] for l in vld_y_raw
]
tst_y = [
    [nl_w2i[t] if t in nl_w2i.keys() else nl_w2i["<UNK>"] for t in l] for l in tst_y_raw
]


if args.method in ["seq2seq", "deepcom"]:
    Model = Seq2seqModel
elif args.method in ["codenn"]:
    Model = CodennModel
elif args.method in ["childsum"]:
    Model = ChildsumModel
elif args.method in ["multiway"]:
    Model = MultiwayModel
elif args.method in ["nary"]:
    Model = NaryModel


model = Model(
    args.dim,
    args.dim,
    args.dim,
    len(code_w2i),
    len(nl_w2i),
    dropout=args.drop,
    lr=args.lr,
    layer=args.layer,
)
model.to("cuda:" + args.gpu)
epochs = args.epochs
batch_size = args.batch
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
history = {"loss": [], "loss_val": [], "bleu_val": []}

if args.method in ["deepcom"]:
    Datagen = Datagen_deepcom
elif args.method in ["codenn"]:
    Datagen = Datagen_set
elif args.method in ["childsum", "multiway"]:
    Datagen = Datagen_tree
elif args.method in ["nary"]:
    Datagen = Datagen_binary


trn_gen = Datagen(trn_x, trn_y, batch_size, code_w2i, nl_i2w, train=True)
vld_gen = Datagen(vld_x, vld_y, batch_size, code_w2i, nl_i2w, train=False)
tst_gen = Datagen(tst_x, tst_y, batch_size, code_w2i, nl_i2w, train=False)

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
for epoch in range(1, epochs + 1):
    batch_turn = 0
    # train
    loss_tmp = []
    t = tqdm(trn_gen(0))
    for x, y, _, _ in t:
        try:
            model.optimizer.zero_grad()
            loss = model.train_on_batch(x, torch.tensor(y))
            loss_tmp.append(loss.item())
            t.set_description(
                "epoch:{:03d}, loss = {:.6f}".format(epoch, np.mean(loss_tmp))
            )
            batch_turn += 1
            history["loss"].append(np.sum(loss_tmp) / len(t))
            writer.add_scalar("loss", np.sum(loss_tmp) / len(t), epoch)
            if batch_turn % 1899 == 0:
                torch.save(
                    model.state_dict(),
                    f"/store/travail/vamaj/Leto/src/summarization_tf/checkpoints/epoch_{epoch}_batch_{batch_turn}.pth",
                )
        except RuntimeError as e:
            print(e)
            pass
    json.dump(history, open(f"{checkpoint_dir}/train_{epoch}.json", "w"))
    # t = tqdm(vld_gen(0))
    # preds = []
    # trues = []
    # bleus = []
    # for x, y, _, y_raw in t:
    #     res = model.translate(x, nl_i2w, nl_w2i)
    #     preds += res
    #     trues += [s[1:-1] for s in y_raw]
    #     bleus += [bleu4(tt, p) for tt, p in zip(trues, preds)]
    #     t.set_description("epoch:{:03d}, bleu_val = {:.6f}".format(epoch, np.mean(bleus)))
    # history["bleu_val"].append(np.mean(bleus))
    # writer.add_scalar("bleu_val", np.mean(bleus), epoch)
# validate loss
loss_tmp = []
t = tqdm(vld_gen(0))
for x, y, _, _ in t:
    loss = model.evaluate_on_batch(x, y)
    loss_tmp.append(loss)
    t.set_description(
        "epoch:{:03d}, loss_val = {:.6f}".format(epoch, np.mean(loss_tmp))
    )
history["loss_val"].append(np.sum(loss_tmp) / len(t))
writer.add_scalar("loss_val", np.sum(loss_tmp) / len(t), epoch)

# checkpoint
torch.save(
    model.state_dict(),
    f"/store/travail/vamaj/Leto/src/summarization_tf/checkpoints/epoch_{epoch}_{args.dim}.pth",
)
if history["bleu_val"][-1] == max(history["bleu_val"]):
    best_model_path = f"checkpoints/epoch_{epoch}.pth"
    print(f"Now best model is at {best_model_path}")

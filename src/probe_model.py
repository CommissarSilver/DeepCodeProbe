import os, logging, sys
import numpy as np
import torch

from gensim.models.word2vec import Word2Vec
from ast_probe.probe import (
    probe as ast_probe,
    loss as ast_probe_loss,
    utils as ast_probe_utils,
)

from ast_nn.src.model import BatchProgramCC as model_under_test
from ast_nn.src.data_pipeline import process_input as ast_nn_process_input


word2vec = Word2Vec.load("src/ast_nn/node_w2v_128").wv
MAX_TOKENS = word2vec.syn0.shape[0]
EMBEDDING_DIM = word2vec.syn0.shape[1]
HIDDEN_DIM = 100
ENCODE_DIM = 128
LABELS = 1
EPOCHS = 5
BATCH_SIZE = 32
USE_GPU = False
embeddings = np.zeros((MAX_TOKENS + 1, EMBEDDING_DIM), dtype="float32")
model = model_under_test(
    EMBEDDING_DIM,
    HIDDEN_DIM,
    MAX_TOKENS + 1,
    ENCODE_DIM,
    LABELS,
    BATCH_SIZE,
    USE_GPU,
    embeddings,
)
model.load_state_dict(torch.load("src/ast_nn/trained_models/astnn_model_c.pkl"))
prober = ast_probe.ParserProbe(100, 100, 100, 100)

import os

import numpy as np
from gensim.models.word2vec import Word2Vec

# from ast_nn.src.data_pipeline import process_input
from ast_nn.src.model import BatchProgramCC

language = "java"
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

model = BatchProgramCC(
    embedding_dim=word2vec.syn0.shape[1],
    hidden_dim=100,
    vocab_size=word2vec.syn0.shape[0] + 1,
    encode_dim=128,
    label_size=1,
    batch_size=2,
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
# model.encode(
#     [
#         "public String getValue(String key) {\n        KeyValue kv = getKV(key);\n        return kv == null ? null : kv.getValue();\n    }"
#     ]
# )
model.encode(
    [
        "public String getValue(String key) {\n        KeyValue kv = getKV(key);\n        return kv == null ? null : kv.getValue();\n    }",
        "protected DocPath pathString(ClassDoc cd, DocPath name) {\n        return pathString(cd.containingPackage(), name);\n    }",
    ]
)
[
    'System.out.println("Hello World!");',
    'public static void main(String[] args) { System.out.println("Hello World!"); }',
    'public class Test { public static void main(String[] args) { System.out.println("Hello World!"); } }',
]

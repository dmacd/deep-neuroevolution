from mxnet import gluon
from mxnet import nd
from mxnet.contrib import text
import collections
from typing import Set


def get_embedding_for_text(text_data: str) -> text.embedding.FastText:
    counter = text.utils.count_tokens_from_str(text_data)
    vocab = text.vocab.Vocabulary(counter)
    embedding = text.embedding.create('fasttext',
                                      pretrained_file_name='wiki.simple.vec',
                                      vocabulary=vocab)
    return embedding


def layer_for_embedding(embedding: text.embedding.FastText) -> gluon.Block:
    layer = gluon.nn.Embedding(len(embedding), embedding.vec_len)
    layer.initialize()
    layer.weight.set_data(embedding.idx_to_vec)
    return layer


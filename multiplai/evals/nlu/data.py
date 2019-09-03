import os
import json
import multiplai.base as base

from mxnet import gluon
from mxnet import nd
from mxnet.contrib import text
import collections
from typing import Set, Tuple, Dict, List

TRAIN_DATA_PATH = 'multiplai/evals/nlu/nlu-benchmark/2017-06-custom-intent-engines/GetWeather/train_GetWeather_full.json'

def load_train_data(path=TRAIN_DATA_PATH):
    with open(os.path.join(base.repo_root(), TRAIN_DATA_PATH), 'r') as f:
        train_data = json.load(f)
    return train_data

def get_train_data_text_and_entities(train_data) -> Tuple[str, Set[str]]:

    all_entities = set()
    all_text = ""

    for k, v in train_data.items():
        for example in v:
            for chunk in example['data']:
                all_text += (chunk['text'])
                if 'entity' in chunk:
                    #                 print(chunk['entity'])
                    all_entities.add(chunk['entity'])
            all_text += ('\n')

    # print('all_entities =', all_entities)
    max_labels = len(all_entities)
    # print('max_labels =', max_labels)
    # print('all_text = """%s"""' % all_text)

    all_text = preprocess_text(all_text)
    all_text += " comma"
    all_text += " period"

    return all_text, all_entities

def preprocess_text(s):
    s = s.replace(',', ' , ')
    s = s.replace('.', ' . ')
    s = s.replace(':', ' : ')
    s = s.replace('/', ' / ')
    s = s.replace('?', ' ? ')
    return s

def get_word_to_idx(embedding: text.embedding.FastText) -> Dict[str, int]:
    return embedding.token_to_idx

def get_entity_to_idx(all_entities: Set[str]) -> Dict[str, int]:
    return {ent: idx for idx, ent in enumerate(['<none>'] + sorted(list(
        all_entities)))}

def example_to_training_pair(example, word_to_idx, entity_to_idx):
    pair = []
    for chunk in example['data']:
        if 'entity' in chunk:
            label = entity_to_idx[chunk['entity']]
        else:
            label = 0

        for word in preprocess_text(chunk['text']).strip().split(' '):
            if word not in word_to_idx:
                # TODO: handle
                # - commas
                # - uppercase words in proper nouns
                # - place names
                #  - compound common nouns
                if word.lower() in word_to_idx:
                    pair.append((word_to_idx[word.lower()], label))
                # elif word == ',':
                #     pair.append((word_to_idx['comma'], label))
                #     print(example)
                #     print(pair)
                # elif word == '.':
                #     pair.append((word_to_idx['period'], label))
                #     print(example)
                #     print(pair)
                elif word == '':
                    pass
                else:
                    print("***************************************")
                    print("chunk:", chunk)
                    print("chunk text split:", chunk['text'].strip().split(' '))
                    print("missing word: '%s'" % word)
                    print('example:', example)
                    return []
            else:
                pair.append((word_to_idx[word], label))

    return pair

def transpose_training_pair(pair: List[Tuple[int,int]]) -> Tuple[nd.array,
                                                                  nd.array]:

   return nd.array(tuple(zip(*pair)),dtype='int')
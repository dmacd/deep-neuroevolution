import pytest
from multiplai.evals.nlu import data
from multiplai.evals.nlu import embedding as emb
from multiplai.evals.nlu.data import WordIntentPair
from multiplai.evals.nlu.gym.env import nlu_benchmark_env

@pytest.fixture(scope='session')
def train_data():
    return data.load_train_data()


@pytest.fixture(scope='session')
def _all_text_all_entities(train_data):
  return data.get_train_data_text_and_entities(train_data)
@pytest.fixture(scope='session')
def embedding(train_data, _all_text_all_entities):
    all_text, all_entities = _all_text_all_entities
    return emb.get_embedding_for_text(all_text)

@pytest.fixture(scope='session')
def all_entities(train_data, _all_text_all_entities):
    all_text, all_entities = _all_text_all_entities
    return all_entities

@pytest.fixture(scope='session')
def train_pairs(train_data, embedding, all_entities):

  training_pairs = data.pairs_from_data(train_data['GetWeather'],
                                        all_entities=all_entities,
                                        embedding=embedding)
  return training_pairs


@pytest.fixture
def single_example_env(train_data, embedding, all_entities):
  pairs = WordIntentPair.pairs_from_data(train_data['GetWeather'],
                                         all_entities=all_entities,
                                         embedding=embedding)

  n_words = len(embedding.token_to_idx.keys())
  n_labels = len(all_entities)

  env = nlu_benchmark_env.SingleExampleEnv(pairs=pairs,
                       n_symbols=n_words,
                       n_classes=n_labels)
  return env
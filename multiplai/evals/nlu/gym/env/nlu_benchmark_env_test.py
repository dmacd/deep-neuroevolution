import pytest
from multiplai.evals.nlu.data import WordIntentPair
from multiplai.evals.nlu.gym.env import nlu_benchmark_env


# @pytest.fixture
# def single_example_env(train_data, embedding, all_entities):
#   pairs = WordIntentPair.pairs_from_data(train_data['GetWeather'],
#                                          all_entities=all_entities,
#                                          embedding=embedding)
#
#   n_words = len(embedding.token_to_idx.keys())
#   n_labels = len(all_entities)
#
#   env = nlu_benchmark_env.SingleExampleEnv(pairs=pairs,
#                        n_symbols=n_words,
#                        n_classes=n_labels)
#   return env


def test_single_example_env(single_example_env):
  env = single_example_env
  # test init

  # test dumb stuff
  env.seed(0)
  obs = env.reset()
  assert obs[0] == 10
  obs, reward, done, info = env.step(action=0)

  # print(env._pairs[env._pair_idx])
  # print(obs, reward, done, info)

  assert obs[0] == 12
  assert reward == 1
  assert not done

def test_single_example_env_terminates(single_example_env):

  env = single_example_env
  env.seed(0)
  obs = env.reset()

  # check episodes terminate

  for _ in range(100):
    obs, reward, done, info = env.step(action=0)
    if done:
      break
  else:
    assert False, "Episode never finished!"

  assert obs is None
  assert reward == 1
  assert done


def test_single_example_env_randomness(single_example_env):
  env = single_example_env
  # test randomness
  env.seed(1)
  obs = env.reset()
  assert obs[0] == 61

  obs, reward, done, info = env.step(action=0)

  # print("check randomness")
  # print(env._pairs[env._pair_idx])
  # print(obs, reward, done, info)

  assert obs[0] == 80
  assert reward == 1
  assert not done

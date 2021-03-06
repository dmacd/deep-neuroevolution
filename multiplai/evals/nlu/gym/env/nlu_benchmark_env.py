import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import gym
from gym import spaces
from multiplai.evals.nlu.data import WordIntentPair
from typing import List, Set, Tuple, Optional, Dict
from mxnet import ndarray as nd
from multiplai.evals.nlu import data
from multiplai.evals.nlu import embedding as emb
from multiplai.evals.nlu.data import WordIntentPair
from multiplai.evals.nlu.gym.env import nlu_benchmark_env


# start with single instance classification task
# todo: add env_args to config and init code

ActionT = int
RewardT = float
ObservationT = Optional[nd.array]
DebugInfoT = Dict

class SingleExampleEnvBase(gym.Env):
  """Env that reinforces correct classification of a single nlu training
  example"""
  metadata = {'render.modes': ['human']}

  def __init__(self, pairs: List[WordIntentPair],
               n_symbols: int,
               n_classes: int):
    super(SingleExampleEnvBase, self).__init__()

    self.observation_space = spaces.Discrete(n_symbols)
    self.action_space = spaces.Discrete(n_classes)

    # self.observation_space = spaces.Box(low=0, high=255, shape=
    #                 (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
    self._symbol_idx = 0
    self._pair_idx = 0
    self._pairs = pairs
    self._n_classes = n_classes
    self._n_symbols = n_symbols

    # debugging
    self._last_action: Optional[ActionT] = None
    self._last_reward: Optional[float] = None
    self._last_observation: Optional[ObservationT] = None

    self.seed(0)

  def seed(self, seed=None):
    self._rng, _seed = seeding.np_random(seed)
    return _seed

  def step(self, action: ActionT) -> Tuple[ObservationT, RewardT, bool,
                                           DebugInfoT]:
    self._last_action = action
    correct_class = self._pairs[self._pair_idx].class_indices[self._symbol_idx]

    reward = 0
    if action == correct_class:
      reward += 1
      if action != 0:
        reward += 3   # boost for exploration?
    else:
      reward += -1

    self._symbol_idx += 1

    if self._symbol_idx >= len(self._pairs[self._pair_idx].word_indices):
      done = True
      obs = None
    else:
      done = False
      obs = self._pairs[self._pair_idx].word_indices[self._symbol_idx]

    self._last_observation = obs
    self._last_reward = reward
    return obs, reward, done, {}

  def reset(self) -> ObservationT:
    # Reset the state of the environment to an initial state
    self._symbol_idx = 0
    self._pair_idx = self._rng.randint(0, len(self._pairs))
    obs = self._pairs[self._pair_idx].word_indices[self._symbol_idx]
    self._last_observation = obs
    return obs

  def render(self, mode='human', close=False):
    # Render the environment to the screen

    print("-------------------------------------------------------------------")
    print("%s.render()" % self.__class__.__name__)
    print(self._pairs[self._pair_idx])
    print("symbol_idx:", self._symbol_idx)
    print("last_reward:" , self._last_reward)
    print("last_action:", self._last_action)
    print("last_observation:", self._last_observation)
    print("-------------------------------------------------------------------")


  def close(self):
    pass


################################################################################
## concrete environments

class SingleExample_GetWeather(SingleExampleEnvBase):
  def __init__(self):

    train_data = data.load_train_data()

    all_text, all_entities = data.get_train_data_text_and_entities(
      train_data)

    embedding = emb.get_embedding_for_text(all_text)

    pairs = WordIntentPair.pairs_from_data(train_data['GetWeather'],
                                           all_entities=all_entities,
                                           embedding=embedding)

    n_words = len(embedding.token_to_idx.keys())
    n_labels = len(all_entities)

    super().__init__(pairs=pairs,
                     n_symbols=n_words,
                     n_classes=n_labels)



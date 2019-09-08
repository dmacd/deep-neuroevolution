import gym
from gym import error, spaces, utils
from gym.utils import seeding

import gym
from gym import spaces
from multiplai.evals.nlu.data import WordIntentPair
from typing import List, Set, Tuple

# start with single instance classification task
# todo: add env_args to config and init code

ActionT = int
ObservationT = int
DebugInfoT = dict


class SingleExampleEnv(gym.Env):
  """Env that reinforces correct classification of a single nlu training
  example"""
  metadata = {'render.modes': ['human']}

  def __init__(self, pairs: List[WordIntentPair],
               n_symbols: int,
               n_classes: int):
    super(SingleExampleEnv, self).__init__()

    self.observation_space = spaces.Discrete(n_symbols)
    self.action_space = spaces.Discrete(n_classes)

    # self.observation_space = spaces.Box(low=0, high=255, shape=
    #                 (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)
    self._symbol_idx = 0
    self._pair_idx = 0
    self._pairs = pairs
    self._n_classes = n_classes
    self._n_symbols = n_symbols

  def step(self, action: int) -> Tuple[ActionT, ObservationT, bool, DebugInfoT]:
    obs = self._pairs[self._pair_idx].word_indices[self._symbol_idx]

    correct_class = self._pairs[self._pair_idx].class_indices[self._symbol_idx]
    if action == correct_class:
      reward = 1
    else:
      reward = -1

    self._symbol_idx += 1

    if self._symbol_idx >= len(self._pairs[self._pair_idx].word_indices):
      done = True
    else:
      done = False

    return obs, reward, done, {}

  def reset(self):
    # Reset the state of the environment to an initial state
    ...

  def render(self, mode='human', close=False):
    # Render the environment to the screen
    ...

from typing import List, Optional, Tuple, Callable, Union
import mxnet as mx
import mxnet.ndarray as nd
import numpy as np
from tqdm import tqdm
from multiplai.structural_plasticity.clique import clique as c

DEFAULT_RNG = np.random.RandomState(0)


# generate M distinct random patterns with N clusters and C units per cluster
# the maximum possible number of such patterns is  C^N


def generate_random_patterns(n_clusters, cluster_size, total_patterns,
                             rng: np.random.RandomState = DEFAULT_RNG) \
    -> nd.NDArray:
  """Patterns uniformly sampled from the space of possible patterns of this
  size"""

  # encode each cluster as a number.
  # pick n independent random numbers to select the active neuron in that
  # cluster

  patterns = rng.randint(0, cluster_size, size=(total_patterns, n_clusters))
  # patterns = nd.random_randint(0, cluster_size, shape=(total_patterns,
  #                                                     n_clusters))
  return nd.array(patterns)


def patterns_to_onehot(patterns: nd.NDArray, cluster_size: int) -> nd.NDArray:
  """Patterns array to array of one-hot inputs

    :param patterns: shape (n_patterns, n_clusters)
    :return: onehotted patterns of shape (n_patterns, n_clusters*cluster_size)
    """
  return nd.one_hot(patterns, cluster_size,
                    dtype=patterns.dtype).reshape(patterns.shape[0], -1)


def unique_patterns(patterns: nd.NDArray) -> nd.NDArray:
  """Given patterns of shape (n_patterns, x) return an array of same shape
  but with only the unique patterns """

  import numpy as np
  return nd.array(np.unique(patterns.asnumpy(), axis=0))


def generate_unique_random_patterns(n_clusters, cluster_size, total_patterns,
                                    rng: np.random.RandomState = DEFAULT_RNG
                                    ):
  """Returns a set of unique random patterns """

  n_patterns = total_patterns
  while True:
    patterns = generate_random_patterns(n_clusters, cluster_size,
                                        n_patterns, rng)
    patterns = unique_patterns(patterns)
    if patterns.shape[0] >= total_patterns:
      return patterns[0:total_patterns, :]

    n_patterns = int(n_patterns * 1.3)
    print("generate_unique_random_patterns(): "
          "not enough unique patterns, trying again with %d" % n_patterns)

    if n_patterns > 10 * total_patterns:
      assert False, "Too hard or impossible to generate unique " \
                    "patterns under these conditions"


def concat_patterns(patterns: List[nd.NDArray]) -> nd.NDArray:
  return nd.concat(*patterns, dim=1)


def generate_unique_prefix_random_patterns(n_clusters, cluster_size,
                                           total_patterns,
                                           n_prefix_clusters,
                                           rng: np.random.RandomState =
                                           DEFAULT_RNG) -> nd.NDArray:
  """Generates random unique patterns where additionally the first
  n_prefix_clusters's portion of the patterns are unique"""

  prefix_patterns = generate_unique_random_patterns(n_prefix_clusters,
                                                    cluster_size,
                                                    total_patterns,
                                                    rng)

  assert unique_patterns(prefix_patterns).shape[0] == total_patterns, \
    "prefixes werent unique!"

  suffix_patterns = generate_random_patterns(n_clusters - n_prefix_clusters,
                                             cluster_size, total_patterns,
                                             rng)

  return concat_patterns([prefix_patterns, suffix_patterns])


# procedure for testing recall of distinct patterns:

# - generate two sets of unique patterns and concat them. the first set,
# forming a unique prefix of the concatenated whole, can serve as the lookup


class AssociativeMemoryInterface:
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def store(self, pattern: nd.NDArray):
    raise NotImplementedError('abstract base class')

  def recall(self, partial_pattern):
    raise NotImplementedError('abstract base class')


class AssociativeLayer(c.Layer, AssociativeMemoryInterface):
  store_iterations: int = 7
  recall_iterations: int = 7
  clear_iterations: int = 5
  input_strength: float = 3

  def __init__(self, *args, **kwargs):
    super(AssociativeLayer, self).__init__(*args, **kwargs)

  def store(self, pattern: nd.NDArray):

    self.reset()
    for _ in range(self.store_iterations):
      self.step(pattern*self.input_strength)

    # experiment: let intermediate weights decay between patterns so we dont
    # accidentally pick up spurious stuff
    self.reset()
    # for _ in range(self.clear_iterations):
    #   self.step(0*pattern)

  def recall(self, partial_pattern):
    """Attempts to complete a partial pattern. Weights are NOT updated"""

    self.reset()
    activations = partial_pattern
    for _ in range(self.recall_iterations):
      activations = self.step(partial_pattern, update_weights=False)
      # partial_pattern = 0*partial_pattern

    return activations


# Q: multilayer setup any different than a chain of overlapping k->v mappings?
#  i.e. do we need more of the dynamics to get something that usefully backprops
#  and forms deep concepts than just running a simple forward and reverse
#  lookup?
#  think the answer needs to be YES unless everyone has just overlooked
#  something really basic!?


# absract store and retrieve functions

def store_patterns(alayer, patterns, cluster_size):
  for p in tqdm(patterns, unit='patterns', desc='store_patterns'):
    # print(p)
    onehot = patterns_to_onehot(p.expand_dims(0), cluster_size=cluster_size)
    # print(onehot)
    alayer.store(onehot[0, :])

    nd.waitall()


def check_recall(alayer, patterns, drop_fraction, cluster_size):
  errors = []
  for p in tqdm(patterns, unit='patterns', desc='check_recall'):
    # print("p.shape", p.shape)
    onehot = patterns_to_onehot(p.expand_dims(0), cluster_size=cluster_size)
    # print("onehot.shape", onehot.shape)
    keep_bits = int(onehot.shape[1] * (1-drop_fraction))
    onehot_half = onehot.copy()
    onehot_half[0, keep_bits:] = 0
    # print("keep bits:", keep_bits)
    activations = alayer.recall(onehot_half[0, :])

    # print("target     :", onehot[0].asnumpy())
    # print("activations:", activations.asnumpy())
    # print("recall active bits:", sum(sum(activations)).asnumpy()[0])
    # print("recall error bits:", sum(sum(onehot - activations)).asnumpy()[0])

    if sum(sum(nd.abs(onehot - activations))).asnumpy()[0] > 0:
      errors.append((onehot, onehot_half, activations))

  return errors


def test_store_recall(n_patterns, n_clusters, cluster_size, drop_fraction,
                      ctx=mx.cpu(),
                      dtype='float32'
                      ):
  alayer = AssociativeLayer(n_clusters=n_clusters,
                            cluster_size=cluster_size,
                            ctx=ctx,
                            dtype=dtype)


  n_prefix_clusters = min(n_clusters, int(n_clusters*(1 - drop_fraction)))
  # print("prefix clusters, bits:", n_prefix_clusters,
  #       n_prefix_clusters*cluster_size)

  patterns = generate_unique_prefix_random_patterns(
    n_clusters=n_clusters,
    cluster_size=cluster_size,
    # paranoid: ensure recall errors are not due to conflicting prefixes
    n_prefix_clusters=n_prefix_clusters,
    total_patterns=n_patterns).as_in_context(ctx).astype(dtype)

  store_patterns(alayer, patterns, cluster_size)
  nd.waitall()
  errors = check_recall(alayer,
                        patterns,
                        drop_fraction=drop_fraction,
                        cluster_size=cluster_size)
  nonzero_weights = nd.sum(alayer.weights > 0.5)[0].asscalar()

  stats = dict(total_errors=len(errors),
               error_rate=len(errors) / n_patterns,
               nonzero_weights=nonzero_weights,
               density=nonzero_weights / np.prod(alayer.weights.shape))
  return stats, errors, alayer

# routines for plotting capacity vs noise curves

import networkx as nx

def unit_pos(k, n_clusters, cluster_size):
  cluster_num = k // cluster_size

  total_units = n_clusters * cluster_size

  # r = 10 + (cluster_num % 2)
  r = 10
  #     theta = 2*np.pi*k/total_units
  #     theta = 2*np.pi*(cluster_num+(k%cluster_size))/(total_units +cluster_size)

  within_cluster = (k % cluster_size) / (5 * cluster_size)
  cluster_angle = 2 * np.pi * (cluster_num) / n_clusters

  theta = cluster_angle + 2 * np.pi * within_cluster

  return (r * np.cos(theta), r * np.sin(theta))

def plot_weight_graph(weights, cluster_size, activations):
  G = nx.DiGraph()

  total_units = weights.shape[0]
  n_clusters = total_units // cluster_size

  for k in range(total_units):
    G.add_node(k, pos=unit_pos(k, n_clusters, cluster_size))

  for i in range(total_units):
    for j in range(total_units):
      if weights[i, j] > 0.5:
        G.add_edge(i, j, weight=weights[i, j])

  pos = nx.get_node_attributes(G, 'pos')

  nx.draw_networkx(G, pos, node_color=activations)

  nx.draw_networkx_edges(G, pos=pos)

# tools to test catastrophic forgetting


# test: very large networks millions, billions of units
#   - approaching limit of practical size for GPUs
#   - do the properties hold even at larger scales


################################################################################
# ultimately: explore what happens in overlapping clique nets!
#  - can i get something like backprop wrt to deep concept hierarchies to
#  emerge?


# what is the simplest test case?
#   - online binary function learning?
#
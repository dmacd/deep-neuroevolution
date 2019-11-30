
import pytest
import numpy as np
import mxnet as mx
import mxnet.ndarray as nd
from multiplai.structural_plasticity.clique import clique as c
from multiplai.structural_plasticity.clique import eval as e




def test_unique_random_patterns():

  with pytest.raises(AssertionError):
    e.generate_unique_random_patterns(1, 4, 5)

  patterns = e.generate_unique_random_patterns(2, 4, 10)

  assert np.unique(patterns.asnumpy(), axis=0).shape[0] == 10


def test_unique_prefix_random_patterns():

  patterns = e.generate_unique_prefix_random_patterns(n_clusters=10,
                                                      cluster_size=4,
                                                      n_prefix_clusters=4,
                                                      total_patterns=256
                                                      )
  # patterns are unique
  assert e.unique_patterns(patterns).shape[0] == 256

  # prefixes are unique
  assert e.unique_patterns(patterns[:,0:4]).shape[0] == 256

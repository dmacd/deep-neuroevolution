import pytest
from mxnet import ndarray as nd
import numpy as np
import mxnet as mx
import multiplai.structural_plasticity.clique.clique as c
from multiplai.structural_plasticity.assembly_test import allclose


def test_cluster_wta():
  activations = nd.array([1, 2, 3,
                          4, 2, 1,
                          5, 6, 0])

  wta = c.cluster_wta(3, activations)

  assert wta.shape == activations.shape

  assert np.allclose(wta.asnumpy(),
                     np.array([0, 0, 1,
                               1, 0, 0,
                               0, 1, 0]))


def test_cluster_wta_with_threshold():
  activations = nd.array([0, 0, .05])
  wta = c.cluster_wta(3, activations, minimum_activity_threshold=.1)
  assert allclose(wta, nd.zeros_like(activations))

  activations = nd.array([0, 0, .05])
  wta = c.cluster_wta(3, activations, minimum_activity_threshold=.01)
  assert allclose(wta, [0, 0, 1])


def test_weight_sigmoid():
  assert c.weight_sigmoid(nd.array([0]))[0] == 0
  assert c.weight_sigmoid(nd.array([1]))[0] == 1


def test_outer():
  x = nd.array([1, 2, 3])
  y = nd.array([1, 2, 3])

  assert allclose(c.outer(x, y),
                  nd.array(
                    [[1., 2., 3.],
                     [2., 4., 6.],
                     [3., 6., 9.], ]))


def test_step():
  # empirically takes 5 steps before
  # getting above .5
  epsilon = .2
  n_clusters = 2
  cluster_size = 3
  total_neurons = n_clusters * cluster_size
  weights = nd.ones((total_neurons, total_neurons)) * .25

  activations = nd.zeros(total_neurons)

  inputs = nd.zeros_like(activations)

  # step with zero activations...

  for k in range(10):
    # print("before step %d" % k)
    # print("weights:", weights)
    # print("activations:", activations)
    weights, activations = c.step(weights, activations, inputs, epsilon,
                                  cluster_size)
    assert allclose(activations, 0)

  # should lead to zero activations
  # and decaying weights??
  # TODO: (later, maybe) decaying weights in presence of zero activation are
  #  part of the original formalism, but we may later add a term that lets
  #  weights remain if there is no activity in a cluster

  # check that weights decay when too small
  assert allclose(weights, 0)

  weights: nd.NDArray = nd.zeros_like(weights)

  weights[0, 2] = 1
  weights[1, 3] = 1

  for k in range(10):
    weights, activations = c.step(weights, activations, inputs, epsilon,
                                  cluster_size)
    assert allclose(activations, 0)

  assert weights[0, 2] == 1
  assert weights[1, 3] == 1

  # check than inputs influence activations

  inputs = nd.array([0, 0, 1, 0, 1, 0])
  for k in range(10):
    # print("before step %d" % k)
    # print("weights:", weights)
    # print("activations:", activations)
    #
    weights, activations = c.step(weights, activations, inputs,
                                  epsilon,
                                  cluster_size)
    assert allclose(activations, inputs)

  # check that weights grow with recurrent activation
  # check that new synapses do form

  # todo cluster index func
  assert weights[4, 2] == 1
  assert weights[2, 4] == 1

  # independent check: recurrent weights
  assert weights[4, 4] == 1
  assert weights[2, 2] == 1

  # original weights didnt decay
  assert weights[0, 2] == 1
  assert weights[1, 3] == 1

  # check that activations propagate according to weights
  # message recovery

  # TODO: check that recovery happens after single/short partial presentation
  #  as well

  inputs = nd.array([0, 0, 1, 0, 0, 0])
  activations = nd.zeros_like(activations)
  for k in range(10):
    # print("before step %d" % k)
    # print("weights:", weights)
    # print("activations:", activations)
    weights, activations = c.step(weights, activations, inputs,
                                  epsilon,
                                  cluster_size)

  assert allclose(activations, [0, 0, 1, 0, 1, 0])

  # TODO: IMPORTANT(??): verify recurrent weights are suppressed

  # check behavior when two inputs in a cluster are active (is this realistic?)

  inputs = nd.array([0, 1, 1, 0, 0, 0])
  activations = nd.zeros_like(activations)
  for k in range(10):
    print("before step %d" % k)
    print("weights:", weights)
    print("activations:", activations)

    weights, activations = c.step(weights, activations, inputs,
                                  epsilon,
                                  cluster_size)
    # assert allclose(activations, inputs)

  try:
    assert allclose(activations, [0, 0, 1, 0, 1, 0])
  except AssertionError as ae:
    print("WARNING: WTA is undefined with equal activations in same cluster ")

  # next steps:
  # - check larger networks for storage and retrieval properties
  # - store many random patterns
  # - grade recover error
  # - plot vs size/topology
  # - add noise

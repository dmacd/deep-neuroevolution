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

  assert np.allclose(wta.asnumpy(),
                     np.array([0, 0, 1,
                               1, 0, 0,
                               0, 1, 0]))


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

  n_clusters = 2
  cluster_size = 3
  total_neurons = n_clusters*cluster_size
  weights = nd.ones((total_neurons,total_neurons)) * .25

  activations = nd.zeros(total_neurons)

  inputs = nd.zeros_like(activations)

  # step with zero activations

  for k in range(10):
    print("before step %d" % k)
    print("weights:", weights)
    print("activations:", activations)
    weights, activations = c.step(weights, activations, inputs, .1,
                                  cluster_size)

  # TODO:  network thresholds WTA so that zero activation doesnt lead
  #  to spurious learning
  # next step ^^
  # - then write test cases around it

  
  # check than inputs influence activations

  # check that activations propagate according to weights


  # check that weights decay when too small


  # check that weights grow with recurrent activation


  # check effects of input


  # TODO: IMPORTANT: verify recurrent weights are suppressed
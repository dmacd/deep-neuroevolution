"""Implementation of "Robust Associative Memories Naturally Occuring From
Recurrent Hebbian Networks Under Noise"
https://arxiv.org/abs/1709.08367v1


"""
from mxnet import ndarray as nd
import numpy as np
import mxnet as mx


# V ~ activation
#     tensor layout ( total_neurons_per_layer )
#

# HUGE QUESTION: can weights be nonzero between cluster elements?
# probably not since they would decay immediately during competition
# but is it worth storing the weights more compactly? if there are many
# clusters, probably not since only 1/n_clusters worth of weights per
# cluster will be zero

# total weights: (n_clusters*cluster_size)**2

# weights per neuron: n_clusters*cluster_size
# zero weights per neuron: cluster_size
# total zero weights: n_clusters*n_cluster_size i.e. sqrt(total_weights)


def cluster_wta(cluster_size, activations):
  # reshape
  n_clusters = activations.shape[0] // cluster_size
  a = activations.reshape((n_clusters, cluster_size))

  # okay
  return nd.topk(nd.array(a), ret_typ='mask').reshape((-1,))




class Layer:

  weights: nd.NDArray
  activations: nd.NDArray
  epsilon: float
  p_insert: float
  p_erase: float
  cluster_size: int
  n_clusters: int





def weight_sigmoid(x):
  # compensate for horrible numerical bug in ndarray where tan(-pi/2) -> +inf
  x = nd.clip(x, 0.0001, .9999)
  return .5 + .5 * nd.tanh(nd.tan(np.pi * x - np.pi / 2))

def outer(x, y):

  return nd.linalg.gemm2(x.expand_dims(0),
                         y.expand_dims(0), transpose_a=True)

def step(weights, activations, inputs, epsilon, cluster_size):


  activations = nd.dot(weights, activations) + inputs
  activations = cluster_wta(cluster_size, activations)

  weights = weight_sigmoid(epsilon * outer(activations, activations)
                           + weights)

  # TODO: exclude recurrent self-weights properly!

  # TODO: add ins/erase noise

  return weights, activations
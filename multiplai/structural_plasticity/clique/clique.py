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


def cluster_wta(cluster_size, activations, minimum_activity_threshold=0.01):
  # reshape
  n_clusters = activations.shape[0] // cluster_size
  a = activations.reshape((n_clusters, cluster_size))

  # okay
  mask = nd.topk(nd.array(a), ret_typ='mask')
  thresholded = nd.where(a > minimum_activity_threshold,
                         nd.ones_like(a),
                         nd.zeros_like(a))
  masked = nd.multiply(mask, thresholded)
  return masked.reshape((-1,))


class Layer:
  weights: nd.NDArray
  activations: nd.NDArray
  epsilon: float
  p_insert: float
  p_erase: float
  cluster_size: int
  n_clusters: int

  def __init__(self, n_clusters, cluster_size,
               epsilon=.18, p_insert=0.05, p_erase=.2,
               ctx=mx.cpu(),
               dtype='float32'):
    self.n_clusters = n_clusters
    self.cluster_size = cluster_size
    self.epsilon = epsilon
    self.p_insert = p_insert
    self.p_erase = p_erase
    self.ctx = ctx

    total_neurons = self.n_clusters*self.cluster_size

    self.weights = nd.zeros((total_neurons, total_neurons), ctx=ctx,
                            dtype=dtype)
    self.activations = nd.zeros(total_neurons, ctx=ctx, dtype=dtype)

  def step(self, inputs: nd.NDArray, update_weights=True) -> nd.NDArray:
    self.weights, self.activations = step(self.weights, self.activations,
                                          inputs, self.epsilon,
                                          self.cluster_size,
                                          update_weights=update_weights)
    return self.activations

  def reset(self):
    """Zeros previous activations"""
    self.activations = self.activations*0






def weight_sigmoid_tan(x):
  # compensate for horrible numerical bug in ndarray where tan(-pi/2) -> +inf
  assert False, "FUCK!!!"
  x = nd.clip(x, 0.0001, .9999)
  return .5 + .5 * nd.tanh(nd.tan(np.pi * x - np.pi / 2))

def weight_sigmoid_hard(x):
  return nd.hard_sigmoid(x-.5, alpha=1)

weight_sigmoid = weight_sigmoid_hard

def outer(x, y):
  return nd.linalg.gemm2(x.expand_dims(0),
                         y.expand_dims(0), transpose_a=True)


def step(weights, activations, inputs, epsilon, cluster_size,
         update_weights=True):
  assert inputs.shape == activations.shape, "inputs and activation shapes " \
                                            "dont match!"

  # print("step(): activations:", activations)
  activations = nd.dot(weights, activations) + inputs


  # if update_weights:  # DEBUG: check if wta is picking wrongly

  # print("step(): activations after W*a:", activations)
  activations = cluster_wta(cluster_size, activations)
  # print("step(): activations after wta:", activations)
  assert nd.sum(activations) <= activations.shape[0]//cluster_size

  if update_weights:
    weights = weight_sigmoid(epsilon * outer(activations, activations)
                             + weights)
    # experimental: force zero self-weights
    # weights = nd.multiply(weights, 1-nd.eye(weights.shape[0]))

  # TODO: exclude recurrent self-weights properly!

  # TODO: add ins/erase noise

  return weights, activations

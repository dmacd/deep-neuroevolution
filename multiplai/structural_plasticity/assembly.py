from mxnet import ndarray as nd
import numpy as np
from dataclasses import dataclass
from typing import Callable

@dataclass
class SynapseProperties:
  p_g = 0.1

  # probs named action-given-learning signal (e|s, c|s, d|s in the paper

  p_elim_1 = 0
  p_elim_0 = 0.01

  p_cons_1 = 0.97
  p_cons_0 = 0.03

  p_deco_0 = 0.8
  p_deco_1 = 0.2

  def __post_init__(self):
    # for now make sure these two add to 1
    assert np.isclose(1 - self.p_cons_1, self.p_cons_0)
    assert np.isclose(1 - self.p_deco_0, self.p_deco_0)


# @dataclass
# class AssemblyProperties:

#     k = 3
#     n = 10

#     # zip net rule parameter to set fraction of neuron's synapses that should be potentiated
#     p_1 = 0

#     def __post_init(self):
#         self.p_1 = self.k / self.n

# balance / homeostatic constants

# ?? desired fraction of each neuron's synapses to be potentiated
#  ... how the f do we maintain this? does this drive the inhibition somehow?
# p_1 = 0.1 ???

# some definitions from the paper

# P       - anatomical connectivity: fraction of neuron pairs in an assembly connected by
#           at least one actual synapse
# P_pot   - fraction of neuron pairs connected by at least one potential synapse
# P_eff   - fraction of "required synapses" (given a learning signal S_ij) that
#           have been realized

# key stuff
# - cell assembly sizes must be limited to k for this to make sense,
# thats a constraint on output patterns size if we want recurrent k-WTA inhibition to be
# applied



class SimpleAssembly:
  """
  Simple cell assembly.
  Really just encapsulates the synaptic weights from one set of inputs to one
  set of outputs.

  NOT as stateful simulation approach, self.u and self.v are cached but
  should probably be removed

  This is an implicit state, implicit event based formulation with design
  driven by the module interface semantics, not the underlying network dynamics.



  """
  # properties: AssemblyProperties = AssemblyProperties()

  def __init__(self, n_input: int, n_output: int,
               assembly_size: int,
               synapse_rule: Callable[[nd.array, nd.array], nd.array],
               potentiated_synapse_fraction: float,
               ):
    self.n_u = n_input
    self.n_v = n_output

    self.R = synapse_rule
    self.k = assembly_size
    self.p_1 = potentiated_synapse_fraction
    self.u = nd.zeros(n_input)
    self.v = nd.zeros(n_output)

    self.W = nd.zeros((n_input, n_output))
    self.R_totals = nd.zeros((n_input, n_output))
    
  def random_init_synapses(self, rng: np.random.RandomState):

    self.W = nd.from_numpy(rng.uniform(size=self.W.shape)).astype('float32')
    #self.W = nd.random.uniform(shape=self.W.shape)
  

  def forward(self, input_pattern: nd.array):
    # check that its k or smaller in size?
    assert input_pattern.shape == self.u.shape
    self.u = input_pattern
    self.v = fire(W_synapse=self.W, u_input=self.u, k_wta=self.k)
    return self.v

  def assoc(self, input_pattern: nd.array, output_pattern: nd.array):

    assert input_pattern.shape == self.u.shape
    assert output_pattern.shape == self.v.shape

    self._update_weights(input_pattern, output_pattern)

  def _update_weights(self, input_pattern, output_pattern):
    # use zip net rule for now
    # TODO: figure out how the willshaw rule might apply in later stages of learning/
    #  consolidation

    self.R_totals += self.R(input_pattern, output_pattern, p=self.k / self.n_v)

    # how to get just p1 synapses per (output) neuron potentiated?
    # well each output neuron is a column in v
    # number of active synapses is the number of non-zero values in the weight matrix
    #  NEXT STEP: finish implementing this dynamic thresholding
    #  -- do i really need to compute theta_ij dynamically for every neuron? like a second weight matrix?
    #  -- or can i just take the topk at every timestep...hmmmmmm

    #self.W = self.R_totals - self.assembl

    active_synapses_per_neuron = int(self.n_u * self.p_1)

    # ensure that each output neuron has the right number of potentiated
    # synapses

    # synapse weights for each output neuron correspond to rows of W
    # take the top k of those and set to one
    self.W = self.R_totals.topk(axis=1, ret_typ='mask',
                                k=active_synapses_per_neuron,
                                is_ascend=False)


def R_homosynaptic(u, v, p):
  """Returns a matrix of pairs of values from u and v
  using the homosynaptic rule with probability (mean activation) p"""

  # return -p *

  # TODO:
  # may be able to use khatri_rao and some fanciness to make this
  # vectorized on gpu later

  W = nd.zeros((u.shape[0], v.shape[0]))
  for i, u_i in enumerate(u):
    for j, v_j in enumerate(v):
      W[i, j] = -p * u_i + (1 * v_j) * u_i

  return W
  # return -p * u + (1 * v)*u

def fire(W_synapse, u_input, k_wta):
  """Returns binary activation after k-wta is applied on the postsynaptic neurons"""

  v = nd.dot(W_synapse, u_input)

  # zero out activations that are not in the top-k
  v = v.topk(k=k_wta, ret_typ='mask')
  return v




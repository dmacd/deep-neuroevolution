from mxnet import ndarray as nd
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple

#@dataclass






class SimpleAssembly:
  """
  Simple cell assembly implemented as an LSTM where the hidden state
  encodes the synaptic weights

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

    # self.W = nd.zeros((n_input, n_output))
    self.R_totals = nd.zeros((n_input, n_output))

  def random_init_synapses(self, rng: np.random.RandomState):

    # TODO: maybe dont want to be so explicit here?
    # maybe hidden state can just be initialized to small random values?
    # self.W = nd.from_numpy(rng.uniform(size=self.W.shape)).astype('float32')
    pass



  def forward(self, input_pattern: nd.array) -> Tuple[nd.array, nd.array]:
    """Run network for a step, returns (output, hidden_state)

    Can just ignore whatever updates happen to the hidden step when doing
    pure inference?

    Alt approach: fix the hidden state for each step so it doesnt change


    """
    assert input_pattern.shape == self.u.shape


    # run lstm for a step
    # return
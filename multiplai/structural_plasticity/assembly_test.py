import pytest
import numpy as np
import mxnet.ndarray as nd
import multiplai.structural_plasticity.assembly as asm


def test_R_homo():
  assert allclose(asm.R_homosynaptic(nd.array([0, 1]), nd.array([0, 1]),
                                         p=.1),
                      nd.array([[0, 0], [-0.1, 0.9]]))

@pytest.fixture
def rng():
  return np.random.RandomState(seed=0)

def test_forward(rng):


  assembly_size = 3
  assembly = asm.SimpleAssembly(n_input=10, n_output=10,
                         assembly_size=assembly_size,
                         synapse_rule=asm.R_homosynaptic,
                         potentiated_synapse_fraction=.5
                         )
  assembly.random_init_synapses(rng)
  print(assembly.W)


  u = nd.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
  v = assembly.forward(u)
  print("u:", u)
  print("v:", v)

  assert sum(v) == assembly_size, "Should activate no more than assembly size!"

def test_weight_update(rng):


  assembly_size = 3
  assembly = asm.SimpleAssembly(n_input=10, n_output=10,
                         assembly_size=assembly_size,
                         synapse_rule=asm.R_homosynaptic,
                         potentiated_synapse_fraction=.5
                         )

  assembly.random_init_synapses(rng)


  # try binding a simple pattern and reproducing it


  u = nd.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
  v = assembly.forward(u)

  assert allclose(v, [0, 1, 1, 0, 0, 0, 0, 1, 0, 0])
  print("v:", v)


  v_prime = nd.array([1, 0, 0, 0, 0, 1, 0, 0, 0, 1])
  print('v_prime:', v_prime)

  for _ in range(5):
    assembly.assoc(u, v_prime)
    should_be_v_prime = assembly.forward(u)
    print('should be v_prime:', should_be_v_prime)

  assert allclose(should_be_v_prime, v_prime), "assoc didnt associate right"


  # NEXT STEP:
  # - decide what i actually want to reproduce
  #   - willshaw rule for sparse cell assemblies seems easier to get going on
  #     - could build and validate that, build memory capacity tests,
  #     then extend to zip net rules later once i have robust assays in place
  # - switch back to simulation-basis for cell assembly logic
  #   - update(pre, post, state) => state -- advance the sim one timestep with
  #                                          given pre and
  #                                          post values
  #   - forward(pre) -> post              -- run inference without affecting
  #                                          simulation state
  #
  #  can build assoc and query on top of these functions with one or more
  #  assemblies in play


################################################################################
## utils

def allclose(a, b):
  if isinstance(a, list):
    a = nd.array(a)
  if isinstance(b, list):
    b = nd.array(b)
  return nd.sum(nd.abs(a - b)) < 1e-6

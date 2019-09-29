import logging
import pickle
import time
import h5py

import numpy as np

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn, Block
from mxnet import ndarray as F
from mxnet import ndarray as nd

from gym import spaces
from multiplai.evals.nlu import model
from multiplai.evals.nlu import policy


def test_policy_init(embedding):

  p = policy.ClassifierPolicy(
    observation_space=spaces.Discrete(
    embedding.idx_to_vec.shape[0]),
    action_space=spaces.Discrete(10)
  )

  print("trainable params:")
  print(p.collect_trainable_params())

  print('total num trainable params:', p.num_params)

  # verify that frozen params dont show up in trainable params list
  assert 'embedding' in [v for v in p.block.collect_params().values()][0].name
  assert 'embedding' not in p.collect_trainable_params()[0].name

  # test getting from flat
  w_flat = p.get_trainable_flat()
  # print(w_flat)
  w_flat_incremented = w_flat + 1
  # TODO: assert we have values that are actually from the classifier params?

  # test setting from flat
  p.set_trainable_flat(w_flat_incremented)

  assert np.allclose(p.get_trainable_flat(),
    w_flat_incremented)


  w_flat_incremented[100] += 2
  p.set_trainable_flat(w_flat_incremented)

  assert np.isclose(3, p.get_trainable_flat()[100] -
                    w_flat[100])


  # test initialization
  w_flat = p.get_trainable_flat()
  p.reinitialize()

  assert not np.allclose(w_flat, p.get_trainable_flat())

def test_policy_act(single_example_env, embedding):
  env = single_example_env

  p = policy.ClassifierPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space
  )
  _action = p.act(observation=nd.array([0]))



def test_policy_rollout(single_example_env, embedding):

  env = single_example_env

  p = policy.ClassifierPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space
  )


  env.seed(0)
  p.rollout(env, render=True, timestep_limit=100,
            save_obs=True, random_stream=None)


# next tests
# - check env seeding / resetting works properly when reused many times
#    - looks like es was designed to run the same environment on different
#    params
#     - > implies that we'll want a multi-example env instead of a single
#     example
#     - the multi example presentation should still be build on top of a test
#     single example
#     - expect single example to converge rapidly to max reward on the (
#     identical) single example

# - test in context?
#

# - >>> test load and save properly and carefully!!!


# >>>>>> START HERE
#  -- figure out why h5 file thinks its closed on load!!

def test_policy_serialization(single_example_env):

  env = single_example_env

  p = policy.ClassifierPolicy(
    observation_space=env.observation_space,
    action_space=env.action_space,
    # kwargs for testing roundtrip
    hidden_size=5
  )

  import tempfile
  fd, fname = tempfile.mkstemp(prefix="policy_serialization_test",
                               suffix=".h5")


  # tweak some weight by a little just to be sure
  # i.e. set all elements of each param to its index in the param list


  w_flat = p.get_trainable_flat()

  w_flat[0:100] = 7.77
  p.set_trainable_flat(w_flat)
  p.save(fname)

  # load a new policy from the file

  p_loaded = policy.ClassifierPolicy.Load(fname)

  # check that the tweaked weights were right

  assert np.allclose(p_loaded.get_trainable_flat(), w_flat)
  assert np.allclose(p_loaded.get_trainable_flat()[0:100], 7.77)

  print('loaded args:', p_loaded.args)
  print('loaded kwargs:', p_loaded.kwargs)
  assert p_loaded.args == p.args
  assert p_loaded.kwargs == p.kwargs
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

  # n_labels = 10
  # classifier = model.ClassifierRNN(embedding, hidden_size=8,
  #                                  max_labels=n_labels)
  #
  # classifier.initialize(ctx=mx.cpu(), force_reinit=True)
  #
  # # force some params to be frozen
  # classifier.embedding.collect_params().setattr('grad_req', 'null')
  #
  # p = policy.MxNetPolicyBase(block=classifier)

  p = policy.ClassifierPolicy(
    observation_space=spaces.Discrete(
    embedding.idx_to_vec.shape[0]),
    action_space=spaces.Discrete(10)
  )

  # print("trainable params:")
  # print(p.collect_trainable_params())

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

  assert np.allclose(p.get_trainable_flat().asnumpy(),
    w_flat_incremented.asnumpy())


  w_flat_incremented[100] += 2
  p.set_trainable_flat(w_flat_incremented)

  assert np.isclose(3, p.get_trainable_flat().asnumpy()[100] -
                    w_flat.asnumpy()[100])


  # test initialization

  w_flat = p.get_trainable_flat()
  p.reinitialize()

  assert not np.allclose(w_flat.asnumpy(), p.get_trainable_flat().asnumpy())

  # NEXT STEP:
  # select only TRAINABLE parameters instead of ALL parameters. blech


# test rollout

def test_policy_rollout():


  pass
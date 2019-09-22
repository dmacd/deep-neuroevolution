import logging
import pickle
import time

import h5py
from typing import List, Tuple, Set, Optional

import numpy as np
# import tensorflow as tf
# import tensorflow.contrib.layers as layers
# from . import tf_util as U

import mxnet as mx
from mxnet import gluon, autograd
from mxnet.gluon import nn, rnn, Block
from mxnet import ndarray as F
from mxnet import ndarray as nd
from gym import spaces

from multiplai.evals.nlu import model

log = logging.getLogger(__name__)

class MxNetPolicyBase:
  def __init__(self, *args, **kwargs):
    self.args, self.kwargs = args, kwargs
    self.block = self._initialize(*args, **kwargs)
    self.ctx = mx.cpu()  # slight hack

    log.info("All parameters (%d)" % len(self.block.collect_params().keys()))
    log.info(self.block.collect_params())

    # NEXT STEP:
    # x trainable params only!!!!
    # - then implement and test rollout

    # self.scope = self._initialize(*args, **kwargs)
    # self.all_variables = tf.get_collection(tf.GraphKeys.VARIABLES, self.scope.name)
    #
    # self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope.name)
    # self.num_params = sum(int(np.prod(v.get_shape().as_list())) for v in self.trainable_variables)
    # self._setfromflat = U.SetFromFlat(self.trainable_variables)
    # self._getflat = U.GetFlat(self.trainable_variables)
    #
    # logger.info('Trainable variables ({} parameters)'.format(self.num_params))
    # for v in self.trainable_variables:
    #     shp = v.get_shape().as_list()
    #     logger.info('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))
    # logger.info('All variables')
    # for v in self.all_variables:
    #     shp = v.get_shape().as_list()
    #     logger.info('- {} shape:{} size:{}'.format(v.name, shp, np.prod(shp)))
    #
    # placeholders = [tf.placeholder(v.value().dtype, v.get_shape().as_list()) for v in self.all_variables]
    # self.set_all_vars = U.function(
    #     inputs=placeholders,
    #     outputs=[],
    #     updates=[tf.group(*[v.assign(p) for v, p in zip(self.all_variables, placeholders)])]
    # )
    
  def collect_trainable_params(self) -> List[gluon.Parameter]:
    return [p for p in self.block.collect_params().values()
            if p.grad_req != 'null']

  def reinitialize(self):
    self.block.initialize(ctx=self.ctx, force_reinit=True)

  def _initialize(self, *args, **kwargs) -> gluon.Block:
      raise NotImplementedError

  # def save(self, filename):
  #     assert filename.endswith('.h5')
  #     with h5py.File(filename, 'w', libver='latest') as f:
  #         for v in self.all_variables:
  #             f[v.name] = v.eval()
  #         # TODO: it would be nice to avoid pickle, but it's convenient to pass Python objects to _initialize
  #         # (like Gym spaces or numpy arrays)
  #         f.attrs['name'] = type(self).__name__
  #         f.attrs['args_and_kwargs'] = np.void(pickle.dumps((self.args, self.kwargs), protocol=-1))
  #
  # @classmethod
  # def Load(cls, filename, extra_kwargs=None):
  #     with h5py.File(filename, 'r') as f:
  #         args, kwargs = pickle.loads(f.attrs['args_and_kwargs'].tostring())
  #         if extra_kwargs:
  #             kwargs.update(extra_kwargs)
  #         policy = cls(*args, **kwargs)
  #         policy.set_all_vars(*[f[v.name][...] for v in policy.all_variables])
  #     return policy

  # === Rollouts/training ===

  def rollout(self, env, *, render=False, timestep_limit=None, save_obs=False,
              random_stream=None):
    """
    If random_stream is provided, the rollout will take noisy actions with noise drawn from that stream.
    Otherwise, no action noise will be added.
    """
    env_timestep_limit = env.spec.tags.get(
      'wrapper_config.TimeLimit.max_episode_steps')
    timestep_limit = env_timestep_limit if timestep_limit is None else min(
      timestep_limit, env_timestep_limit)
    rewards = []
    t = 0
    
    if save_obs:
      obs = []
      
    ob = env.reset()
    for _ in range(timestep_limit):
      ac = self.act(ob[None], random_stream=random_stream)[0]
      if save_obs:
        obs.append(ob)
      ob, rew, done, _ = env.step(ac)
      rewards.append(rew)
      t += 1
      if render:
        env.render()
      if done:
        break
    rewards = np.array(rewards, dtype=np.float32)
    
    if save_obs:
      return rewards, t, np.array(obs)
    return rewards, t

  def act(self, ob, random_stream=None):
    raise NotImplementedError

  def set_trainable_flat(self, x: np.array):
    self._set_from_flat(x)

  def get_trainable_flat(self) -> np.array:
    return self._get_from_flat()

  @property
  def needs_ob_stat(self):
    raise NotImplementedError

  def set_ob_stat(self, ob_mean, ob_std):
    raise NotImplementedError

  def _get_from_flat(self) -> nd.array:
    values = [v for v in self.collect_trainable_params()]
    flat = nd.concatenate([v.data().reshape((-1,)) for v in values])
    return flat

  def _set_from_flat(self, w_flat: np.array):
    # iterate over the collected parameters
    start = 0
    for v in self.collect_trainable_params():
      size = np.prod(v.shape)
      chunk = w_flat[start:start+size]
      v.set_data(chunk.reshape(v.shape))
      start += size
      

from multiplai.evals.nlu import data
from multiplai.evals.nlu import embedding as emb


class ClassifierPolicy(MxNetPolicyBase):
  
  def _initialize(self, observation_space: spaces.Discrete,
                  action_space: spaces.Discrete,
                  hidden_size=64) -> gluon.Block:

    # HACK: TODO: initialize embedding without loading all the data!
    # need to preprocessing the whole data space and save the
    # embedding layer somewhere to accomplish this
    # should probably obtain it from the environment itself, actually!

    all_text, all_entities = data.get_train_data_text_and_entities(
      data.load_train_data())

    embedding = emb.get_embedding_for_text(all_text)

    n_labels = action_space.n
    assert len(observation_space.shape) == 0
    assert embedding.idx_to_vec.shape[0] == observation_space.n

    classifier = model.ClassifierRNN(embedding, hidden_size=hidden_size,
                                     max_labels=n_labels)

    classifier.initialize(ctx=mx.cpu(), force_reinit=True)

    # force some params to be frozen
    classifier.embedding.collect_params().setattr('grad_req', 'null')

    return classifier
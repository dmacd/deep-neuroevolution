import gym
from es_distributed.es import get_policy_type
import numpy as np

import multiplai.evals.nlu.gym

import click


@click.command()
@click.argument('env_id')
@click.argument('policy_file')
@click.argument('policy_type')
@click.option('--record', is_flag=True)
@click.option('--stochastic', is_flag=True)
@click.option('--extra_kwargs')
def main(env_id, policy_file, policy_type,
         record, stochastic, extra_kwargs):
  # is_atari_policy = "NoFrameskip" in env_id

  env = gym.make(env_id)
  # if is_atari_policy:
  #   env = wrap_deepmind(env)

  # if record:
  #   import uuid
  #   env = wrappers.Monitor(env, '/tmp/' + str(uuid.uuid4()), force=True)

  if extra_kwargs:
    import json
    extra_kwargs = json.loads(extra_kwargs)

  policy_type = get_policy_type(policy_type)

  policy = policy_type.Load(policy_file, extra_kwargs=extra_kwargs)

  while True:
    rews, t, novelty_vector = policy.rollout(
      env, render=True,
      timestep_limit=1000,
      random_stream=np.random if stochastic else None)
    print('return={:.4f} len={}'.format(rews.sum(), t))

    if record:
      env.close()
      return


if __name__ == '__main__':
  main()

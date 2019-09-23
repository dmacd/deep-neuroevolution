

from gym.envs.registration import register

register(
    id='NLU-Bench-SingleExample-GetWeather-v0',
    entry_point='multiplai.evals.nlu.gym.env.nlu_benchmark_env'
                ':SingleExample_GetWeather',
)
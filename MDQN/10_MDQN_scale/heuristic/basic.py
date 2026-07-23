import numpy as np
from gymnasium import Env
from time import time
from joblib import Parallel, delayed


class Agent:
    """Example agent.
    """
    def __init__(self):
        pass

    def act(self, obs):
        pass

    def set_parameters(self, parameters:dict):
        for key, value in parameters.items():
            setattr(self, key, value)


def _evaluate_policy_single_process(agent:Agent, env:Env, length:int=None, repeats:int=1):
    cost_list = []
    for idx_repeat in range(repeats):
        terminated = False
        if not length is None:
            env.max_steps = length
        obs, info = env.reset()
        while not terminated:
            action = agent.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            cost_list.append(info["costs"]["total"])
    env.close()
    return np.mean(cost_list)


def evaluate_policy_multi_process(agent, env, length, repeats, parameters_list, n_jobs:int=-1):
    time_start = time()
    tasks = []
    for parameters in parameters_list:
        agent = agent.__class__(env, parameters)
        tasks.append([agent, env, length, repeats])
    if n_jobs == 1:
        performances = [_evaluate_policy_single_process(*task) for task in tasks]
    else:
        try:
            performances = Parallel(n_jobs=n_jobs,
                                    backend="loky",
                                    batch_size="auto")(delayed(_evaluate_policy_single_process)(*task) for task in tasks)
        except PermissionError:
            performances = [_evaluate_policy_single_process(*task) for task in tasks]
    best_parameters, best_performance = min(zip(parameters_list, performances), key=lambda x: x[1])
    time_end = time()
    log = dict()
    log["parameters"] = best_parameters
    log['performance'] = best_performance.item()
    log["time"] = time_end - time_start
    return log
    

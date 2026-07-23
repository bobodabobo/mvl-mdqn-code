from time import time

from gymnasium import Env
from joblib import Parallel, delayed

from metrics import aggregate_episode_metrics


class Agent:
    """Example agent."""

    def __init__(self):
        pass

    def act(self, obs):
        pass

    def set_parameters(self, parameters: dict):
        for key, value in parameters.items():
            setattr(self, key, value)


def _rollout_episode(agent: Agent, env: Env, reset_seed=None):
    terminated = False
    if reset_seed is None:
        obs, info = env.reset()
    else:
        obs, info = env.reset(seed=int(reset_seed))
    costs = []
    while not terminated:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        costs.append(
            {
                "timely": info["costs"]["timely"],
                "delayed": info["costs"]["delayed"],
                "total": info["costs"]["total"],
            }
        )
    return costs


def _evaluate_policy_single_process(agent: Agent, env: Env, length: int = None, repeats: int = 1, gamma: float = 1.0, seeds=None):
    if length is not None:
        env.max_steps = length
    n_repeats = 1 if repeats is None else repeats
    episode_costs = []
    if seeds is None:
        for _ in range(n_repeats):
            episode_costs.append(_rollout_episode(agent, env))
    else:
        for seed in seeds:
            for repeat_idx in range(n_repeats):
                episode_costs.append(_rollout_episode(agent, env, reset_seed=int(seed) + repeat_idx))
    env.close()
    return aggregate_episode_metrics(
        episode_costs,
        gamma,
        delay_time=getattr(env, "reward_delay_time", 0),
        use_delayed_cost_assignment=True,
    )


def evaluate_policy_multi_process(agent, env, length, repeats, parameters_list, gamma: float = 1.0, seeds=None, n_jobs: int = -1):
    time_start = time()
    tasks = []
    for parameters in parameters_list:
        agent = agent.__class__(env, parameters)
        tasks.append([agent, env, length, repeats, gamma, seeds])
    performances = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        batch_size="auto",
    )(delayed(_evaluate_policy_single_process)(*task) for task in tasks)
    best_parameters, best_metrics = min(zip(parameters_list, performances), key=lambda x: x[1]["performance"])
    time_end = time()
    log = dict()
    log["parameters"] = best_parameters
    log["performance"] = best_metrics["performance"]
    log["mean_cost"] = best_metrics["mean_cost"]
    log["mean_assigned_cost"] = best_metrics["mean_assigned_cost"]
    log["raw_discounted_cost"] = best_metrics["raw_discounted_cost"]
    log["metric"] = "normalized_discounted_cost"
    log["gamma_eval"] = gamma
    log["eval_seeds"] = list(seeds) if seeds is not None else None
    log["time"] = time_end - time_start
    return log

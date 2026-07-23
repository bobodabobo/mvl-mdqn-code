import os
import pickle
from time import time

import numpy as np
from joblib import Parallel, delayed

from DRL import train_test_DDQN, train_test_MDQN, train_test_PPO
from simulators import MultiItemInventory, multi_item_configs
from simulators import SerialMultiEchelonInventory, serial_multi_echelon_configs


DRL_METHODS = ["DDQN", "PPO", "MDQN"]
N_SEEDS = 16
TEST_REPEATS = 100
RESULTS_PATH = "results/DQN_results.pkl"

SYSTEM_SPECS = [
    ("ME", "Multi-Echelon", SerialMultiEchelonInventory, serial_multi_echelon_configs),
    ("MI", "Multi-Item", MultiItemInventory, multi_item_configs),
]


def solve_time(seconds: int):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours}h{minutes}m{seconds}s"


def run_parallel_seeds(func, seeds, **kwargs):
    n_jobs = int(os.environ.get("MDQN_DRL_N_JOBS", "-1"))
    try:
        results = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
            delayed(func)(seed=seed, **kwargs) for seed in seeds
        )
    except (PermissionError, OSError):
        results = [func(seed=seed, **kwargs) for seed in seeds]
    return sorted(results, key=lambda item: item["seed"])


def _get_train_test_fn(method: str):
    if method == "DDQN":
        return train_test_DDQN
    if method == "PPO":
        return train_test_PPO
    if method == "MDQN":
        return train_test_MDQN
    raise ValueError(f"Unsupported method: {method}")


def drl_experiment():
    print("===DRL Experiments===")
    np.random.seed(0)
    rl_seeds = np.random.choice(900, replace=False, size=N_SEEDS) + 100
    rl_seeds = rl_seeds.tolist()
    experiment_log = {}

    for system_key, system_title, env_cls, configs in SYSTEM_SPECS:
        print(f"---{system_title}---")
        t0 = time()
        experiment_log[system_key] = {}
        for task_idx, config in enumerate(configs, start=1):
            print(f"-task-{task_idx}-")
            task_name = f"{system_key}{task_idx}"
            env = env_cls(config)
            experiment_log[system_key][str(task_idx)] = {}
            for method in DRL_METHODS:
                print(method)
                t00 = time()
                train_fn = _get_train_test_fn(method)
                params = {
                    "env": env,
                    "task_name": task_name,
                    "test_repeats": TEST_REPEATS,
                }
                if method != "MDQN":
                    params["DRL_method"] = method
                results = run_parallel_seeds(train_fn, rl_seeds, **params)
                for result in results:
                    print(
                        f"Test: {task_name}_seed={result['seed']}, "
                        f"performance={result['performance']:.4f}@{result['step']}."
                    )
                experiment_log[system_key][str(task_idx)][method] = results
                print(f"{method} time cost:", solve_time(int(time() - t00)))
        print(f"{system_key} time cost:", solve_time(int(time() - t0)))
        print()

    with open(RESULTS_PATH, "wb") as file:
        pickle.dump(experiment_log, file)


if __name__ == "__main__":
    drl_experiment()

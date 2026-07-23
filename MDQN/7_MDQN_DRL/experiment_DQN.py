from simulators import LostSalesInventory, lost_sale_configs
from heuristic import CappedBaseStock
from simulators import PerishableInventory, perishable_configs
from heuristic import BSPLowEW
from simulators import DualSourcingInventory, dual_sourcing_configs
from heuristic import CappedDualIndex
from DRL import train_test_DQN, train_test_PPO, train_test_SAC, train_test_MDQN

from time import time
import pickle
import numpy as np
from joblib import Parallel, delayed


DRL_methods = ['DQN', 'DQN-L',
               'TDQN', 'TDQN-L',
               'DDQN', 'DDQN-L',
               'RSDQN', 'RSDQN-L',
               'PPO', 'PPO-L',
               'SAC', 'SAC-L']

RSDQN_methods = ['RSDQN', 'RSDQN-L']

N_tasks = 4

N_seeds = 16

TEST_REPEATS = 100


def solve_time(seconds:int):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours}h{minutes}m{seconds}s"


def run_parallel_seeds(func, seeds, **kwargs):
    results = Parallel(n_jobs=-1, backend="loky", batch_size="auto")(
        delayed(func)(seed=seed, **kwargs) for seed in seeds
    )
    results = sorted(results, key=lambda x: x["seed"])
    return results


def get_train_test_fn(DRL_method:str):
    if DRL_method.startswith("PPO"):
        return train_test_PPO
    if DRL_method.startswith("SAC"):
        return train_test_SAC
    return train_test_DQN


def DQN_experiment():
    print("===DRL Experiments===")
    # train seeds
    np.random.seed(0)
    RL_seeds = np.random.choice(900, replace=False, size=N_seeds)
    RL_seeds = RL_seeds + 100
    RL_seeds = RL_seeds.tolist()
    # total log
    experiment_log = dict()
    # 1.Lost Sales
    print("---Lost Sales---")
    t0 = time()
    experiment_log["LS"] = dict()
    for task_idx in range(N_tasks):
        print(f"-task-{task_idx + 1}-")
        task_name = f"LS{task_idx+1}"
        env = LostSalesInventory(lost_sale_configs[task_idx])
        experiment_log["LS"][str(task_idx + 1)] = dict()
        for DRL_method in DRL_methods:
            print(DRL_method)
            t00 = time()
            drl_params = {'DRL_method': DRL_method,
                          'env': env,
                          'task_name': task_name,
                          'test_repeats': TEST_REPEATS}
            train_fn = get_train_test_fn(DRL_method)
            if DRL_method in RSDQN_methods:
                teacher_agent = CappedBaseStock(env)
                with open("results/heuristic_results.pkl", "rb") as file:
                    heuristic_results = pickle.load(file)
                parameters = heuristic_results["LS"][f"{task_idx + 1}"]["CBS"]["parameters"]
                teacher_agent.set_parameters(parameters)
                results = run_parallel_seeds(train_test_DQN, RL_seeds, teacher_agent=teacher_agent, **drl_params)
            else:
                results = run_parallel_seeds(train_fn, RL_seeds, **drl_params)
            for result in results:
                print(f"Test: {task_name}_seed={result['seed']}, performance={result['performance']:.4f}@{result['step']}.")
            experiment_log["LS"][str(task_idx + 1)][DRL_method] = results
            t11 = time()
            print(f"{DRL_method} time cost:", solve_time(int(t11-t00)))
        print("MDQN")
        t00 = time()
        mdqn_params = {'env': env, 'task_name': task_name, 'test_repeats': TEST_REPEATS}
        results = run_parallel_seeds(train_test_MDQN, RL_seeds, **mdqn_params)
        for result in results:
            print(f"Test: {task_name}_seed={result['seed']}, performance={result['performance']:.4f}@{result['step']}.")
        experiment_log["LS"][str(task_idx + 1)]["MDQN"] = results
        t11 = time()
        print(f"MDQN time cost:", solve_time(int(t11-t00)))
    t1 = time()
    print("LS time cost:", solve_time(int(t1-t0)))
    print()
    # 2.Perishable
    print("---Perishable---")
    t0 = time()
    experiment_log["PS"] = dict()
    for task_idx in range(N_tasks):
        print(f"-task-{task_idx + 1}-")
        task_name = f"PS{task_idx+1}"
        env = PerishableInventory(perishable_configs[task_idx])
        experiment_log["PS"][str(task_idx + 1)] = dict()
        for DRL_method in DRL_methods:
            t00 = time()
            print(DRL_method)
            drl_params = {'DRL_method': DRL_method, 'env': env, 'task_name': task_name, 'test_repeats': TEST_REPEATS}
            train_fn = get_train_test_fn(DRL_method)
            if DRL_method in RSDQN_methods:
                teacher_agent = BSPLowEW(env)
                with open("results/heuristic_results.pkl", "rb") as file:
                    heuristic_results = pickle.load(file)
                parameters = heuristic_results["PS"][f"{task_idx + 1}"]["BSLEW"]["parameters"]
                teacher_agent.set_parameters(parameters)
                results = run_parallel_seeds(train_test_DQN, RL_seeds, teacher_agent=teacher_agent, **drl_params)
            else:
                results = run_parallel_seeds(train_fn, RL_seeds, **drl_params)
            for result in results:
                print(f"Test: {task_name}_seed={result['seed']}, performance={result['performance']:.4f}@{result['step']}.")
            experiment_log["PS"][str(task_idx + 1)][DRL_method] = results
            t11 = time()
            print(f"{DRL_method} time cost:", solve_time(int(t11-t00)))
        print("MDQN")
        t00 = time()
        mdqn_params = {'env': env, 'task_name': task_name, 'test_repeats': TEST_REPEATS}
        results = run_parallel_seeds(train_test_MDQN, RL_seeds, **mdqn_params)
        for result in results:
            print(f"Test: {task_name}_seed={result['seed']}, performance={result['performance']:.4f}@{result['step']}.")
        experiment_log["PS"][str(task_idx + 1)]["MDQN"] = results
        t11 = time()
        print(f"MDQN time cost:", solve_time(int(t11-t00)))
    t1 = time()
    print("PS time cost:", solve_time(int(t1-t0)))
    print()
    # 3.Dual Sourcing
    print("---Dual Sourcing---")
    t0 = time()
    experiment_log["DS"] = dict()
    for task_idx in range(N_tasks):
        print(f"-task-{task_idx + 1}-")
        task_name = f"DS{task_idx+1}"
        env = DualSourcingInventory(dual_sourcing_configs[task_idx])
        experiment_log["DS"][str(task_idx + 1)] = dict()
        for DRL_method in DRL_methods:
            print(DRL_method)
            t00 = time()
            drl_params = {'DRL_method': DRL_method, 'env': env, 'task_name': task_name, 'test_repeats': TEST_REPEATS}
            train_fn = get_train_test_fn(DRL_method)
            if DRL_method in RSDQN_methods:
                teacher_agent = CappedDualIndex(env)
                with open("results/heuristic_results.pkl", "rb") as file:
                    heuristic_results = pickle.load(file)
                parameters = heuristic_results["DS"][f"{task_idx + 1}"]["CDI"]["parameters"]
                teacher_agent.set_parameters(parameters)
                results = run_parallel_seeds(train_test_DQN, RL_seeds, teacher_agent=teacher_agent, **drl_params)
            else:
                results = run_parallel_seeds(train_fn, RL_seeds, **drl_params)
            for result in results:
                print(f"Test: {task_name}_seed={result['seed']}, performance={result['performance']:.4f}@{result['step']}.")
            experiment_log["DS"][str(task_idx + 1)][DRL_method] = results
            t11 = time()
            print(f"{DRL_method} time cost:", solve_time(int(t11-t00)))
        print("MDQN")
        t00 = time()
        mdqn_params = {'env': env, 'task_name': task_name, 'test_repeats': TEST_REPEATS}
        results = run_parallel_seeds(train_test_MDQN, RL_seeds, **mdqn_params)
        for result in results:
            print(f"Test: {task_name}_seed={result['seed']}, performance={result['performance']:.4f}@{result['step']}.")
        experiment_log["DS"][str(task_idx + 1)]["MDQN"] = results
        t11 = time()
        print(f"MDQN time cost:", solve_time(int(t11-t00)))
    t1 = time()
    print("DS time cost:", solve_time(int(t1-t0)))
    print()
    with open("results/DQN_results.pkl", "wb") as file:
        pickle.dump(experiment_log, file)


if __name__ == "__main__":
    DQN_experiment()

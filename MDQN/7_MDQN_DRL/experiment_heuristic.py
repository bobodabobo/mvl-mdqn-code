from simulators import LostSalesInventory, lost_sale_configs
from heuristic import BaseStock
from heuristic import CappedBaseStock
from simulators import PerishableInventory, perishable_configs
from heuristic import BSPLowEW
from simulators import DualSourcingInventory, dual_sourcing_configs
from heuristic import DualIndex
from heuristic import CappedDualIndex

import pickle

N_tasks = 4

def heuristic_experiment(train_length:int=1000, repeats:int=100):
    print("===Heuristic Experiments===")
    results_dict = dict()

    print("---Lost Sales---")
    results_dict["LS"] = dict()
    for task_id in range(1, N_tasks + 1):
        print(f"-task-{task_id}-")
        results_dict["LS"][str(task_id)] = dict()
        # lost-sales inventory environment
        env = LostSalesInventory(lost_sale_configs[task_id - 1])
        # base-stock policy
        print("BaseStock")
        agent = BaseStock(env)
        log = agent.train(length=train_length, repeats=repeats)
        results_dict["LS"][str(task_id)]["BS"] = log
        print(log)
        # capped base-stock policy
        print("CappedBaseStock")
        agent = CappedBaseStock(env)
        log = agent.train(length=train_length, repeats=repeats)
        results_dict["LS"][str(task_id)]["CBS"] = log
        print(log)
    print()

    print("---Perishable---")
    results_dict["PS"] = dict()
    for task_id in range(1, N_tasks+1):
        print(f"-task-{task_id}-")
        results_dict["PS"][str(task_id)] = dict()
        # perishable inventory environment
        env = PerishableInventory(perishable_configs[task_id - 1])
        # base-stock policy
        print("BaseStock")
        agent = BaseStock(env)
        log = agent.train(length=train_length, repeats=repeats)
        results_dict["PS"][str(task_id)]["BS"] = log
        print(log)
        # BSP-low-EW policy
        print("BSPLowEW")
        agent = BSPLowEW(env)
        log = agent.train(length=train_length, repeats=repeats)
        results_dict["PS"][str(task_id)]["BSLEW"] = log
        print(log)
    print()

    print("---Dual Sourcing---")
    results_dict["DS"] = dict()
    for task_id in range(1, N_tasks+1):
        print(f"-task-{task_id}-")
        results_dict["DS"][str(task_id)] = dict()
        # dual-sourcing inventory environment
        env = DualSourcingInventory(dual_sourcing_configs[task_id - 1])
        # dual-index policy
        print("DualIndex")
        agent = DualIndex(env)
        log = agent.train(length=train_length, repeats=repeats)
        results_dict["DS"][str(task_id)]["DI"] = log
        print(log)
        # capped dual-index policy
        print("CappedDualIndex")
        agent = CappedDualIndex(env)
        log = agent.train(length=train_length, repeats=repeats)
        results_dict["DS"][str(task_id)]["CDI"] = log
        print(log)
    print()
    
    with open("results/heuristic_results.pkl", "wb") as f:
        pickle.dump(results_dict, f) # result_dict["LS"]["1"]["BS"]


if __name__ == "__main__":
    heuristic_experiment()
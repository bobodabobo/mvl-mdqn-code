import argparse
import csv
import pickle
from copy import deepcopy as copy
from pathlib import Path
from statistics import mean, pstdev

from joblib import Parallel, delayed

from DRL import train_test_DQN, train_test_MDQN, train_test_PPO
from heuristic import CappedBaseStock
from simulators import LostSalesInventory, get_xin2020_lost_sale_configs


DEFAULT_SEEDS = [592, 241, 509, 131]
DEFAULT_METHODS = ("DDQN", "PPO", "MDQN", "CBS")
DEFAULT_TEST_REPEATS = 100
DEFAULT_CBS_TRAIN_LENGTH = 1000
DEFAULT_CBS_REPEATS = 10


def run_parallel_tasks(task_specs:list[dict], n_jobs:int=-1):
    if n_jobs == 1:
        results = [run_single_task(task_spec) for task_spec in task_specs]
    else:
        try:
            results = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1)(
                delayed(run_single_task)(task_spec) for task_spec in task_specs
            )
        except PermissionError:
            results = [run_single_task(task_spec) for task_spec in task_specs]
    return results


def run_cbs_search(seed:int, config:dict, train_length:int, repeats:int, search_n_jobs:int=1):
    seeded_config = copy(config)
    seeded_config["seed"] = seed
    env = LostSalesInventory(seeded_config)
    agent = CappedBaseStock(env)
    log = agent.train(length=train_length, repeats=repeats, n_jobs=search_n_jobs)
    log["seed"] = seed
    return log


def summarize_seed_results(results):
    performances = [result["performance"] for result in results]
    best_result = min(results, key=lambda result: result["performance"])
    summary = {
        "mean": mean(performances),
        "std": pstdev(performances) if len(performances) > 1 else 0.0,
        "best": best_result["performance"],
        "best_seed": best_result["seed"],
        "performances": performances,
        "seeds": [result["seed"] for result in results],
    }
    if "parameters" in best_result:
        summary["best_parameters"] = copy(best_result["parameters"])
    return summary


def get_selected_configs(instance_ids=None):
    configs = get_xin2020_lost_sale_configs()
    if instance_ids is None:
        return configs
    config_map = {config["instance_id"]: config for config in configs}
    missing_ids = [instance_id for instance_id in instance_ids if instance_id not in config_map]
    if missing_ids:
        raise ValueError(f"Unknown instance ids: {missing_ids}")
    return [copy(config_map[instance_id]) for instance_id in instance_ids]


def run_rl_seed(method_name:str, seed:int, config:dict, test_repeats:int, train_steps:int=None):
    env = LostSalesInventory(config)
    task_name = config["task_name"]
    if method_name == "DDQN":
        train_fn = train_test_DQN
        params = {"DRL_method": "DDQN"}
    elif method_name == "PPO":
        train_fn = train_test_PPO
        params = {"DRL_method": "PPO"}
    elif method_name == "MDQN":
        train_fn = train_test_MDQN
        params = {}
    else:
        raise ValueError(f"Unsupported RL method {method_name}.")
    params.update(
        {
            "env": env,
            "task_name": task_name,
            "seed": seed,
            "test_repeats": test_repeats,
        }
    )
    if train_steps is not None:
        params["train_steps"] = train_steps
    return train_fn(**params)


def build_task_specs(configs:list[dict], args, train_steps_by_method:dict):
    task_specs = []
    for config in configs:
        for method_name in args.methods:
            for seed in args.seeds:
                task_specs.append(
                    {
                        "config": copy(config),
                        "instance_id": config["instance_id"],
                        "method_name": method_name,
                        "seed": seed,
                        "test_repeats": args.test_repeats,
                        "cbs_train_length": args.cbs_train_length,
                        "cbs_repeats": args.cbs_repeats,
                        "train_steps": train_steps_by_method.get(method_name),
                    }
                )
    return task_specs


def run_single_task(task_spec:dict):
    method_name = task_spec["method_name"]
    seed = task_spec["seed"]
    config = copy(task_spec["config"])
    if method_name == "CBS":
        result = run_cbs_search(
            seed,
            config,
            train_length=task_spec["cbs_train_length"],
            repeats=task_spec["cbs_repeats"],
            search_n_jobs=1,
        )
    else:
        result = run_rl_seed(
            method_name,
            seed,
            config,
            test_repeats=task_spec["test_repeats"],
            train_steps=task_spec["train_steps"],
        )
    return {
        "instance_id": task_spec["instance_id"],
        "method_name": method_name,
        "seed": seed,
        "result": result,
    }


def print_result(method_name:str, task_name:str, result:dict):
    message = (
        f"{task_name} {method_name} seed={result['seed']}: "
        f"performance={result['performance']:.4f}"
    )
    if "step" in result:
        message += f" @ step={result['step']}"
    if method_name == "CBS":
        parameters = result["parameters"]
        message += (
            f", base_stock={parameters['base_stock']}, cap={parameters['cap']}"
        )
    print(message)


def write_summary_csv(output_path:Path, experiment_log:dict):
    meta = experiment_log["meta"]
    n_seed_slots = len(meta["seeds"])
    fieldnames = [
        "instance_id",
        "task_name",
        "method",
        "demand_dist",
        "lead_time",
        "penalty_cost",
        "mean_performance",
        "std_performance",
        "best_performance",
        "best_seed",
    ]
    fieldnames.extend([f"seed_{idx + 1}" for idx in range(n_seed_slots)])
    fieldnames.extend([f"performance_{idx + 1}" for idx in range(n_seed_slots)])
    fieldnames.extend([f"base_stock_{idx + 1}" for idx in range(n_seed_slots)])
    fieldnames.extend([f"cap_{idx + 1}" for idx in range(n_seed_slots)])
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for instance_id in meta["instance_ids"]:
            instance_log = experiment_log["instances"][instance_id]
            config = instance_log["config"]
            for method_name in meta["methods"]:
                method_log = instance_log[method_name]
                summary = method_log["summary"]
                row = {
                    "instance_id": instance_id,
                    "task_name": config["task_name"],
                    "method": method_name,
                    "demand_dist": config["demand_dist"],
                    "lead_time": config["lead_time"],
                    "penalty_cost": config["penalty_cost"],
                    "mean_performance": summary["mean"],
                    "std_performance": summary["std"],
                    "best_performance": summary["best"],
                    "best_seed": summary["best_seed"],
                }
                for idx, result in enumerate(method_log["results"]):
                    row[f"seed_{idx + 1}"] = result["seed"]
                    row[f"performance_{idx + 1}"] = result["performance"]
                    if method_name == "CBS":
                        parameters = result["parameters"]
                        row[f"base_stock_{idx + 1}"] = parameters["base_stock"]
                        row[f"cap_{idx + 1}"] = parameters["cap"]
                writer.writerow(row)


def run_experiment(args):
    train_steps_by_method = {
        "DDQN": args.ddqn_train_steps,
        "PPO": args.ppo_train_steps,
        "MDQN": args.mdqn_train_steps,
    }
    configs = get_selected_configs(args.instance_ids)
    experiment_log = {
        "meta": {
            "systems": ["LS"],
            "methods": list(args.methods),
            "seeds": list(args.seeds),
            "test_repeats": args.test_repeats,
            "cbs_train_length": args.cbs_train_length,
            "cbs_repeats": args.cbs_repeats,
            "instance_ids": [config["instance_id"] for config in configs],
            "train_steps_by_method": train_steps_by_method,
        },
        "instances": {},
    }
    task_specs = build_task_specs(configs, args, train_steps_by_method)
    print(
        f"Dispatching {len(task_specs)} parallel tasks "
        f"({len(configs)} instances x {len(args.methods)} methods x {len(args.seeds)} seeds)."
    )
    task_results = run_parallel_tasks(task_specs, args.n_jobs)
    grouped_results = {}
    for task_result in task_results:
        task_key = (task_result["instance_id"], task_result["method_name"])
        grouped_results.setdefault(task_key, []).append(task_result["result"])
    for config in configs:
        instance_id = config["instance_id"]
        task_name = config["task_name"]
        print(f"=== {instance_id} dist={config['demand_dist']} L={config['lead_time']} p={config['penalty_cost']} ===")
        instance_log = {"config": copy(config)}
        for method_name in args.methods:
            results = sorted(
                grouped_results[(instance_id, method_name)],
                key=lambda result: result["seed"],
            )
            for result in results:
                print_result(method_name, task_name, result)
            instance_log[method_name] = {
                "results": results,
                "summary": summarize_seed_results(results),
            }
        experiment_log["instances"][instance_id] = instance_log
    return experiment_log


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Xin 2020 LS-32 experiments in MDQN_scale."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(DEFAULT_METHODS),
        choices=list(DEFAULT_METHODS),
        help="Methods to evaluate for each Xin-32 LS instance.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Random seeds to run for each instance.",
    )
    parser.add_argument(
        "--instance-ids",
        nargs="+",
        default=None,
        help="Optional subset of Xin instance ids, e.g. XLS01 XLS02.",
    )
    parser.add_argument(
        "--test-repeats",
        type=int,
        default=DEFAULT_TEST_REPEATS,
        help="Test repeats for DDQN/PPO/MDQN.",
    )
    parser.add_argument(
        "--cbs-train-length",
        type=int,
        default=DEFAULT_CBS_TRAIN_LENGTH,
        help="Episode length for each CBS search rollout.",
    )
    parser.add_argument(
        "--cbs-repeats",
        type=int,
        default=DEFAULT_CBS_REPEATS,
        help="Number of repeats used when evaluating each CBS candidate.",
    )
    parser.add_argument(
        "--ddqn-train-steps",
        type=int,
        default=None,
        help="Optional DDQN train step override for quick runs.",
    )
    parser.add_argument(
        "--ppo-train-steps",
        type=int,
        default=None,
        help="Optional PPO train step override for quick runs.",
    )
    parser.add_argument(
        "--mdqn-train-steps",
        type=int,
        default=None,
        help="Optional MDQN train step override for quick runs.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel workers used across all instance-method-seed tasks.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    results_dir = root_dir / "results"
    results_dir.mkdir(exist_ok=True)

    experiment_log = run_experiment(args)

    raw_output = results_dir / "xin2020_ls32_results.pkl"
    summary_output = results_dir / "xin2020_ls32_summary.csv"

    with open(raw_output, "wb") as file:
        pickle.dump(experiment_log, file)
    write_summary_csv(summary_output, experiment_log)

    print(f"Saved raw results to {raw_output}")
    print(f"Saved summary results to {summary_output}")


if __name__ == "__main__":
    main()

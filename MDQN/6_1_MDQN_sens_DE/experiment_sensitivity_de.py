import argparse
import csv
import pickle
from pathlib import Path
from statistics import mean, pstdev

from joblib import Parallel, delayed

from DRL import train_test_MDQN
from simulators import LostSalesInventory, lost_sale_configs


DEFAULT_SEEDS = [592, 241, 509, 131, 670, 693, 973, 499]
DEFAULT_TASKS = ("LS1", "LS2", "LS3", "LS4")
TEST_REPEATS = 100
METHOD_SPECS = (
    ("MDQN", {"final_test_all_heads": True}),
    (
        "MDQN-DEoff",
        {
            "adaptive_exploration": False,
            "fixed_head_idx": 0,
            "final_test_all_heads": True,
        },
    ),
)


def run_parallel_seeds(func, seeds, n_jobs=-1, **kwargs):
    results = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
        delayed(func)(seed=seed, **kwargs) for seed in seeds
    )
    return sorted(results, key=lambda item: item["seed"])


def load_cbs_results(results_dir: Path):
    with open(results_dir / "heuristic_results.pkl", "rb") as file:
        heuristic_results = pickle.load(file)
    return {
        f"LS{task_idx}": heuristic_results["LS"][str(task_idx)]["CBS"]
        for task_idx in range(1, 5)
    }


def summarize_seed_results(results):
    performances = [result["performance"] for result in results]
    best_result = min(results, key=lambda result: result["performance"])
    return {
        "n_seeds": len(results),
        "mean": mean(performances),
        "std": pstdev(performances) if len(performances) > 1 else 0.0,
        "best": best_result["performance"],
        "best_seed": best_result["seed"],
    }


def write_summary_csv(output_path: Path, experiment_log):
    fieldnames = [
        "task",
        "method",
        "n_seeds",
        "mean_performance",
        "std_performance",
        "best_performance",
        "best_seed",
        "cbs_performance",
        "gap_to_cbs",
    ]
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for task_name in experiment_log["meta"]["tasks"]:
            cbs_performance = experiment_log[task_name]["CBS"]["performance"]
            for method_name, _ in METHOD_SPECS:
                summary = experiment_log[task_name][method_name]["summary"]
                writer.writerow(
                    {
                        "task": task_name,
                        "method": method_name,
                        "n_seeds": summary["n_seeds"],
                        "mean_performance": summary["mean"],
                        "std_performance": summary["std"],
                        "best_performance": summary["best"],
                        "best_seed": summary["best_seed"],
                        "cbs_performance": cbs_performance,
                        "gap_to_cbs": summary["mean"] - cbs_performance,
                    }
                )
            writer.writerow(
                {
                    "task": task_name,
                    "method": "CBS",
                    "n_seeds": 0,
                    "mean_performance": cbs_performance,
                    "std_performance": 0.0,
                    "best_performance": cbs_performance,
                    "best_seed": "",
                    "cbs_performance": cbs_performance,
                    "gap_to_cbs": 0.0,
                }
            )


def write_seed_results_csv(output_path: Path, experiment_log):
    fieldnames = [
        "task",
        "method",
        "seed",
        "performance",
        "step",
        "selected_head",
        "selected_horizon",
        "best_test_head",
        "best_test_horizon",
        "cbs_performance",
        "gap_to_cbs",
    ]
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for task_name in experiment_log["meta"]["tasks"]:
            cbs_performance = experiment_log[task_name]["CBS"]["performance"]
            for method_name, _ in METHOD_SPECS:
                for result in experiment_log[task_name][method_name]["results"]:
                    writer.writerow(
                        {
                            "task": task_name,
                            "method": method_name,
                            "seed": result["seed"],
                            "performance": result["performance"],
                            "step": result["step"],
                            "selected_head": result["selected_head"],
                            "selected_horizon": result["selected_horizon"],
                            "best_test_head": result["best_test_head"],
                            "best_test_horizon": result["best_test_horizon"],
                            "cbs_performance": cbs_performance,
                            "gap_to_cbs": result["performance"] - cbs_performance,
                        }
                    )


def write_final_head_tests_csv(output_path: Path, experiment_log):
    max_heads = 0
    for task_name in experiment_log["meta"]["tasks"]:
        for method_name, _ in METHOD_SPECS:
            for result in experiment_log[task_name][method_name]["results"]:
                head_tests = result["head_test_performances"] or []
                max_heads = max(max_heads, len(head_tests))
    fieldnames = [
        "task",
        "method",
        "seed",
        "selected_head",
        "best_test_head",
    ] + [f"head_{idx}" for idx in range(max_heads)]
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for task_name in experiment_log["meta"]["tasks"]:
            for method_name, _ in METHOD_SPECS:
                for result in experiment_log[task_name][method_name]["results"]:
                    row = {
                        "task": task_name,
                        "method": method_name,
                        "seed": result["seed"],
                        "selected_head": result["selected_head"],
                        "best_test_head": result["best_test_head"],
                    }
                    for idx_head, performance in enumerate(result["head_test_performances"] or []):
                        row[f"head_{idx_head}"] = performance
                    writer.writerow(row)


def build_task_env(task_name: str):
    task_idx = int(task_name[2:]) - 1
    return LostSalesInventory(lost_sale_configs[task_idx]), lost_sale_configs[task_idx]


def run_experiment(tasks, seeds, test_repeats, n_jobs):
    root_dir = Path(__file__).resolve().parent
    results_dir = root_dir / "results"
    cbs_results = load_cbs_results(results_dir)
    experiment_log = {
        "meta": {
            "systems": ["LS"],
            "tasks": list(tasks),
            "methods": [method_name for method_name, _ in METHOD_SPECS] + ["CBS"],
            "seeds": list(seeds),
            "test_repeats": test_repeats,
        }
    }

    for task_name in tasks:
        env, task_config = build_task_env(task_name)
        task_log = {
            "CBS": cbs_results[task_name],
            "config": task_config,
        }
        for method_name, overrides in METHOD_SPECS:
            params = {
                "env": env,
                "task_name": task_name,
                "test_repeats": test_repeats,
                **overrides,
            }
            results = run_parallel_seeds(train_test_MDQN, seeds, n_jobs=n_jobs, **params)
            task_log[method_name] = {
                "results": results,
                "summary": summarize_seed_results(results),
                "overrides": overrides,
            }
            for result in results:
                print(
                    f"{task_name} {method_name} seed={result['seed']}: "
                    f"performance={result['performance']:.4f} @ step={result['step']} "
                    f"(train_h={result['selected_horizon']}, test_h={result['best_test_horizon']})"
                )
        task_log["CBS"]["performance"] = float(task_log["CBS"]["performance"])
        print(f"{task_name} CBS: performance={task_log['CBS']['performance']:.4f}")
        experiment_log[task_name] = task_log
    return experiment_log


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LS-only MDQN dynamic exploration sensitivity experiments."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(DEFAULT_TASKS),
        choices=list(DEFAULT_TASKS),
        help="Lost-sales tasks to run.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help="Training seeds to run for each LS task.",
    )
    parser.add_argument(
        "--test-repeats",
        type=int,
        default=TEST_REPEATS,
        help="Number of test repeats for each trained seed.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Joblib parallel workers for running seeds.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    results_dir = root_dir / "results"
    results_dir.mkdir(exist_ok=True)

    experiment_log = run_experiment(args.tasks, args.seeds, args.test_repeats, args.n_jobs)

    raw_output = results_dir / "mdqn_de_sensitivity.pkl"
    summary_output = results_dir / "mdqn_de_sensitivity_summary.csv"
    seed_output = results_dir / "mdqn_de_sensitivity_seed_results.csv"
    head_output = results_dir / "mdqn_de_final_head_tests.csv"

    with open(raw_output, "wb") as file:
        pickle.dump(experiment_log, file)
    write_summary_csv(summary_output, experiment_log)
    write_seed_results_csv(seed_output, experiment_log)
    write_final_head_tests_csv(head_output, experiment_log)

    print(f"Saved raw results to {raw_output}")
    print(f"Saved summary table to {summary_output}")
    print(f"Saved seed table to {seed_output}")
    print(f"Saved final head tests to {head_output}")


if __name__ == "__main__":
    main()

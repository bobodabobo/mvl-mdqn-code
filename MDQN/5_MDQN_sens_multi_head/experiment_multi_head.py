import argparse
import csv
import pickle
from pathlib import Path
from statistics import mean, pstdev

from joblib import Parallel, delayed

from DRL import train_test_MDQN
from simulators import LostSalesInventory, lost_sale_configs


DEFAULT_SEEDS = [592, 241]
TEST_REPEATS = 100
METHOD_SPECS = (
    ("MDQN-H1", {"H": 1}),
    ("MDQN-H7", {}),
)
TASK_SPECS = (
    ("1", "LS1"),
    ("2", "LS2"),
    ("3", "LS3"),
    ("4", "LS4"),
)


def run_parallel_seeds(func, seeds, n_jobs=-1, **kwargs):
    results = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
        delayed(func)(seed=seed, **kwargs) for seed in seeds
    )
    return sorted(results, key=lambda item: item["seed"])


def load_cbs_results(results_dir: Path):
    with open(results_dir / "heuristic_results.pkl", "rb") as file:
        heuristic_results = pickle.load(file)
    return {task_idx: heuristic_results["LS"][task_idx]["CBS"] for task_idx, _ in TASK_SPECS}


def summarize_seed_results(results):
    performances = [result["performance"] for result in results]
    best_result = min(results, key=lambda result: result["performance"])
    return {
        "mean": mean(performances),
        "std": pstdev(performances) if len(performances) > 1 else 0.0,
        "best": best_result["performance"],
        "best_seed": best_result["seed"],
        "performances": performances,
        "seeds": [result["seed"] for result in results],
    }


def write_summary_csv(output_path: Path, experiment_log):
    fieldnames = [
        "task",
        "method",
        "seed_1",
        "performance_1",
        "seed_2",
        "performance_2",
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
        for _, task_name in TASK_SPECS:
            cbs_performance = experiment_log[task_name]["CBS"]["performance"]
            for method_name, _ in METHOD_SPECS:
                summary = experiment_log[task_name][method_name]["summary"]
                performances = summary["performances"]
                seeds = summary["seeds"]
                row = {
                    "task": task_name,
                    "method": method_name,
                    "seed_1": seeds[0],
                    "performance_1": performances[0],
                    "seed_2": seeds[1] if len(seeds) > 1 else "",
                    "performance_2": performances[1] if len(performances) > 1 else "",
                    "mean_performance": summary["mean"],
                    "std_performance": summary["std"],
                    "best_performance": summary["best"],
                    "best_seed": summary["best_seed"],
                    "cbs_performance": cbs_performance,
                    "gap_to_cbs": summary["mean"] - cbs_performance,
                }
                writer.writerow(row)


def write_gap_csv(output_path: Path, experiment_log):
    fieldnames = [
        "task",
        "cbs_performance",
        "mdqn_h1_mean",
        "mdqn_h7_mean",
        "mdqn_h1_gap_to_cbs",
        "mdqn_h7_gap_to_cbs",
        "mdqn_h7_minus_h1",
    ]
    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for _, task_name in TASK_SPECS:
            cbs_performance = experiment_log[task_name]["CBS"]["performance"]
            h1_mean = experiment_log[task_name]["MDQN-H1"]["summary"]["mean"]
            h7_mean = experiment_log[task_name]["MDQN-H7"]["summary"]["mean"]
            writer.writerow(
                {
                    "task": task_name,
                    "cbs_performance": cbs_performance,
                    "mdqn_h1_mean": h1_mean,
                    "mdqn_h7_mean": h7_mean,
                    "mdqn_h1_gap_to_cbs": h1_mean - cbs_performance,
                    "mdqn_h7_gap_to_cbs": h7_mean - cbs_performance,
                    "mdqn_h7_minus_h1": h7_mean - h1_mean,
                }
            )


def run_experiment(seeds, test_repeats, n_jobs):
    root_dir = Path(__file__).resolve().parent
    results_dir = root_dir / "results"
    cbs_results = load_cbs_results(results_dir)
    experiment_log = {
        "meta": {
            "systems": ["LS"],
            "tasks": [task_name for _, task_name in TASK_SPECS],
            "methods": [method_name for method_name, _ in METHOD_SPECS],
            "seeds": seeds,
            "test_repeats": test_repeats,
        }
    }

    for task_idx, task_name in TASK_SPECS:
        env = LostSalesInventory(lost_sale_configs[int(task_idx) - 1])
        task_log = {
            "CBS": cbs_results[task_idx],
            "config": lost_sale_configs[int(task_idx) - 1],
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
                    f"performance={result['performance']:.4f} @ step={result['step']}"
                )
        experiment_log[task_name] = task_log
    return experiment_log


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LS-only MDQN multi-head sensitivity experiments."
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

    experiment_log = run_experiment(args.seeds, args.test_repeats, args.n_jobs)

    raw_output = results_dir / "mdqn_multi_head_sensitivity.pkl"
    summary_output = results_dir / "mdqn_multi_head_sensitivity.csv"
    gap_output = results_dir / "mdqn_multi_head_vs_cbs.csv"

    with open(raw_output, "wb") as file:
        pickle.dump(experiment_log, file)
    write_summary_csv(summary_output, experiment_log)
    write_gap_csv(gap_output, experiment_log)

    print(f"Saved raw results to {raw_output}")
    print(f"Saved summary table to {summary_output}")
    print(f"Saved CBS gap table to {gap_output}")


if __name__ == "__main__":
    main()

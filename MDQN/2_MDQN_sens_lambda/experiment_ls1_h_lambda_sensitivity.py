from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from DRL import train_test_MDQN
from simulators import (
    DualSourcingInventory,
    LostSalesInventory,
    PerishableInventory,
    dual_sourcing_configs,
    lost_sale_configs,
    perishable_configs,
)


ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"

DEFAULT_LAMBDA_BASES = [0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
DEFAULT_H = 7
DEFAULT_SEED_POOL_SIZE = 8
DEFAULT_ACTIVE_SEEDS = 8
TEST_REPEATS = 100

SYSTEM_ORDER = ["LS", "PS", "DS"]
SYSTEM_REGISTRY = {
    "LS": {
        "label": "Lost Sales",
        "env_cls": LostSalesInventory,
        "configs": lost_sale_configs,
    },
    "PS": {
        "label": "Perishable",
        "env_cls": PerishableInventory,
        "configs": perishable_configs,
    },
    "DS": {
        "label": "Dual Sourcing",
        "env_cls": DualSourcingInventory,
        "configs": dual_sourcing_configs,
    },
}


def build_seed_pool(n_seeds: int = DEFAULT_SEED_POOL_SIZE) -> list[int]:
    np.random.seed(0)
    seeds = np.random.choice(900, replace=False, size=n_seeds) + 100
    return [int(seed) for seed in seeds]


def build_task_specs() -> list[dict]:
    task_specs = []
    for system in SYSTEM_ORDER:
        meta = SYSTEM_REGISTRY[system]
        for task_index, config in enumerate(meta["configs"], start=1):
            task_specs.append(
                {
                    "system": system,
                    "system_label": meta["label"],
                    "task_index": int(task_index),
                    "task_name": f"{system}{task_index}",
                    "config": dict(config),
                }
            )
    return task_specs


def build_env(system: str, task_index: int):
    meta = SYSTEM_REGISTRY[system]
    config = dict(meta["configs"][task_index - 1])
    return meta["env_cls"](config)


def make_result_paths(n_active_seeds: int):
    stem = f"mdqn_lambda_sensitivity_seeds{n_active_seeds}"
    return (
        RESULTS_DIR / f"{stem}.json",
        RESULTS_DIR / f"{stem}.md",
        RESULTS_DIR / f"{stem}.progress.json",
    )


def format_float(value: float) -> str:
    return f"{value:.12g}"


def build_task_lambda_key(system: str, task_index: int, lambda_base: float) -> str:
    return f"{system}{task_index}|lambda_base={format_float(lambda_base)}"


def build_run_tasks(
    task_specs: list[dict],
    lambda_bases: list[float],
    seeds: list[int],
    H: int,
) -> list[dict]:
    tasks = []
    for task_spec in task_specs:
        for lambda_base in lambda_bases:
            lambda_anc = lambda_base / H
            for seed in seeds:
                tasks.append(
                    {
                        "system": task_spec["system"],
                        "system_label": task_spec["system_label"],
                        "task_index": int(task_spec["task_index"]),
                        "task_name": task_spec["task_name"],
                        "H": int(H),
                        "lambda_base": float(lambda_base),
                        "lambda_anc": float(lambda_anc),
                        "seed": int(seed),
                    }
                )
    return tasks


def run_single_seed(task: dict) -> dict:
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    env = build_env(task["system"], task["task_index"])
    result = train_test_MDQN(
        env=env,
        task_name=f"{task['task_name']}_H={task['H']}_lambda={task['lambda_anc']:g}",
        test_repeats=TEST_REPEATS,
        seed=task["seed"],
        H=task["H"],
        lambda_anc=task["lambda_anc"],
    )
    return {
        "system": task["system"],
        "system_label": task["system_label"],
        "task_index": int(task["task_index"]),
        "task_name": task["task_name"],
        "seed": int(task["seed"]),
        "H": int(task["H"]),
        "lambda_base": float(task["lambda_base"]),
        "lambda_anc": float(task["lambda_anc"]),
        "performance": float(result["performance"]),
        "step": int(result["step"]),
        "final_head": int(result["parameters"]["head"] + 1),
        "history_length": int(len(result["history"])),
    }


def summarize_records(records: list[dict]) -> dict:
    performances = np.asarray([record["performance"] for record in records], dtype=np.float64)
    steps = np.asarray([record["step"] for record in records], dtype=np.int64)
    heads = np.asarray([record["final_head"] for record in records], dtype=np.int64)
    return {
        "n_seeds": int(len(records)),
        "performance_mean": float(performances.mean()),
        "performance_std": float(performances.std(ddof=0)),
        "performance_min": float(performances.min()),
        "performance_max": float(performances.max()),
        "best_step_mean": float(steps.mean()),
        "best_step_min": int(steps.min()),
        "best_step_max": int(steps.max()),
        "final_head_mean": float(heads.mean()),
        "final_head_min": int(heads.min()),
        "final_head_max": int(heads.max()),
    }


def aggregate_task_results(
    records: list[dict],
    task_specs: list[dict],
    lambda_bases: list[float],
    H: int,
    expected_records_per_experiment: int,
) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        key = build_task_lambda_key(record["system"], record["task_index"], record["lambda_base"])
        grouped.setdefault(key, []).append(record)

    tasks = []
    for task_spec in task_specs:
        experiments = []
        for lambda_base in lambda_bases:
            key = build_task_lambda_key(task_spec["system"], task_spec["task_index"], lambda_base)
            config_records = sorted(grouped.get(key, []), key=lambda item: item["seed"])
            if not config_records:
                continue
            experiment = {
                "lambda_base": float(lambda_base),
                "lambda_anc": float(lambda_base / H),
                "records": config_records,
                "completed": len(config_records) == expected_records_per_experiment,
            }
            if experiment["completed"]:
                experiment["summary"] = summarize_records(config_records)
            experiments.append(experiment)

        task_entry = task_spec | {
            "H": int(H),
            "experiments": experiments,
            "completed_experiments": int(sum(1 for experiment in experiments if experiment["completed"])),
            "total_experiments": int(len(lambda_bases)),
        }
        completed_experiments = [experiment for experiment in experiments if experiment["completed"]]
        if completed_experiments:
            best = min(
                completed_experiments,
                key=lambda item: item["summary"]["performance_mean"],
            )
            task_entry["best_lambda"] = {
                "lambda_base": float(best["lambda_base"]),
                "lambda_anc": float(best["lambda_anc"]),
                "performance_mean": float(best["summary"]["performance_mean"]),
                "performance_std": float(best["summary"]["performance_std"]),
            }
        tasks.append(task_entry)
    return tasks


def annotate_task_rankings(tasks: list[dict]) -> None:
    for task in tasks:
        completed_experiments = [experiment for experiment in task["experiments"] if experiment["completed"]]
        if not completed_experiments:
            continue
        completed_experiments.sort(
            key=lambda item: (item["summary"]["performance_mean"], item["lambda_base"])
        )
        best_mean = completed_experiments[0]["summary"]["performance_mean"]
        for rank, experiment in enumerate(completed_experiments, start=1):
            experiment["rank"] = int(rank)
            experiment["gap_to_best"] = float(experiment["summary"]["performance_mean"] - best_mean)


def _find_experiment(task: dict, lambda_base: float) -> dict:
    key = format_float(lambda_base)
    for experiment in task["experiments"]:
        if experiment["completed"] and format_float(experiment["lambda_base"]) == key:
            return experiment
    raise KeyError(f"Missing completed result for {task['task_name']} and lambda_base={lambda_base}.")


def compute_overall_summary(tasks: list[dict], lambda_bases: list[float], H: int) -> dict:
    rows = []
    for lambda_base in lambda_bases:
        experiments = [_find_experiment(task, lambda_base) for task in tasks]
        win_count = sum(
            1
            for task in tasks
            if format_float(task["best_lambda"]["lambda_base"]) == format_float(lambda_base)
        )
        rows.append(
            {
                "lambda_base": float(lambda_base),
                "lambda_anc": float(lambda_base / H),
                "win_count": int(win_count),
                "average_rank": float(np.mean([experiment["rank"] for experiment in experiments])),
                "mean_gap_to_task_best": float(
                    np.mean([experiment["gap_to_best"] for experiment in experiments])
                ),
            }
        )

    sorted_rows = sorted(
        rows,
        key=lambda item: (
            item["average_rank"],
            item["mean_gap_to_task_best"],
            -item["win_count"],
            item["lambda_base"],
        ),
    )
    best_by_average_rank = dict(sorted_rows[0])
    best_by_win_count = dict(
        max(
            rows,
            key=lambda item: (
                item["win_count"],
                -item["average_rank"],
                -item["mean_gap_to_task_best"],
                -item["lambda_base"],
            ),
        )
    )
    return {
        "lambda_rows": rows,
        "best_by_average_rank": best_by_average_rank,
        "best_by_win_count": best_by_win_count,
    }


def compute_system_summaries(tasks: list[dict], lambda_bases: list[float], H: int) -> list[dict]:
    grouped_tasks: dict[str, list[dict]] = {}
    for task in tasks:
        grouped_tasks.setdefault(task["system"], []).append(task)

    summaries = []
    for system in SYSTEM_ORDER:
        system_tasks = grouped_tasks.get(system, [])
        if not system_tasks:
            continue
        lambda_rows = []
        for lambda_base in lambda_bases:
            experiments = [_find_experiment(task, lambda_base) for task in system_tasks]
            win_count = sum(
                1
                for task in system_tasks
                if format_float(task["best_lambda"]["lambda_base"]) == format_float(lambda_base)
            )
            lambda_rows.append(
                {
                    "lambda_base": float(lambda_base),
                    "lambda_anc": float(lambda_base / H),
                    "win_count": int(win_count),
                    "average_rank": float(np.mean([experiment["rank"] for experiment in experiments])),
                    "mean_gap_to_task_best": float(
                        np.mean([experiment["gap_to_best"] for experiment in experiments])
                    ),
                }
            )
        summaries.append(
            {
                "system": system,
                "system_label": system_tasks[0]["system_label"],
                "n_tasks": int(len(system_tasks)),
                "lambda_rows": lambda_rows,
                "best_by_average_rank": dict(
                    min(
                        lambda_rows,
                        key=lambda item: (
                            item["average_rank"],
                            item["mean_gap_to_task_best"],
                            -item["win_count"],
                            item["lambda_base"],
                        ),
                    )
                ),
                "best_by_win_count": dict(
                    max(
                        lambda_rows,
                        key=lambda item: (
                            item["win_count"],
                            -item["average_rank"],
                            -item["mean_gap_to_task_best"],
                            -item["lambda_base"],
                        ),
                    )
                ),
            }
        )
    return summaries


def write_progress_payload(
    progress_path: Path,
    records: list[dict],
    task_specs: list[dict],
    lambda_bases: list[float],
    H: int,
    seed_pool: list[int],
    active_seeds: list[int],
    n_jobs: int,
) -> None:
    tasks = aggregate_task_results(records, task_specs, lambda_bases, H, len(active_seeds))
    completed_task_lambda_configs = int(
        sum(task["completed_experiments"] for task in tasks)
    )
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "running",
        "experiment": {
            "name": "MDQN lambda-base sensitivity",
            "H": int(H),
            "test_repeats": TEST_REPEATS,
        },
        "seed_pool": seed_pool,
        "active_seeds": active_seeds,
        "lambda_bases": lambda_bases,
        "n_jobs": int(n_jobs),
        "task_count": int(len(task_specs)),
        "system_count": int(len(SYSTEM_ORDER)),
        "completed_runs": int(len(records)),
        "total_runs": int(len(task_specs) * len(lambda_bases) * len(active_seeds)),
        "completed_task_lambda_configs": completed_task_lambda_configs,
        "total_task_lambda_configs": int(len(task_specs) * len(lambda_bases)),
        "tasks": tasks,
    }
    progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_report(payload: dict) -> str:
    overall_rows = sorted(payload["overall_summary"]["lambda_rows"], key=lambda item: item["lambda_base"])
    lines = [
        "# MDQN Lambda-Base Sensitivity",
        "",
        f"- Generated at: `{payload['generated_at']}`",
        f"- Fixed H: `{payload['experiment']['H']}`",
        f"- Active seeds: `{payload['active_seeds']}`",
        f"- Lambda bases: `{payload['lambda_bases']}`",
        f"- Inventory systems: `{payload['system_count']}`",
        f"- Tasks: `{payload['task_count']}`",
        "",
        "## Overall Summary",
        "",
        "| lambda_base | lambda_anc | wins | average rank | mean gap to task best |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in overall_rows:
        lines.append(
            f"| {row['lambda_base']:g} | {row['lambda_anc']:.6f} | {row['win_count']} | "
            f"{row['average_rank']:.3f} | {row['mean_gap_to_task_best']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Best Lambda By Task",
            "",
            "| system | task | best lambda_base | best lambda_anc | mean perf | std perf |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for task in payload["tasks"]:
        best = task["best_lambda"]
        lines.append(
            f"| {task['system_label']} | {task['task_name']} | {best['lambda_base']:g} | "
            f"{best['lambda_anc']:.6f} | {best['performance_mean']:.4f} | {best['performance_std']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Full Task-Lambda Summary",
            "",
            "| system | task | lambda_base | lambda_anc | rank | mean perf | std perf | gap to task best |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for task in payload["tasks"]:
        experiments = sorted(task["experiments"], key=lambda item: item["lambda_base"])
        for experiment in experiments:
            summary = experiment["summary"]
            lines.append(
                f"| {task['system_label']} | {task['task_name']} | {experiment['lambda_base']:g} | "
                f"{experiment['lambda_anc']:.6f} | {experiment['rank']} | "
                f"{summary['performance_mean']:.4f} | {summary['performance_std']:.4f} | "
                f"{experiment['gap_to_best']:.4f} |"
            )

    lines.append("")
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the MDQN lambda-base sensitivity sweep across all 12 benchmark tasks."
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=DEFAULT_ACTIVE_SEEDS,
        help="Number of active seeds taken from the fixed seed pool.",
    )
    parser.add_argument(
        "--h",
        type=int,
        default=DEFAULT_H,
        help="Fixed H value used for every task in this sweep.",
    )
    parser.add_argument(
        "--h-values",
        type=int,
        nargs="+",
        default=None,
        help="Deprecated compatibility alias. If provided, it must contain exactly one H value.",
    )
    parser.add_argument(
        "--lambda-bases",
        type=float,
        nargs="+",
        default=DEFAULT_LAMBDA_BASES,
        help="Base values used to form lambda_anc = lambda_base / H.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Number of parallel workers. Default uses all logical CPU cores up to the task count.",
    )
    return parser.parse_args()


def create_executor(n_jobs: int):
    try:
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs)
        backend = "process"
    except PermissionError as exc:
        print(
            "ProcessPoolExecutor is unavailable in the current environment "
            f"({exc}); falling back to ThreadPoolExecutor."
        )
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=n_jobs)
        backend = "thread"
    return executor, backend


def main():
    args = parse_args()
    if args.n_seeds < 1 or args.n_seeds > DEFAULT_SEED_POOL_SIZE:
        raise ValueError(f"n_seeds must be in [1, {DEFAULT_SEED_POOL_SIZE}], got {args.n_seeds}.")

    if args.h_values is not None:
        if len(args.h_values) != 1:
            raise ValueError(
                f"This experiment now fixes H and only sweeps lambda_base, got h-values={args.h_values}."
            )
        H = int(args.h_values[0])
    else:
        H = int(args.h)
    if H < 1:
        raise ValueError(f"H must be a positive integer, got {H}.")

    lambda_bases = [float(value) for value in args.lambda_bases]
    if not lambda_bases:
        raise ValueError("At least one lambda base is required.")
    if min(lambda_bases) < 0:
        raise ValueError(f"Lambda bases must be non-negative, got {lambda_bases}.")

    task_specs = build_task_specs()
    seed_pool = build_seed_pool(DEFAULT_SEED_POOL_SIZE)
    active_seeds = seed_pool[:args.n_seeds]
    run_tasks = build_run_tasks(task_specs, lambda_bases, active_seeds, H)
    n_jobs = args.n_jobs
    if n_jobs is None:
        n_jobs = min(len(run_tasks), os.cpu_count() or 1)
    if n_jobs < 1:
        raise ValueError(f"n_jobs must be positive, got {n_jobs}.")
    n_jobs = min(n_jobs, len(run_tasks))

    json_path, report_path, progress_path = make_result_paths(args.n_seeds)
    all_records = []
    completed_task_lambda_keys = set()

    print(f"Running MDQN lambda-base sensitivity with fixed H={H}")
    print(f"Active seeds: {active_seeds}")
    print(f"Lambda bases: {lambda_bases}")
    print(f"Inventory systems: {len(SYSTEM_ORDER)}, tasks: {len(task_specs)}")
    print(f"Parallel workers: {n_jobs}")
    print(
        "Total runs: "
        f"{len(run_tasks)} across {len(task_specs) * len(lambda_bases)} task-lambda configurations"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_progress_payload(
        progress_path,
        all_records,
        task_specs,
        lambda_bases,
        H,
        seed_pool,
        active_seeds,
        n_jobs,
    )

    executor, executor_backend = create_executor(n_jobs)
    print(f"Executor backend: {executor_backend}")
    with executor as executor:
        future_to_task = {executor.submit(run_single_seed, task): task for task in run_tasks}
        for index, future in enumerate(concurrent.futures.as_completed(future_to_task), start=1):
            task = future_to_task[future]
            try:
                record = future.result()
            except Exception as exc:
                print(
                    f"[failed] {task['task_name']}, lambda_base={task['lambda_base']:g}, "
                    f"lambda_anc={task['lambda_anc']:.6f}, seed={task['seed']}: {exc}"
                )
                raise
            all_records.append(record)
            print(
                f"[run {index}/{len(run_tasks)}] "
                f"{record['task_name']}, lambda_base={record['lambda_base']:g}, "
                f"lambda_anc={record['lambda_anc']:.6f}, seed={record['seed']}, "
                f"performance={record['performance']:.4f}, step={record['step']}, head={record['final_head']}"
            )
            task_lambda_key = build_task_lambda_key(
                record["system"], record["task_index"], record["lambda_base"]
            )
            config_records = [
                item
                for item in all_records
                if build_task_lambda_key(item["system"], item["task_index"], item["lambda_base"])
                == task_lambda_key
            ]
            if (
                len(config_records) == len(active_seeds)
                and task_lambda_key not in completed_task_lambda_keys
            ):
                completed_task_lambda_keys.add(task_lambda_key)
                summary = summarize_records(sorted(config_records, key=lambda item: item["seed"]))
                print(
                    f"[config {len(completed_task_lambda_keys)}/{len(task_specs) * len(lambda_bases)}] "
                    f"{record['task_name']}, lambda_base={record['lambda_base']:g}, "
                    f"mean={summary['performance_mean']:.4f}, std={summary['performance_std']:.4f}"
                )
            write_progress_payload(
                progress_path,
                all_records,
                task_specs,
                lambda_bases,
                H,
                seed_pool,
                active_seeds,
                n_jobs,
            )

    tasks = aggregate_task_results(all_records, task_specs, lambda_bases, H, len(active_seeds))
    tasks = [task for task in tasks if task["completed_experiments"] == len(lambda_bases)]
    if len(tasks) != len(task_specs):
        raise RuntimeError(
            f"Expected {len(task_specs)} completed tasks, but only found {len(tasks)}."
        )
    annotate_task_rankings(tasks)
    for task in tasks:
        task["experiments"].sort(key=lambda item: item["lambda_base"])

    overall_summary = compute_overall_summary(tasks, lambda_bases, H)
    system_summaries = compute_system_summaries(tasks, lambda_bases, H)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "status": "completed",
        "experiment": {
            "name": "MDQN lambda-base sensitivity",
            "H": int(H),
            "test_repeats": TEST_REPEATS,
        },
        "seed_pool": seed_pool,
        "active_seeds": active_seeds,
        "lambda_bases": lambda_bases,
        "n_jobs": int(n_jobs),
        "task_count": int(len(tasks)),
        "system_count": int(len(SYSTEM_ORDER)),
        "tasks": tasks,
        "overall_summary": overall_summary,
        "system_summaries": system_summaries,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    report_path.write_text(build_report(payload), encoding="utf-8")
    progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved JSON to {json_path}")
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()

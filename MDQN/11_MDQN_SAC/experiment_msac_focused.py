import argparse
import csv
import json
import os
import pickle
from pathlib import Path
from time import perf_counter

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed

from simulators import LostSalesInventory, lost_sale_configs
from simulators import PerishableInventory, perishable_configs
from simulators import DualSourcingInventory, dual_sourcing_configs
from DRL import train_test_SAC, train_test_MSAC


SEEDS = [592, 241, 509, 131, 670, 693, 973, 499]
METHODS = ["SAC", "MSAC"]
OUTPUT_DIR = Path("results") / "final_report"
RAW_RESULTS_FILE = OUTPUT_DIR / "msac_focused_raw_results.csv"
STATS_FILE = OUTPUT_DIR / "msac_focused_stats.json"
FIGURE_FILE = OUTPUT_DIR / "msac_final_cost_scatter.pdf"
HEURISTIC_RESULTS_FILE = Path("results") / "heuristic_results.pkl"

TASK_SPECS = {
    "LS1": {
        "system": "LS",
        "task": "1",
        "family": "Lost sales",
        "env_class": LostSalesInventory,
        "config": lost_sale_configs[0],
        "heuristic_key": "CBS",
        "heuristic_label": "CBS",
    },
    "PS1": {
        "system": "PS",
        "task": "1",
        "family": "Perishable inventory",
        "env_class": PerishableInventory,
        "config": perishable_configs[0],
        "heuristic_key": "BSLEW",
        "heuristic_label": "BSEW",
    },
    "DS1": {
        "system": "DS",
        "task": "1",
        "family": "Dual sourcing",
        "env_class": DualSourcingInventory,
        "config": dual_sourcing_configs[0],
        "heuristic_key": "CDI",
        "heuristic_label": "CDI",
    },
}


def _build_env(task_name):
    spec = TASK_SPECS[task_name]
    return spec["env_class"](spec["config"])


def _run_job(task_name, method, seed):
    env = _build_env(task_name)
    train_fn = train_test_MSAC if method == "MSAC" else train_test_SAC
    start = perf_counter()
    result = train_fn(
        DRL_method=method,
        env=env,
        task_name=task_name,
        seed=seed,
        test_repeats=100,
        is_print=False,
    )
    runtime_sec = perf_counter() - start
    selected_head = ""
    if method == "MSAC":
        selected_head = int(result["parameters"]["head"]) + 1
    spec = TASK_SPECS[task_name]
    return {
        "system": spec["system"],
        "task": task_name,
        "task_id": spec["task"],
        "method": method,
        "seed": int(seed),
        "performance": float(result["performance"]),
        "best_step": int(result["step"]),
        "selected_head": selected_head,
        "runtime_sec": float(runtime_sec),
    }


def _load_heuristics():
    with HEURISTIC_RESULTS_FILE.open("rb") as file:
        heuristic_results = pickle.load(file)
    heuristics = {}
    for task_name, spec in TASK_SPECS.items():
        method = spec["heuristic_key"]
        log = heuristic_results[spec["system"]][spec["task"]][method]
        heuristics[task_name] = {
            "method": method,
            "performance": float(log["performance"]),
        }
    return heuristics


def _aggregate(rows, elapsed_wall_sec):
    heuristics = _load_heuristics()
    by_task = {}
    runtime_by_method = {}
    for task_name in TASK_SPECS:
        by_task[task_name] = {
            "family": TASK_SPECS[task_name]["family"],
            "heuristic": heuristics[task_name],
            "methods": {},
        }
        for method in METHODS:
            method_rows = [
                row for row in rows
                if row["task"] == task_name and row["method"] == method
            ]
            performances = np.array([row["performance"] for row in method_rows], dtype=np.float64)
            by_task[task_name]["methods"][method] = {
                "n": int(performances.size),
                "mean": float(np.mean(performances)),
                "std": float(np.std(performances, ddof=0)),
                "min": float(np.min(performances)),
                "max": float(np.max(performances)),
                "best_step_mean": float(np.mean([row["best_step"] for row in method_rows])),
                "runtime_sec_sum": float(np.sum([row["runtime_sec"] for row in method_rows])),
                "runtime_sec_mean": float(np.mean([row["runtime_sec"] for row in method_rows])),
            }
            if method == "MSAC":
                heads = [int(row["selected_head"]) for row in method_rows]
                by_task[task_name]["methods"][method]["selected_heads"] = heads
        by_task[task_name]["msac_minus_sac_mean"] = (
            by_task[task_name]["methods"]["MSAC"]["mean"]
            - by_task[task_name]["methods"]["SAC"]["mean"]
        )
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        runtime_by_method[method] = {
            "jobs": len(method_rows),
            "runtime_sec_sum": float(np.sum([row["runtime_sec"] for row in method_rows])),
            "runtime_sec_mean": float(np.mean([row["runtime_sec"] for row in method_rows])),
            "runtime_sec_max": float(np.max([row["runtime_sec"] for row in method_rows])),
        }
    return {
        "seeds": SEEDS,
        "methods": METHODS,
        "tasks": list(TASK_SPECS.keys()),
        "test_repeats": 100,
        "std_definition": "population",
        "elapsed_wall_sec": float(elapsed_wall_sec),
        "runtime_by_method": runtime_by_method,
        "by_task": by_task,
    }


def _write_raw(rows):
    RAW_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "system",
        "task",
        "task_id",
        "method",
        "seed",
        "performance",
        "best_step",
        "selected_head",
        "runtime_sec",
    ]
    with RAW_RESULTS_FILE.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_stats(stats):
    STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STATS_FILE.open("w") as file:
        json.dump(stats, file, indent=2)


def _read_existing_results():
    with RAW_RESULTS_FILE.open(newline="") as file:
        rows = list(csv.DictReader(file))
    with STATS_FILE.open() as file:
        stats = json.load(file)
    return rows, stats


def _plot_final_cost_scatter(rows, stats):
    task_order = ["LS1", "PS1", "DS1"]
    colors = {"SAC": "#bf6847", "MSAC": "#357a96"}
    markers = {"SAC": "o", "MSAC": "s"}
    jitter_map = {
        str(seed): offset
        for seed, offset in zip(
            stats["seeds"],
            [-0.09, -0.065, -0.04, -0.015, 0.015, 0.04, 0.065, 0.09],
        )
    }

    figure, axes = plt.subplots(1, 3, figsize=(7.6, 2.45), sharey=False)
    for axis, task_name in zip(axes, task_order):
        task_rows = [row for row in rows if row["task"] == task_name]
        heuristic = stats["by_task"][task_name]["heuristic"]
        heuristic_cost = float(heuristic["performance"])
        heuristic_label = TASK_SPECS[task_name]["heuristic_label"]

        y_values = [float(row["performance"]) for row in task_rows]
        y_values.append(heuristic_cost)
        y_min, y_max = min(y_values), max(y_values)
        padding = max((y_max - y_min) * 0.22, 0.35)
        axis.set_ylim(y_min - padding * 0.65, y_max + padding)

        for x_position, method in enumerate(METHODS):
            method_rows = [row for row in task_rows if row["method"] == method]
            axis.scatter(
                [x_position + jitter_map.get(str(row["seed"]), 0.0) for row in method_rows],
                [float(row["performance"]) for row in method_rows],
                s=22,
                color=colors[method],
                marker=markers[method],
                alpha=0.95,
                edgecolors="none",
                zorder=3,
            )

            method_stats = stats["by_task"][task_name]["methods"][method]
            mean = float(method_stats["mean"])
            std = float(method_stats["std"])
            axis.errorbar(
                [x_position],
                [mean],
                yerr=[[std], [std]],
                fmt="D",
                markersize=4.8,
                markerfacecolor="white",
                markeredgecolor="black",
                markeredgewidth=0.85,
                ecolor="black",
                elinewidth=0.9,
                capsize=3.0,
                capthick=0.9,
                zorder=4,
            )

        axis.axhline(
            heuristic_cost,
            linestyle=(0, (4, 3)),
            linewidth=0.8,
            color="#555555",
            zorder=1,
        )
        y_lower, y_upper = axis.get_ylim()
        axis.text(
            0.5,
            heuristic_cost + 0.024 * (y_upper - y_lower),
            heuristic_label,
            fontsize=7.1,
            color="#555555",
            ha="center",
            va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4},
            zorder=5,
        )

        axis.set_title(
            {"LS1": "(a) LS1", "PS1": "(b) PS1", "DS1": "(c) DS1"}[task_name],
            fontsize=9.5,
            pad=4,
        )
        axis.set_xticks([0, 1])
        axis.set_xticklabels(METHODS, fontsize=8.5)
        axis.tick_params(axis="y", labelsize=8)
        axis.grid(axis="y", color="#d7d7d7", linewidth=0.45, alpha=0.9)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_linewidth(0.8)
        axis.spines["bottom"].set_linewidth(0.8)

    axes[0].set_ylabel("Final test cost", fontsize=9)
    figure.tight_layout(w_pad=1.0)
    FIGURE_FILE.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FIGURE_FILE, bbox_inches="tight")
    plt.close(figure)


def _build_jobs():
    return [
        (task_name, method, seed)
        for task_name in TASK_SPECS
        for method in METHODS
        for seed in SEEDS
    ]


def run(n_jobs):
    jobs = _build_jobs()
    start = perf_counter()
    rows = Parallel(n_jobs=n_jobs, backend="loky", batch_size="auto")(
        delayed(_run_job)(task_name, method, seed)
        for task_name, method, seed in jobs
    )
    elapsed_wall_sec = perf_counter() - start
    rows = sorted(rows, key=lambda row: (row["system"], row["task_id"], row["method"], SEEDS.index(row["seed"])))
    stats = _aggregate(rows, elapsed_wall_sec)
    _write_raw(rows)
    _write_stats(stats)
    return rows, stats


def main():
    parser = argparse.ArgumentParser(description="Run the focused SAC versus MSAC Task-1 comparison.")
    parser.add_argument("--n-jobs", type=int, default=18, help="Number of parallel workers.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Print the planned jobs without training.")
    mode.add_argument(
        "--plot-only",
        action="store_true",
        help="Rebuild the final-cost scatter from the existing raw results and statistics.",
    )
    args = parser.parse_args()
    if args.dry_run:
        jobs = _build_jobs()
        for task_name, method, seed in jobs:
            print(f"{task_name},{method},{seed}")
        print(f"planned_jobs={len(jobs)}")
        return
    if args.plot_only:
        rows, stats = _read_existing_results()
        _plot_final_cost_scatter(rows, stats)
        print(f"wrote figure to {FIGURE_FILE}")
        return
    rows, stats = run(args.n_jobs)
    _plot_final_cost_scatter(rows, stats)
    print(f"wrote {len(rows)} rows to {RAW_RESULTS_FILE}")
    print(f"wrote stats to {STATS_FILE}")
    print(f"wrote figure to {FIGURE_FILE}")
    print(f"elapsed_wall_sec={stats['elapsed_wall_sec']:.1f}")


if __name__ == "__main__":
    main()

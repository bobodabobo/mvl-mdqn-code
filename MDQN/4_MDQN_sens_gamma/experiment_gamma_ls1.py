import os

# Keep each worker single-threaded so process-level parallelism can scale.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import csv
import json
import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np

from DRL import train_test_DQN, train_test_MDQN
from DRL.configs import DQN_config, MDQN_config
from heuristic import CappedBaseStock
from simulators import LostSalesInventory, lost_sale_configs


GAMMAS = [0.1, 0.2, 0.5, 0.8, 0.9, 0.99, 1.0]
RL_SEEDS = [592, 241, 509, 131, 670, 693, 973, 499]
RL_METHODS = ("TDQN", "MDQN")

RAW_FIELDNAMES = [
    "method",
    "gamma",
    "seed",
    "performance",
    "mean_cost",
    "mean_assigned_cost",
    "raw_discounted_cost",
    "step",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LS1 gamma sensitivity with DCA-consistent discounted evaluation."
    )
    parser.add_argument("--results-dir", default="results/gamma_ls1_dca_eval")
    parser.add_argument("--tdqn-train-steps", type=int, default=None)
    parser.add_argument("--mdqn-train-steps", type=int, default=None)
    parser.add_argument("--eval-times", type=int, default=DQN_config["eval_times"])
    parser.add_argument("--n-repeats-eval", type=int, default=DQN_config["n_repeats_eval"])
    parser.add_argument("--test-repeats", type=int, default=DQN_config["test_repeats"])
    parser.add_argument("--heuristic-repeats", type=int, default=1)
    parser.add_argument("--heuristic-n-jobs", type=int, default=-1)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def gamma_key(gamma: float):
    return f"{gamma:.2f}".rstrip("0").rstrip(".")


def solve_time(seconds: float):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours}h{minutes}m{seconds}s"


def configure_worker_threads():
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def _empty_payload(args, max_workers: int):
    env = LostSalesInventory(lost_sale_configs[0])
    return {
        "metadata": {
            "task": "LS1",
            "methods": ["CBS", "TDQN", "MDQN"],
            "metric": "dca_discounted_cost",
            "evaluation_notes": (
                "For gamma < 1, discounted evaluation applies delayed cost assignment "
                "before temporal discounting, matching the MDQN reward construction. "
                "For gamma = 1, performance falls back to mean total cost."
            ),
            "gammas": GAMMAS,
            "rl_seeds": RL_SEEDS,
            "heuristic_repeats": args.heuristic_repeats,
            "test_repeats": args.test_repeats,
            "eval_times": args.eval_times,
            "n_repeats_eval": args.n_repeats_eval,
            "tdqn_train_steps": args.tdqn_train_steps or DQN_config["train_steps"],
            "mdqn_train_steps": args.mdqn_train_steps or MDQN_config["train_steps"],
            "reward_delay_time": env.reward_delay_time,
            "lead_time": env.lead_time,
            "max_workers": max_workers,
        },
        "results": {
            "CBS": {},
            "TDQN": {},
            "MDQN": {},
        },
    }


def load_payload(results_dir: Path):
    path = results_dir / "results.pkl"
    if not path.exists():
        return None
    with path.open("rb") as file:
        return pickle.load(file)


def save_pickle(results_dir: Path, payload):
    with (results_dir / "results.pkl").open("wb") as file:
        pickle.dump(payload, file)


def build_summary_rows(payload):
    rows = []
    for gamma in payload["metadata"]["gammas"]:
        key = gamma_key(gamma)
        cbs_log = payload["results"]["CBS"].get(key)
        if cbs_log is not None:
            rows.append(
                {
                    "method": "CBS",
                    "gamma": gamma,
                    "n_runs": len(cbs_log["eval_seeds"]),
                    "mean_performance": cbs_log["performance"],
                    "std_performance": 0.0,
                    "mean_cost": cbs_log["mean_cost"],
                    "mean_assigned_cost": cbs_log["mean_assigned_cost"],
                    "raw_discounted_cost": cbs_log["raw_discounted_cost"],
                }
            )
        for method in RL_METHODS:
            logs = payload["results"][method].get(key)
            if not logs:
                continue
            rows.append(
                {
                    "method": method,
                    "gamma": gamma,
                    "n_runs": len(logs),
                    "mean_performance": float(np.mean([item["performance"] for item in logs])),
                    "std_performance": float(np.std([item["performance"] for item in logs])),
                    "mean_cost": float(np.mean([item["mean_cost"] for item in logs])),
                    "mean_assigned_cost": float(np.mean([item["mean_assigned_cost"] for item in logs])),
                    "raw_discounted_cost": float(np.mean([item["raw_discounted_cost"] for item in logs])),
                }
            )
    return rows


def save_summary(results_dir: Path, payload):
    rows = build_summary_rows(payload)
    fieldnames = [
        "method",
        "gamma",
        "n_runs",
        "mean_performance",
        "std_performance",
        "mean_cost",
        "mean_assigned_cost",
        "raw_discounted_cost",
    ]
    with (results_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_raw_results(results_dir: Path, payload):
    rows = []
    for method in RL_METHODS:
        for gamma in payload["metadata"]["gammas"]:
            key = gamma_key(gamma)
            for item in payload["results"][method].get(key, []):
                rows.append(
                    {
                        "method": method,
                        "gamma": gamma,
                        "seed": item["seed"],
                        "performance": item["performance"],
                        "mean_cost": item["mean_cost"],
                        "mean_assigned_cost": item["mean_assigned_cost"],
                        "raw_discounted_cost": item["raw_discounted_cost"],
                        "step": item["step"],
                    }
                )
    rows = sorted(rows, key=lambda item: (item["method"], item["gamma"], item["seed"]))
    with (results_dir / "raw_results.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RAW_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def save_metadata(results_dir: Path, payload):
    with (results_dir / "run_metadata.json").open("w") as file:
        json.dump(payload["metadata"], file, indent=2)


def run_single_rl_job(job: dict):
    configure_worker_threads()
    env = LostSalesInventory(lost_sale_configs[0])
    gamma = job["gamma"]
    if job["method"] == "TDQN":
        log = train_test_DQN(
            DRL_method="TDQN",
            env=env,
            task_name=job["task_name"],
            seed=job["seed"],
            gamma=gamma,
            gamma_eval=gamma,
            train_steps=job["train_steps"],
            eval_times=job["eval_times"],
            n_repeats_eval=job["n_repeats_eval"],
            test_repeats=job["test_repeats"],
        )
    else:
        log = train_test_MDQN(
            env=env,
            task_name=job["task_name"],
            seed=job["seed"],
            gamma=gamma,
            gamma_eval=gamma,
            train_steps=job["train_steps"],
            eval_times=job["eval_times"],
            n_repeats_eval=job["n_repeats_eval"],
            test_repeats=job["test_repeats"],
        )
    return {
        "method": job["method"],
        "gamma": gamma,
        "seed": job["seed"],
        "log": log,
    }


def build_rl_jobs(args, payload):
    jobs = []
    existing = {
        (method, gamma_key(gamma), item["seed"])
        for method in RL_METHODS
        for gamma in payload["metadata"]["gammas"]
        for item in payload["results"][method].get(gamma_key(gamma), [])
    }
    for gamma in GAMMAS:
        key = gamma_key(gamma)
        for seed in RL_SEEDS:
            tdqn_job = (
                "TDQN",
                key,
                seed,
            )
            if tdqn_job not in existing:
                jobs.append(
                    {
                        "method": "TDQN",
                        "gamma": gamma,
                        "seed": seed,
                        "task_name": f"LS1_gamma_{key}",
                        "train_steps": args.tdqn_train_steps or DQN_config["train_steps"],
                        "eval_times": args.eval_times,
                        "n_repeats_eval": args.n_repeats_eval,
                        "test_repeats": args.test_repeats,
                    }
                )
            mdqn_job = (
                "MDQN",
                key,
                seed,
            )
            if mdqn_job not in existing:
                jobs.append(
                    {
                        "method": "MDQN",
                        "gamma": gamma,
                        "seed": seed,
                        "task_name": f"LS1_gamma_{key}",
                        "train_steps": args.mdqn_train_steps or MDQN_config["train_steps"],
                        "eval_times": args.eval_times,
                        "n_repeats_eval": args.n_repeats_eval,
                        "test_repeats": args.test_repeats,
                    }
                )
    return jobs


def run_heuristic_sweep(args, payload):
    results_dir = Path(args.results_dir)
    for gamma in GAMMAS:
        key = gamma_key(gamma)
        if key in payload["results"]["CBS"]:
            continue
        env = LostSalesInventory(lost_sale_configs[0])
        print(f"[gamma={key}] CBS search")
        cbs_agent = CappedBaseStock(env)
        cbs_log = cbs_agent.train(
            length=env.max_steps,
            repeats=args.heuristic_repeats,
            gamma=gamma,
            seeds=RL_SEEDS,
            n_jobs=args.heuristic_n_jobs,
        )
        payload["results"]["CBS"][key] = cbs_log
        save_pickle(results_dir, payload)
        save_summary(results_dir, payload)
        save_raw_results(results_dir, payload)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    if args.force and results_dir.exists():
        for path in results_dir.iterdir():
            if path.is_file():
                path.unlink()
    results_dir.mkdir(parents=True, exist_ok=True)

    default_workers = max(1, (os.cpu_count() or 1) - 1)
    max_workers = args.max_workers or default_workers

    payload = load_payload(results_dir)
    if payload is None or args.force:
        payload = _empty_payload(args, max_workers)
    else:
        payload["metadata"]["max_workers"] = max_workers
    save_metadata(results_dir, payload)

    run_heuristic_sweep(args, payload)

    jobs = build_rl_jobs(args, payload)
    if not jobs:
        save_pickle(results_dir, payload)
        save_summary(results_dir, payload)
        save_raw_results(results_dir, payload)
        print(f"No pending RL jobs. Rebuilt outputs in {results_dir}.")
        return

    max_workers = min(max_workers, len(jobs))
    print(f"Launching {len(jobs)} RL jobs with max_workers={max_workers}.")
    start_time = perf_counter()
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        future_to_job = {executor.submit(run_single_rl_job, job): job for job in jobs}
        for idx, future in enumerate(as_completed(future_to_job), start=1):
            job = future_to_job[future]
            result = future.result()
            key = gamma_key(result["gamma"])
            payload["results"][result["method"]].setdefault(key, [])
            payload["results"][result["method"]][key].append(result["log"])
            payload["results"][result["method"]][key] = sorted(
                payload["results"][result["method"]][key],
                key=lambda item: item["seed"],
            )
            save_pickle(results_dir, payload)
            save_summary(results_dir, payload)
            save_raw_results(results_dir, payload)

            elapsed = perf_counter() - start_time
            avg_time = elapsed / idx
            remaining = len(jobs) - idx
            eta = avg_time * remaining
            print(
                f"[{idx}/{len(jobs)}] {job['method']} gamma={job['gamma']:g} seed={job['seed']} "
                f"-> perf={result['log']['performance']:.4f}, mean_cost={result['log']['mean_cost']:.4f}, "
                f"assigned_mean={result['log']['mean_assigned_cost']:.4f}, step={result['log']['step']}, "
                f"elapsed={solve_time(elapsed)}, eta={solve_time(eta)}"
            )

    save_metadata(results_dir, payload)
    save_pickle(results_dir, payload)
    save_summary(results_dir, payload)
    save_raw_results(results_dir, payload)
    total_time = perf_counter() - start_time
    print(f"Finished all RL jobs in {solve_time(total_time)}.")
    print(f"Saved outputs to {results_dir}")


if __name__ == "__main__":
    main()

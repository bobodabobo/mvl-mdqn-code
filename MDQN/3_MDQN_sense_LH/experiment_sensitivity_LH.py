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
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy as copy
from pathlib import Path
from statistics import mean, variance
from time import perf_counter

from openpyxl import Workbook

from DRL import train_test_MDQN
from simulators import LostSalesInventory, lost_sale_configs


DEFAULT_LEAD_TIMES = [1, 2, 3, 4, 5, 6]
DEFAULT_HORIZONS = [1, 2, 3, 4, 5, 6, 7]
DEFAULT_SEEDS = [131, 160, 241, 372, 412, 445, 499, 506]

METHOD_SPECS = {
    "MDQN": {
        "label": "MDQN",
        "use_delayed_cost_assignment": True,
    },
    "MDQN_no_DCA": {
        "label": "MDQN w/o DCA",
        "use_delayed_cost_assignment": False,
    },
}

RAW_FIELDNAMES = [
    "method_id",
    "method_label",
    "use_delayed_cost_assignment",
    "lead_time",
    "H",
    "seed",
    "performance",
    "best_step",
    "selected_head",
]

TABLE_METHOD_ORDER = ["MDQN", "MDQN_no_DCA"]


def parse_int_list(text:str):
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_method_list(text:str):
    method_ids = [item.strip() for item in text.split(",") if item.strip()]
    invalid_ids = [method_id for method_id in method_ids if method_id not in METHOD_SPECS]
    if invalid_ids:
        raise ValueError(f"Invalid method ids: {invalid_ids}. Valid ids: {list(METHOD_SPECS)}")
    return method_ids


def solve_time(seconds:float):
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours}h{minutes}m{seconds}s"


def build_ls1_config(lead_time:int):
    config = copy(lost_sale_configs[0])
    config["lead_time"] = int(lead_time)
    return config


def job_key(row:dict):
    return (row["method_id"], row["lead_time"], row["H"], row["seed"])


def configure_worker_threads():
    try:
        import torch

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def run_single_job(job:dict):
    configure_worker_threads()
    method_spec = METHOD_SPECS[job["method_id"]]
    env = LostSalesInventory(build_ls1_config(job["lead_time"]))
    result = train_test_MDQN(
        env=env,
        task_name=f"LS1_L{job['lead_time']}_H{job['H']}_{job['method_id']}",
        seed=job["seed"],
        H=job["H"],
        test_repeats=job["test_repeats"],
        use_delayed_cost_assignment=method_spec["use_delayed_cost_assignment"],
        include_weights_in_log=False,
        is_print=False,
        **job["train_overrides"],
    )
    return {
        "method_id": job["method_id"],
        "method_label": method_spec["label"],
        "use_delayed_cost_assignment": int(method_spec["use_delayed_cost_assignment"]),
        "lead_time": int(job["lead_time"]),
        "H": int(job["H"]),
        "seed": int(job["seed"]),
        "performance": float(result["performance"]),
        "best_step": int(result["step"]),
        "selected_head": int(result["selected_head"]),
    }


def load_existing_results(path:Path):
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                {
                    "method_id": row["method_id"],
                    "method_label": row["method_label"],
                    "use_delayed_cost_assignment": int(row["use_delayed_cost_assignment"]),
                    "lead_time": int(row["lead_time"]),
                    "H": int(row["H"]),
                    "seed": int(row["seed"]),
                    "performance": float(row["performance"]),
                    "best_step": int(row["best_step"]),
                    "selected_head": int(row["selected_head"]),
                }
            )
    return rows


def sort_records(records:list[dict]):
    return sorted(records, key=lambda row: (TABLE_METHOD_ORDER.index(row["method_id"]), row["lead_time"], row["H"], row["seed"]))


def write_raw_results(path:Path, records:list[dict]):
    records = sort_records(records)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RAW_FIELDNAMES)
        writer.writeheader()
        writer.writerows(records)


def build_summary_stats(records:list[dict]):
    summary = {}
    for row in records:
        key = (row["method_id"], row["lead_time"], row["H"])
        summary.setdefault(key, []).append(row["performance"])
    return summary


def format_table_cell(values:list[float]):
    if not values:
        return ""
    avg = mean(values)
    if len(values) < 2:
        return f"{avg:.4f}(NA)"
    return f"{avg:.4f}({variance(values):.4f})"


def write_summary_stats(path:Path, records:list[dict], lead_times:list[int], horizons:list[int]):
    summary = build_summary_stats(records)
    with path.open("w", newline="") as file:
        fieldnames = ["method_id", "method_label", "lead_time", "H", "count", "mean", "variance"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for method_id in TABLE_METHOD_ORDER:
            method_label = METHOD_SPECS[method_id]["label"]
            for lead_time in lead_times:
                for horizon in horizons:
                    values = summary.get((method_id, lead_time, horizon), [])
                    if not values:
                        continue
                    avg = mean(values)
                    var_value = variance(values) if len(values) >= 2 else ""
                    writer.writerow(
                        {
                            "method_id": method_id,
                            "method_label": method_label,
                            "lead_time": lead_time,
                            "H": horizon,
                            "count": len(values),
                            "mean": f"{avg:.6f}",
                            "variance": f"{var_value:.6f}" if var_value != "" else "",
                        }
                    )


def build_table_rows(records:list[dict], method_id:str, lead_times:list[int], horizons:list[int]):
    summary = build_summary_stats(records)
    rows = [["L\\H"] + horizons]
    for lead_time in lead_times:
        row = [lead_time]
        for horizon in horizons:
            values = summary.get((method_id, lead_time, horizon), [])
            row.append(format_table_cell(values))
        rows.append(row)
    return rows


def write_table_csv(path:Path, rows:list[list]):
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def write_table_markdown(path:Path, method_tables:dict[str, list[list]]):
    lines = ["# LH Sensitivity Tables", ""]
    for method_id in TABLE_METHOD_ORDER:
        if method_id not in method_tables:
            continue
        lines.append(f"## {METHOD_SPECS[method_id]['label']}")
        rows = method_tables[method_id]
        header = rows[0]
        lines.append("| " + " | ".join(map(str, header)) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for row in rows[1:]:
            lines.append("| " + " | ".join(map(str, row)) + " |")
        lines.append("")
    path.write_text("\n".join(lines))


def write_table_workbook(path:Path, method_tables:dict[str, list[list]]):
    workbook = Workbook()
    default_sheet = workbook.active
    workbook.remove(default_sheet)
    for method_id in TABLE_METHOD_ORDER:
        if method_id not in method_tables:
            continue
        worksheet = workbook.create_sheet(title=method_id)
        for row in method_tables[method_id]:
            worksheet.append(row)
    workbook.save(path)


def write_summary_outputs(output_dir:Path, records:list[dict], lead_times:list[int], horizons:list[int]):
    raw_results_path = output_dir / "raw_results.csv"
    summary_stats_path = output_dir / "summary_stats.csv"
    table_md_path = output_dir / "sensitivity_tables.md"
    workbook_path = output_dir / "sensitivity_tables.xlsx"
    method_tables = {
        method_id: build_table_rows(records, method_id, lead_times, horizons)
        for method_id in TABLE_METHOD_ORDER
    }
    write_raw_results(raw_results_path, records)
    write_summary_stats(summary_stats_path, records, lead_times, horizons)
    write_table_markdown(table_md_path, method_tables)
    write_table_workbook(workbook_path, method_tables)
    write_table_csv(output_dir / "table_MDQN.csv", method_tables["MDQN"])
    write_table_csv(output_dir / "table_MDQN_no_DCA.csv", method_tables["MDQN_no_DCA"])


def write_metadata(
    path:Path,
    args:argparse.Namespace,
    max_workers:int,
    n_jobs_total:int,
    n_jobs_existing:int,
    n_jobs_pending:int,
):
    metadata = {
        "lead_times": args.lead_times,
        "horizons": args.horizons,
        "seeds": args.seeds,
        "methods": args.methods,
        "output_dir": str(args.output_dir),
        "max_workers": max_workers,
        "test_repeats": args.test_repeats,
        "train_overrides": build_train_overrides(args),
        "n_jobs_total": n_jobs_total,
        "n_jobs_existing": n_jobs_existing,
        "n_jobs_pending": n_jobs_pending,
    }
    path.write_text(json.dumps(metadata, indent=2))


def build_train_overrides(args:argparse.Namespace):
    overrides = {}
    if args.train_steps is not None:
        overrides["train_steps"] = args.train_steps
    if args.eval_times is not None:
        overrides["eval_times"] = args.eval_times
    if args.train_repeats is not None:
        overrides["train_repeats"] = args.train_repeats
    if args.len_epi_train is not None:
        overrides["len_epi_train"] = args.len_epi_train
    if args.len_epi_eval is not None:
        overrides["len_epi_eval"] = args.len_epi_eval
    if args.n_repeats_eval is not None:
        overrides["n_repeats_eval"] = args.n_repeats_eval
    return overrides


def build_jobs(args:argparse.Namespace):
    train_overrides = build_train_overrides(args)
    jobs = []
    for method_id in args.methods:
        for lead_time in args.lead_times:
            for horizon in args.horizons:
                for seed in args.seeds:
                    jobs.append(
                        {
                            "method_id": method_id,
                            "lead_time": lead_time,
                            "H": horizon,
                            "seed": seed,
                            "test_repeats": args.test_repeats,
                            "train_overrides": train_overrides,
                        }
                    )
    return jobs


def parse_args():
    parser = argparse.ArgumentParser(description="LH sensitivity experiments for LS1.")
    parser.add_argument("--lead-times", type=parse_int_list, default=DEFAULT_LEAD_TIMES)
    parser.add_argument("--horizons", type=parse_int_list, default=DEFAULT_HORIZONS)
    parser.add_argument("--seeds", type=parse_int_list, default=DEFAULT_SEEDS)
    parser.add_argument("--methods", type=parse_method_list, default=list(TABLE_METHOD_ORDER))
    parser.add_argument("--test-repeats", type=int, default=100)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--eval-times", type=int, default=None)
    parser.add_argument("--train-repeats", type=int, default=None)
    parser.add_argument("--len-epi-train", type=int, default=None)
    parser.add_argument("--len-epi-eval", type=int, default=None)
    parser.add_argument("--n-repeats-eval", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--max-jobs", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results") / "lh_sensitivity")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_results_path = output_dir / "raw_results.csv"
    completed_records = [] if args.force else load_existing_results(raw_results_path)
    completed_keys = {job_key(row) for row in completed_records}

    all_jobs = build_jobs(args)
    pending_jobs = [job for job in all_jobs if job_key(job) not in completed_keys]
    if args.max_jobs is not None:
        pending_jobs = pending_jobs[: args.max_jobs]

    default_workers = max(1, (os.cpu_count() or 1) - 1)
    max_workers = args.max_workers or default_workers
    max_workers = min(max_workers, max(1, len(pending_jobs))) if pending_jobs else max_workers

    write_metadata(
        output_dir / "run_metadata.json",
        args,
        max_workers,
        len(all_jobs),
        len(completed_records),
        len(pending_jobs),
    )

    if not pending_jobs:
        write_summary_outputs(output_dir, completed_records, args.lead_times, args.horizons)
        print(f"No pending jobs. Rebuilt summary tables from {raw_results_path}.")
        return

    print(
        f"Launching {len(pending_jobs)} jobs with max_workers={max_workers} "
        f"(skipped {len(all_jobs) - len(pending_jobs)} completed jobs)."
    )
    start_time = perf_counter()
    records_by_key = {job_key(row): row for row in completed_records}
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as executor:
        future_to_job = {executor.submit(run_single_job, job): job for job in pending_jobs}
        for idx, future in enumerate(as_completed(future_to_job), start=1):
            job = future_to_job[future]
            row = future.result()
            records_by_key[job_key(row)] = row
            current_records = sort_records(list(records_by_key.values()))
            write_summary_outputs(output_dir, current_records, args.lead_times, args.horizons)

            elapsed = perf_counter() - start_time
            completed = idx
            avg_time = elapsed / completed
            remaining = len(pending_jobs) - completed
            eta = avg_time * remaining
            print(
                f"[{completed}/{len(pending_jobs)}] "
                f"{job['method_id']} L={job['lead_time']} H={job['H']} seed={job['seed']} "
                f"-> perf={row['performance']:.4f}, head={row['selected_head']}, "
                f"elapsed={solve_time(elapsed)}, eta={solve_time(eta)}"
            )

    final_records = sort_records(list(records_by_key.values()))
    write_summary_outputs(output_dir, final_records, args.lead_times, args.horizons)
    total_time = perf_counter() - start_time
    print(f"Finished {len(pending_jobs)} jobs in {solve_time(total_time)}.")
    print(f"Raw results: {raw_results_path}")
    print(f"Workbook: {output_dir / 'sensitivity_tables.xlsx'}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import inspect
import json
import os
import pickle
import sys
import time
from contextlib import contextmanager
from copy import deepcopy as copy
from datetime import datetime
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed


ROOT = Path(__file__).resolve().parent
EXPERIMENT_DIRS = {
    "stationary": ROOT / "7_MDQN_DRL",
    "dynamic": ROOT / "8_MDQN_dynamic",
}
SYSTEMS = ("LS", "PS", "DS")
TASKS = ("1", "2", "3", "4")
METHOD = "FHTDL"
FHTDL_H = 64
BEHAVIOR_HEAD = 0
EVAL_HEAD = FHTDL_H - 1
DEFAULT_PROGRESS_EVERY_SECONDS = 300.0
DEFAULT_PROGRESS_EVERY_STEPS = 10_000

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


def fixed_main_seeds(n_seeds: int = 16) -> list[int]:
    np.random.seed(0)
    seeds = np.random.choice(900, replace=False, size=n_seeds) + 100
    return [int(seed) for seed in seeds]


def clear_experiment_modules() -> None:
    prefixes = ("DRL", "simulators", "heuristic")
    for name in list(sys.modules):
        if name in prefixes or any(name.startswith(f"{prefix}.") for prefix in prefixes):
            del sys.modules[name]


@contextmanager
def experiment_context(experiment_dir: Path):
    old_cwd = Path.cwd()
    old_path = list(sys.path)
    clear_experiment_modules()
    os.chdir(experiment_dir)
    sys.path.insert(0, str(experiment_dir))
    try:
        yield
    finally:
        os.chdir(old_cwd)
        sys.path[:] = old_path
        clear_experiment_modules()


def make_env(system: str, task: str):
    task_idx = int(task) - 1
    if system == "LS":
        from simulators import LostSalesInventory, lost_sale_configs

        return LostSalesInventory(lost_sale_configs[task_idx])
    if system == "PS":
        from simulators import PerishableInventory, perishable_configs

        return PerishableInventory(perishable_configs[task_idx])
    if system == "DS":
        from simulators import DualSourcingInventory, dual_sourcing_configs

        return DualSourcingInventory(dual_sourcing_configs[task_idx])
    raise ValueError(f"Unknown system: {system}")


def build_mdqn_agent(MDQN, state_size: int, action_size: int, lr: float, seed: int, device: str | None):
    signature = inspect.signature(MDQN)
    kwargs = {"lambda_anc": 0.0, "seed": seed}
    if "device" in signature.parameters and device is not None:
        kwargs["device"] = device
    return MDQN(state_size, action_size, FHTDL_H, lr, **kwargs)


def test_fhtdl(env, agent, n_repeats: int, idx_head: int, is_eval: bool = False) -> float:
    rewards = []
    if not is_eval:
        env = copy(env)
    for _ in range(n_repeats):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.act(state, idx_head=idx_head)
            state, reward, done, truncate, info = env.step(action)
            rewards.append(reward)
    return -np.mean(rewards).item()


def train_fhtdl_batch(agent, transition: tuple, gamma: float) -> float:
    import torch

    s, a, r, s_next = transition
    device = getattr(agent, "device", None)
    s = torch.as_tensor(s, dtype=torch.float32, device=device)
    a = torch.as_tensor(a, dtype=torch.int64, device=device)
    r = torch.as_tensor(r, dtype=torch.float32, device=device)
    s_next = torch.as_tensor(s_next, dtype=torch.float32, device=device)

    with torch.no_grad():
        agent.net.eval()
        q_next = agent.net(s_next)
        agent.net.train()
        horizons = agent.horizons.to(r.device)
        prev_ratios = agent.prev_ratios.to(r.device)
        target = r.view(-1, 1) / horizons
        q_next_prev = torch.max(q_next[:, :-1, :], dim=-1)[0]
        target[:, 1:] += gamma * prev_ratios[:, 1:] * q_next_prev

    a_idx = a.view(-1, 1, 1).expand(-1, agent.n_heads, 1)
    q = agent.net(s).gather(-1, a_idx).squeeze(-1)
    head_losses = 0.5 * ((q - target) ** 2).mean(dim=0)
    loss = torch.sum(agent.loss_weights.to(q.device) * head_losses)
    agent.opt.zero_grad()
    loss.backward()
    agent.opt.step()
    return loss.item()


def train_test_fhtdl(
    experiment_dir: str,
    system: str,
    task: str,
    seed: int,
    train_steps_override: int | None = None,
    eval_times_override: int | None = None,
    test_repeats_override: int | None = None,
    progress_every_steps: int | None = DEFAULT_PROGRESS_EVERY_STEPS,
    progress_every_seconds: float | None = DEFAULT_PROGRESS_EVERY_SECONDS,
    progress_label: str | None = None,
) -> dict:
    experiment_path = Path(experiment_dir)
    with experiment_context(experiment_path):
        from DRL.MDQN_agent import MDQN
        from DRL.configs import DQN_config
        from DRL.utilize import ReplayBuffer

        env = make_env(system, task)
        task_name = f"{system}{task}"
        gamma = DQN_config["gamma"]
        train_steps = int(train_steps_override or DQN_config["train_steps"])
        len_epi_train = DQN_config["len_epi_train"]
        epsilon_start = DQN_config["epsilon_start"]
        epsilon_end = DQN_config["epsilon_end"]
        cache_size = DQN_config["cache_size"]
        batch_size = DQN_config["batch_size"]
        lr = DQN_config["lr"]
        eval_times = int(eval_times_override or DQN_config["eval_times"])
        eval_frq = max(1, int(train_steps / eval_times))
        len_epi_eval = DQN_config["len_epi_eval"]
        n_repeats_eval = DQN_config["n_repeats_eval"]
        test_repeats = int(test_repeats_override or DQN_config["test_repeats"])
        device = DQN_config.get("device")

        train_env, eval_env, test_env = copy(env), copy(env), copy(env)
        train_env.max_steps = len_epi_train
        eval_env.max_steps = len_epi_eval
        train_env._set_seed(seed)
        eval_env._set_seed(seed)

        replay_buffer = ReplayBuffer(train_steps, seed)
        agent = build_mdqn_agent(
            MDQN,
            train_env.observation_space.shape[0],
            train_env.action_space.n,
            lr,
            seed,
            device,
        )
        agent.current_head = BEHAVIOR_HEAD

        best_performance_eval_acc = np.inf
        performance_history_eval = []
        best_step = 0
        best_parameters = None
        done = True
        run_start = time.perf_counter()
        last_progress_step = 0
        last_progress_time = run_start
        latest_eval: float | None = None
        progress_name = progress_label or f"{experiment_path.name} {task_name} seed={seed}"

        print(
            f"SEED_START {progress_name}: H={FHTDL_H}, behavior_h={BEHAVIOR_HEAD + 1}, "
            f"eval_h={EVAL_HEAD + 1}, train_steps={train_steps}, eval_frq={eval_frq}",
            flush=True,
        )

        for train_step in range(train_steps + 1):
            if done:
                state, _ = train_env.reset()
                done = False

            epsilon = epsilon_start - (epsilon_start - epsilon_end) * (train_step / train_steps)
            action = agent.act(state, epsilon, idx_head=BEHAVIOR_HEAD)
            state_next, reward, done, truncated, info = train_env.step(action)
            replay_buffer.store((state, action, reward, state_next))
            state = state_next

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample_batch(batch_size)
                train_fhtdl_batch(agent, batch, gamma)

            if train_step % eval_frq == 0:
                performance_eval = test_fhtdl(eval_env, agent, n_repeats_eval, EVAL_HEAD, is_eval=True)
                performance_history_eval.append(performance_eval)
                performance_eval_acc = np.mean(performance_history_eval[-cache_size:])
                if performance_eval_acc < best_performance_eval_acc:
                    best_performance_eval_acc = performance_eval_acc
                    best_step = train_step
                    best_parameters = agent.get_parameters()
                latest_eval = performance_eval

            now = time.perf_counter()
            step_due = progress_every_steps is not None and train_step - last_progress_step >= progress_every_steps
            time_due = progress_every_seconds is not None and now - last_progress_time >= progress_every_seconds
            if train_step == train_steps or step_due or time_due:
                elapsed = now - run_start
                completed_steps = max(1, train_step)
                steps_per_second = completed_steps / elapsed
                eta = (train_steps - train_step) / steps_per_second if steps_per_second > 0 else float("inf")
                eval_text = "NA" if latest_eval is None else f"{latest_eval:.4f}"
                print(
                    f"SEED_PROGRESS {progress_name}: step={train_step}/{train_steps}, "
                    f"elapsed={format_duration(elapsed)}, steps_per_sec={steps_per_second:.2f}, "
                    f"eta={format_duration(eta)}, latest_eval={eval_text}, "
                    f"best_eval_acc={best_performance_eval_acc:.4f}, best_step={best_step}",
                    flush=True,
                )
                last_progress_step = train_step
                last_progress_time = now

        test_start = time.perf_counter()
        print(f"SEED_TEST_START {progress_name}: best_step={best_step}", flush=True)
        if best_parameters is not None:
            agent.load_parameters(best_parameters)
        performance_test = test_fhtdl(test_env, agent, test_repeats, EVAL_HEAD)
        test_elapsed = time.perf_counter() - test_start
        total_elapsed = time.perf_counter() - run_start
        print(
            f"SEED_DONE {progress_name}: train_eval_elapsed={format_duration(total_elapsed - test_elapsed)}, "
            f"test_elapsed={format_duration(test_elapsed)}, total_elapsed={format_duration(total_elapsed)}, "
            f"performance={performance_test:.4f}, history_len={len(performance_history_eval)}, "
            f"best_step={best_step}",
            flush=True,
        )

    return {
        "seed": seed,
        "history": performance_history_eval,
        "performance": performance_test,
        "step": best_step,
        "parameters": agent.get_parameters(),
        "method": METHOD,
        "H": FHTDL_H,
        "behavior_head": BEHAVIOR_HEAD + 1,
        "eval_head": EVAL_HEAD + 1,
        "lambda_anc": 0.0,
        "uses_dca": False,
        "uses_adaptive_exploration": False,
    }


def load_results(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        return pickle.load(handle)


def save_results(path: Path, results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(results, handle)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(seconds, 60.0)
    hours, minutes = divmod(int(minutes), 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{sec:04.1f}s"
    return f"{minutes:d}m{sec:04.1f}s"


def iter_task_specs(systems: list[str], tasks: list[str]) -> list[tuple[str, str, str]]:
    return [(system, task, f"{system}{task}") for system in systems for task in tasks]


def task_summary(
    experiment_label: str,
    task_name: str,
    task_results: list[dict],
    elapsed_seconds: float,
    seeds_run: int,
    expected_seed_count: int,
) -> dict:
    performances = np.array([item["performance"] for item in task_results], dtype=float)
    steps = np.array([item["step"] for item in task_results], dtype=float)
    best_index = int(np.argmin(performances))
    return {
        "event": "TASK_DONE",
        "experiment": experiment_label,
        "task": task_name,
        "elapsed_seconds": round(float(elapsed_seconds), 3),
        "elapsed": format_duration(elapsed_seconds),
        "seeds_run": int(seeds_run),
        "seeds_saved": int(len(task_results)),
        "seeds_expected": int(expected_seed_count),
        "best": round(float(performances.min()), 6),
        "mean": round(float(performances.mean()), 6),
        "std": round(float(performances.std()), 6),
        "best_seed": int(task_results[best_index]["seed"]),
        "mean_best_step": round(float(steps.mean()), 3),
    }


def run_experiment(
    experiment_label: str,
    systems: list[str],
    tasks: list[str],
    seeds: list[int],
    n_jobs: int,
    overwrite: bool,
    train_steps_override: int | None,
    eval_times_override: int | None,
    test_repeats_override: int | None,
    progress_every_steps: int | None,
    progress_every_seconds: float | None,
) -> None:
    experiment_dir = EXPERIMENT_DIRS[experiment_label]
    output_path = experiment_dir / "results" / "FHTDL_results.pkl"
    results = {} if overwrite else load_results(output_path)
    task_specs = iter_task_specs(systems, tasks)
    print(
        f"EXPERIMENT_START {experiment_label}: tasks={len(task_specs)}, "
        f"seeds_per_task={len(seeds)}, n_jobs={n_jobs}, "
        f"progress_every_steps={progress_every_steps}, "
        f"progress_every_seconds={progress_every_seconds}, output={output_path}",
        flush=True,
    )

    for system in systems:
        results.setdefault(system, {})
    for system, task, task_name in task_specs:
        task_start = time.perf_counter()
        task_started_at = datetime.now().isoformat(timespec="seconds")
        try:
            results[system].setdefault(task, {})
            existing = results[system][task].get(METHOD, [])
            completed = {int(item["seed"]): item for item in existing}
            remaining = [seed for seed in seeds if seed not in completed]
            if not remaining:
                task_results = [completed[seed] for seed in sorted(completed)]
                summary = task_summary(
                    experiment_label,
                    task_name,
                    task_results,
                    time.perf_counter() - task_start,
                    seeds_run=0,
                    expected_seed_count=len(seeds),
                )
                summary["started_at"] = task_started_at
                summary["reused"] = True
                print(f"TASK_LOG {json.dumps(summary, sort_keys=True)}", flush=True)
                continue

            print(
                f"TASK_START {experiment_label} {task_name}: remaining_seeds={len(remaining)}, "
                f"already_completed={len(completed)}, started_at={task_started_at}",
                flush=True,
            )
            result_generator = Parallel(
                n_jobs=n_jobs,
                backend="loky",
                batch_size="auto",
                return_as="generator_unordered",
            )(
                delayed(train_test_fhtdl)(
                    str(experiment_dir),
                    system,
                    task,
                    seed,
                    train_steps_override,
                    eval_times_override,
                    test_repeats_override,
                    progress_every_steps,
                    progress_every_seconds,
                    f"{experiment_label} {task_name} seed={seed}",
                )
                for seed in remaining
            )
            for result in result_generator:
                completed[int(result["seed"])] = result
                task_results = [completed[seed] for seed in sorted(completed)]
                results[system][task][METHOD] = task_results
                save_results(output_path, results)
            task_results = [completed[seed] for seed in sorted(completed)]
            results[system][task][METHOD] = task_results
            save_results(output_path, results)
            summary = task_summary(
                experiment_label,
                task_name,
                task_results,
                time.perf_counter() - task_start,
                seeds_run=len(remaining),
                expected_seed_count=len(seeds),
            )
            summary["started_at"] = task_started_at
            summary["reused"] = False
            print(f"TASK_LOG {json.dumps(summary, sort_keys=True)}", flush=True)
        except Exception as exc:
            elapsed = time.perf_counter() - task_start
            print(
                f"TASK_FAILED {experiment_label} {task_name}: "
                f"elapsed={format_duration(elapsed)}, error={exc!r}",
                flush=True,
            )
            raise


def parse_args():
    parser = argparse.ArgumentParser(description="Run the FHTDL H=64 baseline on the main benchmark tasks.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        choices=["stationary", "dynamic", "all"],
        default=["all"],
        help="Benchmark demand suites to run.",
    )
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--n-seeds", type=int, default=16)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--train-steps", type=int, default=None, help="Optional smoke-test override.")
    parser.add_argument("--eval-times", type=int, default=None, help="Optional smoke-test override.")
    parser.add_argument("--test-repeats", type=int, default=None, help="Optional smoke-test override.")
    parser.add_argument("--progress-every-steps", type=int, default=DEFAULT_PROGRESS_EVERY_STEPS)
    parser.add_argument("--progress-every-seconds", type=float, default=DEFAULT_PROGRESS_EVERY_SECONDS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one short FHTDL training call without writing result files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiments = list(EXPERIMENT_DIRS) if "all" in args.experiments else args.experiments
    seeds = fixed_main_seeds(args.n_seeds)

    if args.dry_run:
        print(f"Experiments: {experiments}")
        print(f"Systems: {args.systems}")
        print(f"Tasks: {args.tasks}")
        print(f"Task order: {[task_name for _, _, task_name in iter_task_specs(args.systems, args.tasks)]}")
        print(f"Seeds: {seeds}")
        print(f"FHTDL: H={FHTDL_H}, behavior h={BEHAVIOR_HEAD + 1}, eval h={EVAL_HEAD + 1}")
        print(f"Progress every steps: {args.progress_every_steps}")
        print(f"Progress every seconds: {args.progress_every_seconds:g}")
        return

    if args.smoke_test:
        experiment_label = experiments[0]
        system = args.systems[0]
        task = args.tasks[0]
        log = train_test_fhtdl(
            str(EXPERIMENT_DIRS[experiment_label]),
            system,
            task,
            seeds[0],
            train_steps_override=args.train_steps or 1,
            eval_times_override=args.eval_times or 1,
            test_repeats_override=args.test_repeats or 1,
            progress_every_steps=args.progress_every_steps,
            progress_every_seconds=args.progress_every_seconds,
            progress_label=f"smoke {experiment_label} {system}{task} seed={seeds[0]}",
        )
        print(
            f"smoke {experiment_label} {system}{task} seed={log['seed']}: "
            f"method={log['method']}, H={log['H']}, behavior h={log['behavior_head']}, "
            f"eval h={log['eval_head']}, performance={log['performance']:.4f}, "
            f"history_len={len(log['history'])}"
        )
        return

    for experiment_label in experiments:
        run_experiment(
            experiment_label,
            args.systems,
            args.tasks,
            seeds,
            args.n_jobs,
            args.overwrite,
            args.train_steps,
            args.eval_times,
            args.test_repeats,
            args.progress_every_steps,
            args.progress_every_seconds,
        )


if __name__ == "__main__":
    main()

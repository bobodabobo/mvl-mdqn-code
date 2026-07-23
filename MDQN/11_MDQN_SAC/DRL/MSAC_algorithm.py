import numpy as np
from copy import deepcopy as copy
import gymnasium as gym

from .configs import DQN_config, MSAC_config
from .MSAC_agent import MSAC
from .utilize import CostDelayCache, ReplayBuffer


def _update_ema_score(previous_score:float, current_cost:float, weight:float):
    if not np.isfinite(previous_score):
        return current_cost
    return weight * current_cost + (1.0 - weight) * previous_score


def _get_head_neighborhood(center:int, radius:int, n_heads:int):
    left = max(0, center - radius)
    right = min(n_heads, center + radius + 1)
    return list(range(left, right))


def train_test_MSAC(**kwargs):
    task_name = kwargs.get("task_name", "task-default")
    env = kwargs["env"]
    seed = kwargs.get("seed", MSAC_config["seed"])
    is_print = kwargs.get("is_print", False)
    DRL_method = kwargs.get("DRL_method", "MSAC")
    if DRL_method != "MSAC":
        raise ValueError(f"DRL_method {DRL_method} not supported.")
    gamma = kwargs.get("gamma", MSAC_config["gamma"])
    H = kwargs.get("H", MSAC_config["H"])
    w = kwargs.get("w", MSAC_config["w"])
    lambda_anc = kwargs.get("lambda_anc", MSAC_config["lambda_anc"])
    eval_radius = kwargs.get("eval_radius", MSAC_config["eval_radius"])
    reward_delay_time = env.reward_delay_time
    train_steps = kwargs.get("train_steps", MSAC_config["train_steps"])
    len_epi_train = kwargs.get("len_epi_train", MSAC_config["len_epi_train"])
    train_repeats = kwargs.get("train_repeats", MSAC_config["train_repeats"])
    replay_buffer_size = int(train_steps)
    update_frq = kwargs.get("update_frq", MSAC_config["update_frq"])
    batch_size = kwargs.get("batch_size", MSAC_config["batch_size"])
    lr = kwargs.get("lr", MSAC_config["lr"])
    tau = kwargs.get("tau", MSAC_config["tau"])
    alpha_init = kwargs.get("alpha_init", MSAC_config["alpha_init"])
    alpha_lr = kwargs.get("alpha_lr", MSAC_config["alpha_lr"])
    target_entropy_scale = kwargs.get("target_entropy_scale", MSAC_config["target_entropy_scale"])
    eval_times = kwargs.get("eval_times", MSAC_config["eval_times"])
    eval_frq = max(1, int(train_steps / eval_times))
    len_epi_eval = kwargs.get("len_epi_eval", MSAC_config["len_epi_eval"])
    n_repeats_eval = kwargs.get("n_repeats_eval", MSAC_config["n_repeats_eval"])
    test_repeats = kwargs.get("test_repeats", MSAC_config["test_repeats"])

    train_env, eval_env, test_env = copy(env), copy(env), copy(env)
    train_env.max_steps = len_epi_train
    eval_env.max_steps = len_epi_eval
    train_env._set_seed(seed)
    eval_env._set_seed(seed)

    replay_buffer = ReplayBuffer(replay_buffer_size, seed)
    cost_cache = CostDelayCache(reward_delay_time, gamma)
    agent = MSAC(train_env.observation_space.shape[0],
                 train_env.action_space.n,
                 H,
                 lr,
                 tau,
                 alpha_init,
                 alpha_lr,
                 target_entropy_scale,
                 lambda_anc=lambda_anc,
                 seed=seed)

    scores = np.full(H, np.inf, dtype=np.float64)
    best_performance, best_step = np.inf, 0
    best_parameters = agent.get_parameters()
    performance_history = []
    done = True
    for train_step in range(train_steps + 1):
        if done:
            cost_cache.clear()
            state, _ = train_env.reset()
            done = False
        action, _ = agent.act(state)
        state_next, reward, done, truncated, info = train_env.step(action)
        transition = cost_cache.push((state,
                                      action,
                                      info["costs"]["timely"],
                                      info["costs"]["delayed"],
                                      state_next))
        if transition is not None:
            replay_buffer.store(transition)
        state = state_next
        if len(replay_buffer) >= batch_size:
            for _ in range(train_repeats):
                batch = replay_buffer.sample_batch(batch_size)
                agent.train(batch, gamma)
            if train_step % update_frq == 0:
                agent.update_target()
        if train_step % eval_frq == 0:
            candidate_heads = _get_head_neighborhood(agent.current_head, eval_radius, H)
            performances_eval = {}
            for idx_head in candidate_heads:
                performance_eval = _testMSAC(eval_env, agent, n_repeats_eval, is_eval=True, idx_head=idx_head)
                performances_eval[idx_head] = performance_eval
                scores[idx_head] = _update_ema_score(scores[idx_head], performance_eval, w)
            current_head = min(candidate_heads, key=lambda idx: scores[idx])
            agent.current_head = current_head
            performance_log = performances_eval[current_head]
            performance_history.append(performance_log)
            performance_acc_best = scores[current_head]
            if performance_acc_best < best_performance:
                best_parameters = agent.get_parameters()
                best_performance = performance_acc_best
                best_step = train_step
            if is_print:
                print(
                    f"step-{train_step}: performance_eval={performance_log:.3f}, "
                    f"h={agent.current_head + 1}, best_performance={best_performance:.3f}, "
                    f"best_h={best_parameters['head'] + 1}@{best_step}."
                )
    if best_parameters is not None:
        agent.load_parameters(best_parameters)
    performance_test = _testMSAC(test_env, agent, test_repeats)
    if is_print:
        print(
            f"Test {task_name}_{DRL_method}_seed={seed}: performance={performance_test:.3f}, "
            f"h={agent.current_head + 1}, step={best_step}."
        )
    log = {
        "seed": seed,
        "history": performance_history,
        "performance": performance_test,
        "step": best_step,
        "parameters": agent.get_parameters(),
    }
    return log


def _testMSAC(env:gym.Env, agent:MSAC, n_repeats:int=10, is_eval:bool=False, idx_head:int=None):
    rewards = []
    if not is_eval:
        env = copy(env)
    for _ in range(n_repeats):
        state, _ = env.reset()
        done = False
        while not done:
            action, _ = agent.act(state, deterministic=True, idx_head=idx_head)
            state, reward, done, truncate, info = env.step(action)
            rewards.append(reward)
    return -np.mean(rewards).item()

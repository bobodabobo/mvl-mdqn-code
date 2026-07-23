import numpy as np
from copy import deepcopy as copy
import gymnasium as gym

from .PPO_agent import PPO
from .configs import PPO_config
from .utilize import RolloutBuffer


def train_test_PPO(**kwargs):
    task_name = kwargs.get("task_name", "task-default")
    env = kwargs.get("env", None)
    seed = kwargs.get("seed", PPO_config["seed"])
    is_print = kwargs.get("is_print", False)
    DRL_method = "PPO"
    gamma = kwargs.get("gamma", PPO_config["gamma"])
    train_steps = kwargs.get("train_steps", PPO_config["train_steps"])
    len_epi_train = kwargs.get("len_epi_train", PPO_config["len_epi_train"])
    batch_size = kwargs.get("batch_size", PPO_config["batch_size"])
    rollout_size = kwargs.get("rollout_size", PPO_config["rollout_size"])
    lr = kwargs.get("lr", PPO_config["lr"])
    clip_ratio = kwargs.get("clip_ratio", PPO_config["clip_ratio"])
    gae_lambda = kwargs.get("gae_lambda", PPO_config["gae_lambda"])
    update_epochs = kwargs.get("update_epochs", PPO_config["update_epochs"])
    value_coef = kwargs.get("value_coef", PPO_config["value_coef"])
    entropy_coef = kwargs.get("entropy_coef", PPO_config["entropy_coef"])
    max_grad_norm = kwargs.get("max_grad_norm", PPO_config["max_grad_norm"])
    cache_size = kwargs.get("cache_size", PPO_config["cache_size"])
    eval_times = kwargs.get("eval_times", PPO_config["eval_times"])
    eval_frq = int(train_steps / eval_times)
    len_epi_eval = kwargs.get("len_epi_eval", PPO_config["len_epi_eval"])
    n_repeats_eval = kwargs.get("n_repeats_eval", PPO_config["n_repeats_eval"])
    test_repeats = kwargs.get("test_repeats", PPO_config["test_repeats"])
    train_env, eval_env, test_env = copy(env), copy(env), copy(env)
    train_env.max_steps = len_epi_train
    eval_env.max_steps = len_epi_eval
    train_env._set_seed(seed)
    eval_env._set_seed(seed)
    rollout_buffer = RolloutBuffer()
    agent = PPO(train_env.observation_space.shape[0],
                train_env.action_space.n,
                lr,
                batch_size,
                clip_ratio,
                update_epochs,
                value_coef,
                entropy_coef,
                max_grad_norm,
                seed)
    best_performance_eval_acc = np.inf
    performance_history_eval = []
    best_step = 0
    best_parameters = None
    done = True
    for train_step in range(train_steps + 1):
        if done:
            state, _ = train_env.reset()
            done = False
        action, log_prob, value = agent.act(state)
        state_next, reward, done, truncated, info = train_env.step(action)
        rollout_buffer.store(state, action, reward, float(done), log_prob, value)
        state = state_next
        if len(rollout_buffer) >= rollout_size or train_step == train_steps:
            last_value = 0.0 if done else agent.evaluate_value(state)
            batch = rollout_buffer.get(gamma, gae_lambda, last_value)
            agent.train(batch)
            rollout_buffer.clear()
        if train_step % eval_frq == 0:
            performance_eval = testPPO(eval_env, agent, n_repeats_eval, is_eval=True)
            performance_history_eval.append(performance_eval)
            performance_eval_acc = np.mean(performance_history_eval[-cache_size:])
            if is_print:
                print(f"step-{train_step}: performance_eval={performance_eval:.3f}, performance_eval_acc={performance_eval_acc:.3f}.")
            if performance_eval_acc < best_performance_eval_acc:
                best_performance_eval_acc = performance_eval_acc
                best_step = train_step
                best_parameters = agent.get_parameters()
    if best_parameters is not None:
        agent.load_parameters(best_parameters)
    performance_test = testPPO(test_env, agent, test_repeats)
    if is_print:
        print(f"Test {task_name}_{DRL_method}_seed={seed}: performance={performance_test:.3f}, step={best_step}.")
    log = {
        "seed": seed,
        "history": performance_history_eval,
        "performance": performance_test,
        "step": best_step,
        "parameters": agent.get_parameters()
    }
    return log


def testPPO(env:gym.Env, agent:PPO, n_repeats:int=10, is_eval:bool=False):
    rewards = []
    if not is_eval:
        env = copy(env)
    for _ in range(n_repeats):
        state, _ = env.reset()
        done = False
        while not done:
            action, _, _ = agent.act(state, deterministic=True)
            state, reward, done, truncate, info = env.step(action)
            rewards.append(reward)
    return -np.mean(rewards).item()

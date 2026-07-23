import numpy as np
from copy import deepcopy as copy
import gymnasium as gym

from .configs import SAC_config
from .SAC_agent import SAC
from .utilize import ReplayBufferWithDone


def train_test_SAC(**kwargs):
    task_name = kwargs.get("task_name", "task-default")
    env = kwargs.get("env", None)
    seed = kwargs.get("seed", SAC_config["seed"])
    is_print = kwargs.get("is_print", False)
    DRL_method = kwargs.get("DRL_method", "SAC")
    gamma = kwargs.get("gamma", SAC_config["gamma"])
    if "-L" in DRL_method:
        train_steps = kwargs.get("train_steps", SAC_config["train_steps_long"])
    else:
        train_steps = kwargs.get("train_steps", SAC_config["train_steps"])
    len_epi_train = kwargs.get("len_epi_train", SAC_config["len_epi_train"])
    replay_buffer_size = train_steps
    batch_size = kwargs.get("batch_size", SAC_config["batch_size"])
    lr = kwargs.get("lr", SAC_config["lr"])
    alpha_lr = kwargs.get("alpha_lr", SAC_config["alpha_lr"])
    tau = kwargs.get("tau", SAC_config["tau"])
    target_entropy_scale = kwargs.get("target_entropy_scale", SAC_config["target_entropy_scale"])
    alpha_init = kwargs.get("alpha_init", SAC_config["alpha_init"])
    cache_size = kwargs.get("cache_size", SAC_config["cache_size"])
    eval_times = kwargs.get("eval_times", SAC_config["eval_times"])
    eval_frq = int(train_steps / eval_times)
    len_epi_eval = kwargs.get("len_epi_eval", SAC_config["len_epi_eval"])
    n_repeats_eval = kwargs.get("n_repeats_eval", SAC_config["n_repeats_eval"])
    update_frq = kwargs.get("update_frq", SAC_config["update_frq"])
    test_repeats = kwargs.get("test_repeats", SAC_config["test_repeats"])
    train_env, eval_env, test_env = copy(env), copy(env), copy(env)
    train_env.max_steps = len_epi_train
    eval_env.max_steps = len_epi_eval
    train_env._set_seed(seed)
    eval_env._set_seed(seed)
    replay_buffer = ReplayBufferWithDone(replay_buffer_size, seed)
    agent = SAC(train_env.observation_space.shape[0],
                train_env.action_space.n,
                lr,
                tau,
                alpha_init,
                alpha_lr,
                target_entropy_scale,
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
        action, log_prob = agent.act(state)
        state_next, reward, done, truncated, info = train_env.step(action)
        replay_buffer.store((state, action, reward, state_next, float(done)))
        state = state_next
        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample_batch(batch_size)
            agent.train(batch, gamma)
            if train_step % update_frq == 0:
                agent.update_target()
        if train_step % eval_frq == 0:
            performance_eval = testSAC(eval_env, agent, n_repeats_eval, is_eval=True)
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
    performance_test = testSAC(test_env, agent, test_repeats)
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


def testSAC(env:gym.Env, agent:SAC, n_repeats:int=10, is_eval:bool=False):
    rewards = []
    if not is_eval:
        env = copy(env)
    for _ in range(n_repeats):
        state, _ = env.reset()
        done = False
        while not done:
            action, _ = agent.act(state, deterministic=True)
            state, reward, done, truncate, info = env.step(action)
            rewards.append(reward)
    return -np.mean(rewards).item()

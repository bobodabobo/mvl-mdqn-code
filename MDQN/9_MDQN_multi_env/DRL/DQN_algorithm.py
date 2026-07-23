from copy import deepcopy as copy

import gymnasium as gym
import numpy as np

from .DQN_agent import DDQN
from .configs import DQN_config
from .utilize import ReplayBuffer


def train_test_DDQN(**kwargs):
    task_name = kwargs.get("task_name", "task-default")
    env = kwargs["env"]
    seed = kwargs.get("seed", DQN_config["seed"])
    is_print = kwargs.get("is_print", False)
    gamma = kwargs.get("gamma", DQN_config["gamma"])
    train_steps = kwargs.get("train_steps", DQN_config["train_steps"])
    len_epi_train = kwargs.get("len_epi_train", DQN_config["len_epi_train"])
    epsilon_start = kwargs.get("epsilon_start", DQN_config["epsilon_start"])
    epsilon_end = kwargs.get("epsilon_end", DQN_config["epsilon_end"])
    replay_buffer_size = int(train_steps)
    cache_size = kwargs.get("cache_size", DQN_config["cache_size"])
    batch_size = kwargs.get("batch_size", DQN_config["batch_size"])
    lr = kwargs.get("lr", DQN_config["lr"])
    eval_times = kwargs.get("eval_times", DQN_config["eval_times"])
    eval_frq = max(1, int(train_steps / eval_times))
    len_epi_eval = kwargs.get("len_epi_eval", DQN_config["len_epi_eval"])
    n_repeats_eval = kwargs.get("n_repeats_eval", DQN_config["n_repeats_eval"])
    test_repeats = kwargs.get("test_repeats", DQN_config["test_repeats"])
    update_frq = kwargs.get("update_frq", DQN_config["update_frq"])

    train_env, eval_env, test_env = copy(env), copy(env), copy(env)
    train_env.max_steps = len_epi_train
    eval_env.max_steps = len_epi_eval
    train_env._set_seed(seed)
    eval_env._set_seed(seed)

    replay_buffer = ReplayBuffer(replay_buffer_size, seed)
    agent = DDQN(train_env.observation_space.shape[0], train_env.action_space.n, lr, seed)

    best_performance_eval_acc = np.inf
    performance_history_eval = []
    best_step = 0
    best_parameters = None
    done = True

    for train_step in range(train_steps + 1):
        if done:
            state, _ = train_env.reset()
            done = False
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (train_step / train_steps)
        action = agent.act(state, epsilon)
        state_next, reward, done, truncated, info = train_env.step(action)
        replay_buffer.store((state, action, reward, state_next))
        state = state_next

        if len(replay_buffer) >= batch_size:
            batch = replay_buffer.sample_batch(batch_size)
            agent.train(batch, gamma)

        if train_step % eval_frq == 0:
            performance_eval = testDDQN(eval_env, agent, n_repeats_eval, is_eval=True)
            performance_history_eval.append(performance_eval)
            performance_eval_acc = float(np.mean(performance_history_eval[-cache_size:]))
            if is_print:
                print(
                    f"step-{train_step}: performance_eval={performance_eval:.3f}, "
                    f"performance_eval_acc={performance_eval_acc:.3f}."
                )
            if performance_eval_acc < best_performance_eval_acc:
                best_performance_eval_acc = performance_eval_acc
                best_step = train_step
                best_parameters = agent.get_parameters()

        if train_step % update_frq == 0:
            agent.update_target()

    if best_parameters is not None:
        agent.load_parameters(best_parameters)
    performance_test = testDDQN(test_env, agent, test_repeats)
    if is_print:
        print(f"Test {task_name}_DDQN_seed={seed}: performance={performance_test:.3f}, step={best_step}.")
    return {
        "seed": seed,
        "history": performance_history_eval,
        "performance": performance_test,
        "step": best_step,
        "parameters": agent.get_parameters(),
    }


def testDDQN(env: gym.Env, agent: DDQN, n_repeats: int = 10, is_eval: bool = False):
    rewards = []
    if not is_eval:
        env = copy(env)
    for _ in range(n_repeats):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, truncate, info = env.step(action)
            rewards.append(reward)
    return -np.mean(rewards).item()

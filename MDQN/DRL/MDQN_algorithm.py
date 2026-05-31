import numpy as np
from copy import deepcopy as copy
import gymnasium as gym
from collections import deque

from .utilize import ReplayBuffer, CostDelayCache
from .MDQN_agent import MDQN
from .configs import DQN_config, MDQN_config


def train_test_MDQN(**kwargs):
    # 1.load parameters
    # 1.1 task
    task_name = kwargs.get("task_name", "task-default")
    env = kwargs['env']
    seed = kwargs.get("seed", DQN_config["seed"])
    is_print = kwargs.get("is_print", False)
    # 1.2 for RL
    DRL_method = 'MDQN'
    H = kwargs.get("H", MDQN_config["H"])
    w = kwargs.get("w", MDQN_config["w"])
    reward_delay_time = env.reward_delay_time
    train_steps = kwargs.get("train_steps", MDQN_config["train_steps"])
    len_epi_train = kwargs.get("len_epi_train", DQN_config["len_epi_train"])
    train_repeats = kwargs.get("train_repeats", MDQN_config["train_repeats"])
    epsilon_start = kwargs.get("epsilon_start", DQN_config["epsilon_start"])
    epsilon_end = kwargs.get("epsilon_end", DQN_config["epsilon_end"])
    replay_buffer_size = int(train_steps)
    cache_size = kwargs.get("cache_size", DQN_config["cache_size"])
    update_frq = kwargs.get("update_frq", DQN_config["update_frq"])
    # 1.3 for DL
    batch_size = kwargs.get("batch_size", DQN_config["batch_size"])
    lr = kwargs.get("lr", DQN_config["lr"])
    # 1.4 for evaluation
    eval_times = kwargs.get("eval_times", MDQN_config["eval_times"])
    eval_frq = int(train_steps / eval_times)
    len_epi_eval = kwargs.get("len_epi_eval", DQN_config["len_epi_eval"])
    n_repeats_eval = kwargs.get("n_repeats_eval", DQN_config["n_repeats_eval"])
    # 1.5 for test
    test_repeats = kwargs.get("test_repeats", DQN_config["test_repeats"])
    # 2.Initialize
    # 2.1 environments
    train_env, eval_env, test_env = copy(env), copy(env), copy(env)
    train_env.max_steps = len_epi_train
    eval_env.max_steps = len_epi_eval
    train_env._set_seed(seed)
    eval_env._set_seed(seed)
    # 2.2 conponents
    replay_buffer = ReplayBuffer(replay_buffer_size, seed)
    cost_cache = CostDelayCache(reward_delay_time)
    # 2.3 agent
    agent = MDQN(train_env.observation_space.shape[0], train_env.action_space.n, H, lr, seed)
    # 3.train loop
    scores = None
    best_performance, best_step = np.inf, 0
    best_parameters = agent.get_parameters()
    performances_cache = deque(maxlen=cache_size)
    performance_history = []
    done = True
    for train_step in range(train_steps + 1):
        if done:
            cost_cache.clear()
            state, _ = train_env.reset()
            done = False
        # 3.1.explore
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (train_step / train_steps)
        action = agent.act(state, epsilon)
        state_next, reward, done, truncated, info = train_env.step(action)
        transition = cost_cache.push((state,
                                      action,
                                      info["costs"]["timely"],
                                      info["costs"]["delayed"],
                                      state_next))
        if transition is not None:
            replay_buffer.store(transition)
        state = state_next
        # 3.2.train
        if len(replay_buffer) >= batch_size:
            for _ in range(train_repeats):
                batch = replay_buffer.sample_batch(batch_size)
                agent.train(batch)
        # 3.3.update target net
        if train_step % update_frq == 0:
            agent.update_target_net()
        # 3.4.eval
        if train_step % eval_frq == 0:
            # eval paerformance
            performances_eval = []
            new_replay_buffers = []
            for idx_head in range(H):
                performance_eval, replay_buffer_new = _evalMDQN(eval_env,
                                                            agent,
                                                            cost_cache,
                                                            replay_buffer,
                                                            n_repeats_eval)
                performances_eval.append(performance_eval)
                new_replay_buffers.append(replay_buffer_new)
            performances_eval = np.array(performances_eval)
            # update scores
            if scores is None:
                scores = performances_eval
            else:
                scores = scores * w + performances_eval * (1 - w)
            # update agent and log
            current_head = np.argmin(scores)
            agent.current_head = current_head
            performance_log = performances_eval[current_head]
            performance_history.append(performance_log)
            # update replay buffer
            idx_buffer = np.argmin(performances_eval)
            replay_buffer.concate(new_replay_buffers[idx_buffer])
            # uodate best parameters
            performances_cache.append(performances_eval)
            performances_acc = np.mean(np.array(performances_cache), axis=0)
            best_head = np.argmin(performances_acc)
            performance_acc_best = performances_acc[best_head]
            if performance_acc_best < best_performance and len(performances_cache) == cache_size:
                best_parameters = agent.get_parameters()
                best_parameters["head"] = best_head
                best_performance = performance_acc_best
                best_step = train_step
            # print
            if is_print:
                print(f"step-{train_step}: performance_eval={performance_log:.3f}, h={agent.current_head + 1}, best_performance={best_performance:.3f}, best_h={best_parameters["head"] + 1}@{best_step}.")
    # 4.test
    if best_parameters is not None:
        agent.load_parameters(best_parameters)
    performance_test = _testMDQN(test_env, agent, test_repeats)
    if is_print:
        print(f"Test {task_name}_{DRL_method}_seed={seed}: performance={performance_test:.3f}, h={agent.current_head + 1}, step={best_step}.")
    # 5.log
    log = {
        "seed": seed,
        "history": performance_history,
        "performance": performance_test,
        "step": best_step,
        "parameters": agent.get_parameters()
    }
    return log
            

def _testMDQN(env:gym.Env, agent:MDQN, n_repeats:int=10, idx_head:int=None):
    rewards = []
    env = copy(env)
    for t in range(n_repeats):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.act(state, idx_head=idx_head)
            state, reward, done, truncate, info = env.step(action)
            rewards.append(reward)
    return -np.mean(rewards).item()

def _evalMDQN(env:gym.Env, agent:MDQN, cost_cache:CostDelayCache, replay_buffer:ReplayBuffer, n_repeats:int=10, idx_head:int=None):
    replay_buffer = ReplayBuffer(max_size=replay_buffer.max_size)
    rewards = []
    cost_cache = copy(cost_cache)
    for t in range(n_repeats):
        state, _ = env.reset()
        done = False
        cost_cache.clear()
        while not done:
            action = agent.act(state, idx_head=idx_head)
            state_next, reward, done, truncate, info = env.step(action)
            transition = cost_cache.push((state,
                                            action,
                                            info["costs"]["timely"],
                                            info["costs"]["delayed"],
                                            state_next))
            if transition is not None:
                replay_buffer.store(transition)
            rewards.append(reward)
            state = state_next
    return -np.mean(rewards).item(), replay_buffer

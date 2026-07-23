import numpy as np
from copy import deepcopy as copy
import gymnasium as gym
from time import perf_counter

from .utilize import ReplayBuffer
from .DQN_agent import DQN, TDQN, DDQN
from .configs import DQN_config


def train_test_DQN(**kwargs):
    # 1.load parameters
    # 1.1 task
    task_name = kwargs.get("task_name", "task-default")
    env = kwargs.get("env", None)
    seed = kwargs.get("seed", DQN_config["seed"])
    is_print = kwargs.get("is_print", False)
    # 1.2 for RL
    DRL_method = kwargs['DRL_method']
    gamma = kwargs.get("gamma", DQN_config["gamma"])
    if "-L" in DRL_method:
        train_steps = kwargs.get("train_steps", DQN_config["train_steps_long"])
    else:
        train_steps = kwargs.get("train_steps", DQN_config["train_steps"])
    len_epi_train = kwargs.get("len_epi_train", DQN_config["len_epi_train"])
    epsilon_start = kwargs.get("epsilon_start", DQN_config["epsilon_start"])
    epsilon_end = kwargs.get("epsilon_end", DQN_config["epsilon_end"])
    replay_buffer_size = train_steps
    cache_size = kwargs.get("cache_size", DQN_config["cache_size"])
    # 1.3 for DL
    batch_size = kwargs.get("batch_size", DQN_config["batch_size"])
    lr = kwargs.get("lr", DQN_config["lr"])
    # 1.4 for evaluation
    eval_times = kwargs.get("eval_times", DQN_config["eval_times"])
    eval_frq = int(train_steps / eval_times)
    len_epi_eval = kwargs.get("len_epi_eval", DQN_config["len_epi_eval"])
    n_repeats_eval = kwargs.get("n_repeats_eval", DQN_config["n_repeats_eval"])
    # 1.5 for test
    test_repeats = kwargs.get("test_repeats", DQN_config["test_repeats"])
    collect_profile = kwargs.get("collect_profile", False)
    # 2.Initialize
    # 2.1 environments
    train_env, eval_env, test_env = copy(env), copy(env), copy(env)
    train_env.max_steps = len_epi_train
    eval_env.max_steps = len_epi_eval
    train_env._set_seed(seed)
    eval_env._set_seed(seed)
    # 2.2 conponents
    replay_buffer = ReplayBuffer(replay_buffer_size, seed)
    # 2.3 agent
    if DRL_method in ['DQN', "DQN-L"]:
        agent = DQN(train_env.observation_space.shape[0], train_env.action_space.n, lr, seed)
    elif DRL_method in ['TDQN', 'TDQN-L']:
        agent = TDQN(train_env.observation_space.shape[0], train_env.action_space.n, lr, seed)
    elif DRL_method in ['DDQN', 'DDQN-L']:
        agent = DDQN(train_env.observation_space.shape[0], train_env.action_space.n, lr, seed)
    elif DRL_method in ['RSDQN', 'RSDQN-L']:
        agent = DDQN(train_env.observation_space.shape[0], train_env.action_space.n, lr, seed)
        teacher_agent = kwargs['teacher_agent']
        w_teacher = kwargs.get("w_teacher", DQN_config["w_teacher"])
    else:
        raise ValueError(f"DRL_method {DRL_method} not supported.")
    if not DRL_method == 'DQN':
        update_frq = kwargs.get("update_frq", DQN_config["update_frq"])
    # 3.train loop
    best_performance_eval_acc = np.inf
    performance_history_eval = []
    best_step = 0
    best_parameters = None
    profile = None
    if collect_profile:
        profile = {
            "explore_seconds": 0.0,
            "train_seconds": 0.0,
            "validation_seconds": 0.0,
            "test_seconds": 0.0,
            "explore_steps": 0,
            "train_updates": 0,
            "target_syncs": 0,
            "validation_rounds": 0,
            "validation_rollouts": 0,
            "validation_steps": 0,
            "test_rollouts": test_repeats,
            "test_steps": test_repeats * test_env.max_steps,
        }
    done = True
    for train_step in range(train_steps + 1):
        if done:
            state, _ = train_env.reset()
            done = False
        # 3.1.explore
        t_phase = perf_counter() if collect_profile else None
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (train_step / train_steps)
        action = agent.act(state, epsilon)
        state_next, reward, done, truncated, info = train_env.step(action)
        if DRL_method in ['RSDQN', 'RSDQN-L']:
            action_teacher = teacher_agent.act(state)
            reward -= w_teacher * train_env.cal_action_distance(action_teacher, action)
        replay_buffer.store((state, action, reward, state_next))
        state = state_next
        if collect_profile:
            profile["explore_seconds"] += perf_counter() - t_phase
            profile["explore_steps"] += 1
        # 3.2.train
        if len(replay_buffer) >= batch_size:
            t_phase = perf_counter() if collect_profile else None
            batch = replay_buffer.sample_batch(batch_size)
            agent.train(batch, gamma)
            if collect_profile:
                profile["train_seconds"] += perf_counter() - t_phase
                profile["train_updates"] += 1
        # 3.3.eval
        if train_step % eval_frq == 0:
            t_phase = perf_counter() if collect_profile else None
            performance_eval = testDQN(eval_env, agent, n_repeats_eval, is_eval=True)
            performance_history_eval.append(performance_eval)
            performance_eval_acc = np.mean(performance_history_eval[-cache_size:])
            if is_print:
                print(f"step-{train_step}: performance_eval={performance_eval:.3f}, performance_eval_acc={performance_eval_acc:.3f}.")
            if performance_eval_acc < best_performance_eval_acc:
                best_performance_eval_acc = performance_eval_acc
                best_step = train_step
                best_parameters = agent.get_parameters()
            if collect_profile:
                profile["validation_seconds"] += perf_counter() - t_phase
                profile["validation_rounds"] += 1
                profile["validation_rollouts"] += n_repeats_eval
                profile["validation_steps"] += n_repeats_eval * len_epi_eval
        # 3.4.update target q net
        if DRL_method in ['TDQN', 'DDQN', 'RSDQN', 'TDQN-L', 'DDQN-L', 'RSDQN-L']:
            if train_step % update_frq == 0:
                t_phase = perf_counter() if collect_profile else None
                agent.update_target()
                if collect_profile:
                    profile["train_seconds"] += perf_counter() - t_phase
                    profile["target_syncs"] += 1
    # 4.test
    if best_parameters is not None:
        agent.load_parameters(best_parameters)
    t_phase = perf_counter() if collect_profile else None
    performance_test = testDQN(test_env, agent, test_repeats)
    if collect_profile:
        profile["test_seconds"] += perf_counter() - t_phase
    if is_print:
        print(f"Test {task_name}_{DRL_method}_seed={seed}: performance={performance_test:.3f}, step={best_step}.")
    # 5.log
    log = {
        "seed": seed,
        "history": performance_history_eval,
        "performance": performance_test,
        "step": best_step,
        "parameters": agent.get_parameters()
    }
    if collect_profile:
        log["profile"] = profile
    return log
            

def testDQN(env:gym.Env, agent:DQN, n_repeats:int=10, is_eval:bool=False):
    rewards = []
    if not is_eval:
        env = copy(env)
    for t in range(n_repeats):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            state, reward, done, truncate, info = env.step(action)
            rewards.append(reward)
    return -np.mean(rewards).item()

import numpy as np
from copy import deepcopy as copy
import gymnasium as gym

from .utilize import ReplayBuffer, CostDelayCache
from .MDQN_agent import MDQN
from .configs import DQN_config, MDQN_config


def _update_ema_score(previous_score:float, current_cost:float, weight:float):
    if not np.isfinite(previous_score):
        return current_cost
    return weight * current_cost + (1.0 - weight) * previous_score


def _get_head_neighborhood(center:int, radius:int, n_heads:int):
    left = max(0, center - radius)
    right = min(n_heads, center + radius + 1)
    return list(range(left, right))


def train_test_MDQN(**kwargs):
    # 1.load parameters
    # 1.1 task
    task_name = kwargs.get("task_name", "task-default")
    env = kwargs['env']
    seed = kwargs.get("seed", DQN_config["seed"])
    is_print = kwargs.get("is_print", False)
    # 1.2 for RL
    DRL_method = 'MDQN'
    gamma = kwargs.get("gamma", DQN_config["gamma"])
    H = kwargs.get("H", MDQN_config["H"])
    w = kwargs.get("w", MDQN_config["w"])
    lambda_anc = kwargs.get("lambda_anc", MDQN_config["lambda_anc"])
    eval_radius = kwargs.get("eval_radius", MDQN_config["eval_radius"])
    adaptive_exploration = kwargs.get("adaptive_exploration", True)
    fixed_head_idx = kwargs.get("fixed_head_idx", 0)
    final_test_all_heads = kwargs.get("final_test_all_heads", False)
    reward_delay_time = env.reward_delay_time
    train_steps = kwargs.get("train_steps", MDQN_config["train_steps"])
    len_epi_train = kwargs.get("len_epi_train", DQN_config["len_epi_train"])
    train_repeats = kwargs.get("train_repeats", MDQN_config["train_repeats"])
    epsilon_start = kwargs.get("epsilon_start", DQN_config["epsilon_start"])
    epsilon_end = kwargs.get("epsilon_end", DQN_config["epsilon_end"])
    replay_buffer_size = int(train_steps)
    update_frq = kwargs.get("update_frq", DQN_config["update_frq"])
    # 1.3 for DL
    batch_size = kwargs.get("batch_size", DQN_config["batch_size"])
    lr = kwargs.get("lr", DQN_config["lr"])
    # 1.4 for evaluation
    eval_times = kwargs.get("eval_times", MDQN_config["eval_times"])
    eval_frq = max(1, int(train_steps / eval_times))
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
    cost_cache = CostDelayCache(reward_delay_time, gamma)
    # 2.3 agent
    agent = MDQN(train_env.observation_space.shape[0],
                 train_env.action_space.n,
                 H,
                 lr,
                 lambda_anc=lambda_anc,
                 seed=seed)
    if not 0 <= fixed_head_idx < H:
        raise ValueError(f"fixed_head_idx={fixed_head_idx} is out of range for H={H}.")
    if not adaptive_exploration:
        agent.current_head = fixed_head_idx
    # 3.train loop
    scores = np.full(H, np.inf, dtype=np.float64)
    best_performance, best_step = np.inf, 0
    best_parameters = agent.get_parameters()
    best_parameters["head"] = agent.current_head
    performance_history = []
    done = True
    for train_step in range(train_steps + 1):
        if done:
            cost_cache.clear()
            state, _ = train_env.reset()
            done = False
        # 3.1.explore
        epsilon = epsilon_start - (epsilon_start - epsilon_end) * (train_step / train_steps)
        if adaptive_exploration:
            action = agent.act(state, epsilon)
        else:
            action = agent.act(state, epsilon, idx_head=fixed_head_idx)
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
                agent.train(batch, gamma)
        # 3.3.update target net
        if train_step % update_frq == 0:
            agent.update_target_net()
        # 3.4.eval
        if train_step % eval_frq == 0:
            if adaptive_exploration:
                candidate_heads = _get_head_neighborhood(agent.current_head, eval_radius, H)
                performances_eval = {}
                new_replay_buffers = {}
                for idx_head in candidate_heads:
                    performance_eval, replay_buffer_new = _evalMDQN(eval_env,
                                                                    agent,
                                                                    cost_cache,
                                                                    replay_buffer,
                                                                    n_repeats_eval,
                                                                    idx_head=idx_head)
                    performances_eval[idx_head] = performance_eval
                    new_replay_buffers[idx_head] = replay_buffer_new
                    scores[idx_head] = _update_ema_score(scores[idx_head], performance_eval, w)
                # update agent and log within the local neighborhood only
                current_head = min(candidate_heads, key=lambda idx: scores[idx])
                agent.current_head = current_head
                performance_log = performances_eval[current_head]
                performance_history.append(performance_log)
                # update replay buffer
                idx_buffer = min(candidate_heads, key=lambda idx: performances_eval[idx])
                replay_buffer.concate(new_replay_buffers[idx_buffer])
                # update best parameters using the historical EMA score of the selected head
                performance_acc_best = scores[current_head]
                if performance_acc_best < best_performance:
                    best_parameters = agent.get_parameters()
                    best_parameters["head"] = current_head
                    best_performance = performance_acc_best
                    best_step = train_step
                # print
                if is_print:
                    print(
                        f"step-{train_step}: performance_eval={performance_log:.3f}, "
                        f"h={agent.current_head + 1}, best_performance={best_performance:.3f}, "
                        f"best_h={best_parameters['head'] + 1}@{best_step}."
                    )
            else:
                performance_eval, replay_buffer_new = _evalMDQN(eval_env,
                                                                agent,
                                                                cost_cache,
                                                                replay_buffer,
                                                                n_repeats_eval,
                                                                idx_head=fixed_head_idx)
                scores[fixed_head_idx] = _update_ema_score(scores[fixed_head_idx], performance_eval, w)
                performance_log = performance_eval
                performance_history.append(performance_log)
                replay_buffer.concate(replay_buffer_new)
                performance_acc_best = scores[fixed_head_idx]
                if performance_acc_best < best_performance:
                    best_parameters = agent.get_parameters()
                    best_parameters["head"] = fixed_head_idx
                    best_performance = performance_acc_best
                    best_step = train_step
                if is_print:
                    print(
                        f"step-{train_step}: performance_eval={performance_log:.3f}, "
                        f"h={fixed_head_idx + 1}, best_performance={best_performance:.3f}, "
                        f"best_h={best_parameters['head'] + 1}@{best_step}."
                    )
    # 4.test
    if best_parameters is not None:
        agent.load_parameters(best_parameters)
    selected_head = agent.current_head
    head_test_performances = None
    best_test_head = selected_head
    if final_test_all_heads:
        head_test_performances = []
        for idx_head in range(H):
            head_test_performances.append(_testMDQN(test_env, agent, test_repeats, idx_head=idx_head))
        best_test_head = int(np.argmin(head_test_performances))
        performance_test = head_test_performances[best_test_head]
    else:
        performance_test = _testMDQN(test_env, agent, test_repeats)
    if is_print:
        print(
            f"Test {task_name}_{DRL_method}_seed={seed}: performance={performance_test:.3f}, "
            f"train_h={selected_head + 1}, test_h={best_test_head + 1}, step={best_step}."
        )
    # 5.log
    log = {
        "seed": seed,
        "history": performance_history,
        "performance": performance_test,
        "step": best_step,
        "adaptive_exploration": adaptive_exploration,
        "final_test_all_heads": final_test_all_heads,
        "selected_head": selected_head,
        "selected_horizon": selected_head + 1,
        "best_test_head": best_test_head,
        "best_test_horizon": best_test_head + 1,
        "head_test_performances": head_test_performances,
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

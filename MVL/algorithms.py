from copy import deepcopy as copy
import numpy as np
from environment import BairdCounterExample
from agent import ExploreAgent, TargetAgent, cal_importance
from approximator import Approximator


def _TD_semi_gradient(approximator:Approximator,
                     gamma:float,
                     state:int,
                     reward:float,
                     next_state:int):
    feature = approximator.features[state, :]
    next_feature = approximator.features[next_state, :]
    td_error = reward + gamma * np.dot(next_feature, approximator.weights) - np.dot(feature, approximator.weights)
    gradient = 2 * td_error * feature
    return gradient


def _TD_full_gradient(approximator:Approximator,
                     gamma:float,
                     state:int,
                     reward:float,
                     next_state:int):
    feature = approximator.features[state, :]
    weights = approximator.weights
    next_feature = approximator.features[next_state, :]
    td_error = reward + gamma * np.dot(next_feature, weights) - np.dot(feature, weights)
    gradient = 2 * td_error * (feature - gamma * next_feature)
    return gradient


def _MVL_gradient(approximator_now:Approximator,
                 approximator_pre:Approximator,
                 h:int,
                 state:int,
                 reward:float,
                 next_state:int):
    feature = approximator_now.features[state, :]
    next_feature = approximator_pre.features[next_state, :]
    td_error = (reward + (h - 1) * np.dot(next_feature, approximator_pre.weights)) / h - np.dot(feature, approximator_now.weights)
    gradient = 2 * td_error * feature
    return gradient


def on_policy_TD_learning(n_steps:int=1000, gamma:float=0.99, lr:float=0.001, gradient_type:str='semi'):
    if gradient_type == 'semi':
        cal_gradient = _TD_semi_gradient
    elif gradient_type == 'full':
        cal_gradient = _TD_full_gradient
    else:
        raise ValueError(f"Invalid type: {gradient_type}")
    explore_agent = TargetAgent()
    target_agent = TargetAgent()
    env = BairdCounterExample()
    approximator = Approximator(lr=lr)
    log = []
    state, _ = env.reset(seed=0)
    for step in range(n_steps):
        action = explore_agent.act(state)
        next_state, reward, _, _, _ = env.step(action)
        gradient = cal_gradient(approximator, gamma, state, reward, next_state)
        norm = approximator.update(gradient)
        log.append(norm)
        state = next_state
    env.close()
    return log


def on_policy_MVL(H:int=3, n_steps:int=1000, lr:float=0.001):
    explore_agent = TargetAgent()
    target_agent = TargetAgent()
    env = BairdCounterExample()
    approximators = [Approximator(lr=lr) for _ in range(H)]
    log = [[] for _ in range(H)]
    state, _ = env.reset(seed=0)
    for step in range(n_steps):
        action = explore_agent.act(state)
        next_state, reward, _, _, _ = env.step(action)
        for h in range(1, H+1):
            gradient = _MVL_gradient(approximators[h - 1],
                                    approximators[h - 2],
                                    h,
                                    state,
                                    reward,
                                    next_state)
            norm = approximators[h - 1].update(gradient)
            log[h - 1].append(norm)
        state = next_state
    env.close()
    return log


def off_policy_TD_learning(n_steps:int=1000, gamma:float=0.99, lr:float=0.001, gradient_type:str='semi'):
    if gradient_type == 'semi':
        cal_gradient = _TD_semi_gradient
    elif gradient_type == 'full':
        cal_gradient = _TD_full_gradient
    else:
        raise ValueError(f"Invalid type: {gradient_type}")
    explore_agent = ExploreAgent()
    target_agent = TargetAgent()
    env = BairdCounterExample()
    approximator = Approximator(lr=lr)
    log = []
    state, _ = env.reset(seed=0)
    for step in range(n_steps):
        action = explore_agent.act(state)
        next_state, reward, _, _, _ = env.step(action)
        importance = cal_importance(explore_agent, target_agent, state, action)
        gradient = cal_gradient(approximator, gamma, state, reward, next_state)
        norm = approximator.update(importance * gradient)
        log.append(norm)
        state = next_state
    env.close()
    return log


def off_policy_MVL(H:int=3, n_steps:int=1000, lr:float=0.001):
    explore_agent = ExploreAgent()
    target_agent = TargetAgent()
    env = BairdCounterExample()
    approximators = [Approximator(lr=lr) for _ in range(H)]
    log = [[] for _ in range(H)]
    state, _ = env.reset(seed=0)
    for step in range(n_steps):
        action = explore_agent.act(state)
        next_state, reward, _, _, _ = env.step(action)
        importance = cal_importance(explore_agent, target_agent, state, action)
        for h in range(1, H+1):
            gradient = _MVL_gradient(approximators[h - 1],
                                    approximators[h - 2],
                                    h,
                                    state,
                                    reward,
                                    next_state)
            norm = approximators[h - 1].update(gradient * importance)
            log[h - 1].append(norm)
        state = next_state
    env.close()
    return log




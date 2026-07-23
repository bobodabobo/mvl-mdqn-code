from copy import deepcopy as copy
import numpy as np
from joblib import Parallel, delayed
from numba import njit

from .basic import Agent, evaluate_policy_multi_process
from .base_stocks import BaseStock
from simulators import PerishableInventory


class _Const(Agent):
    '''Deterministic policies for restricting the search space of BSP-low-EW'''
    def __init__(self, env:PerishableInventory, parameters:dict=None):
        self.env = copy(env)
        self.max_order_quantity = self.env.max_order_quantity
        if parameters is None:
            parameters = {'action': 0}
        self.set_parameters(parameters)

    def act(self, obs):
        return self.action
    
    def train(self, length:int, repeats:int=1):
        parameters_list = [{"action": action} for action in np.arange(self.max_order_quantity + 1)]
        log = evaluate_policy_multi_process(self, self.env, length, repeats, parameters_list)
        return log["parameters"]["action"]


@njit(cache=True)
def cal_EW(inventory_pipeline:np.ndarray,
           total_inventory:float,
           shelf_life:int,
           lead_time:int,
           mean_value:float,
           is_fifo:bool):
    '''Calculate the expected number of wast goods'''
    ew = 0
    for t in range(lead_time):
        if total_inventory <= 0:
            break
        remaining_demand = mean_value
        indices = np.arange(t, t + shelf_life) if is_fifo else np.arange(t + shelf_life - 1, t - 1, -1)
        for idx in indices:
            if remaining_demand <= 0:
                break
            num_sell = min(remaining_demand, inventory_pipeline[idx])
            remaining_demand -= num_sell
            inventory_pipeline[idx] -= num_sell
            total_inventory -= num_sell
        ew += inventory_pipeline[t]
    return ew


def _cal_QNB(env:PerishableInventory):
    '''Calculate the newsvendor order quantity for restricting the search space of BSP-low-EW'''
    # load parameters
    env = copy(env)
    _ = env.reset()
    demands = env.demand_list
    time_interval = env.shelf_life + env.lead_time
    max_inventory = env.max_inventory
    shortage_cost_rate, wastage_cost_rate = env.shortage_cost_rate, env.wastage_cost_rate
    # prepare sum of demands
    n_acc_demands = len(demands) - time_interval + 1
    acc_demands = [np.sum(demands[idx:idx + time_interval]) for idx in range(n_acc_demands)]
    acc_demands = np.array(acc_demands)
    # threshold
    threshold = shortage_cost_rate / (wastage_cost_rate + shortage_cost_rate)
    # search for QNB
    for QNB in range(max_inventory + 1):
        if np.sum(acc_demands > QNB) / n_acc_demands >= threshold:
            return QNB
    return max_inventory


class BSPLowEW(Agent):
    def __init__(self, env, parameters:dict=None):
        self.env = copy(env)
        self.max_order_quantity = self.env.max_order_quantity
        self.max_inventory = self.env.max_inventory
        self.is_fifo = self.env.policy == "FIFO"
        self.shelf_life = env.shelf_life
        self.lead_time = env.lead_time
        self.mean_value = env.demand_generator.dist_param
        if parameters is None:
            parameters = {'s1': 0, 'b': 0, 's2': 0, 'alpha': 0}
        self.set_parameters(parameters)

    def act(self, obs):
        inventory_level = np.sum(obs)
        ew = cal_EW(copy(obs),
                    inventory_level,
                    self.shelf_life,
                    self.lead_time,
                    self.mean_value,
                    self.is_fifo)
        if inventory_level < self.b:
            order_quantity = self.s1 - self.alpha * inventory_level + ew
        else:
            order_quantity = self.s2 - inventory_level + ew
        order_quantity = min(self.max_order_quantity, max(0, order_quantity))
        order_quantity = int(order_quantity)
        return order_quantity
    
    def train(self, length:int, repeats:int=1):
        # prepare search space
        const_agent = _Const(self.env)
        const = const_agent.train(length, repeats)
        base_stock_agent = BaseStock(self.env)
        info = base_stock_agent.train(length, repeats)
        base_stock = info["parameters"]["base_stock"]
        qnb = _cal_QNB(self.env)
        # warm up for njit
        _ = cal_EW(np.ones(3) * 2, 6, 2, 2, 1, True)
        # search for BSP_low_EW parameters
        parameters_list = []
        # # ----Stock-level range
        # s_1_range = range(0, base_stock + 1)
        # s_2_range = range(base_stock, max(base_stock, qnb) + 1)
        # b_range = range(1, max(1, base_stock) + 1)
        # ----Improved ordering range
        s_1_range = range(0, max(base_stock, const) + 1)
        s_2_range = range(min(base_stock, const), max(base_stock, qnb, const) + 1)
        b_range = range(1, max(const, base_stock) + 1)
        for s1 in s_1_range:
            for s2 in s_2_range:
                for b in b_range:
                    alpha = 1 - (s2 - s1) / b
                    parameters_list.append({'s1': s1, 'b': b, 's2': s2, 'alpha': alpha})
        log = evaluate_policy_multi_process(self, self.env, length, repeats, parameters_list)
        self.set_parameters(log["parameters"])
        return log

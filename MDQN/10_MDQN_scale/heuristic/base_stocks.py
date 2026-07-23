from copy import deepcopy as copy
import numpy as np

from .basic import Agent, evaluate_policy_multi_process


class BaseStock(Agent):
    def __init__(self, env, parameters:dict=None):
        self.env = copy(env)
        default_base_stock_max = self.env.max_inventory + (self.env.lead_time - 1) * self.env.max_order_quantity
        heuristic_base_stock_max = getattr(self.env, "heuristic_base_stock_max", None)
        self.max_inventory = default_base_stock_max if heuristic_base_stock_max is None else int(heuristic_base_stock_max)
        self.max_order_quantity = self.env.max_order_quantity
        if parameters is None:
            parameters = {'base_stock': 0}
        self.set_parameters(parameters)

    def act(self, obs):
        inventory_level = np.sum(obs)
        order_quantity = max(min(self.max_order_quantity, self.base_stock - inventory_level), 0)
        return order_quantity
    
    def train(self, length:int=None, repeats:int=None, n_jobs:int=-1):
        parameters_list = [{"base_stock": base_stock} for base_stock in range(self.max_inventory + 1)]
        log = evaluate_policy_multi_process(self, self.env, length, repeats, parameters_list, n_jobs=n_jobs)
        self.set_parameters(log["parameters"])
        return log


class CappedBaseStock(Agent):
    def __init__(self, env, parameters:dict=None):
        self.env = copy(env)
        default_base_stock_max = self.env.max_inventory + (self.env.lead_time - 1) * self.env.max_order_quantity
        heuristic_base_stock_max = getattr(self.env, "heuristic_base_stock_max", None)
        heuristic_cap_max = getattr(self.env, "heuristic_cap_max", None)
        self.max_inventory = default_base_stock_max if heuristic_base_stock_max is None else int(heuristic_base_stock_max)
        self.max_order_quantity = self.env.max_order_quantity
        self.max_cap = self.max_order_quantity if heuristic_cap_max is None else int(heuristic_cap_max)
        if parameters is None:
            parameters = {'base_stock': 0, 'cap': 0}
        self.set_parameters(parameters)

    def act(self, obs):
        inventory_level = np.sum(obs)
        order_quantity = max(min(self.max_order_quantity, self.base_stock - inventory_level), 0)
        order_quantity = min(order_quantity, self.cap)
        return order_quantity
    
    def train(self, length:int, repeats:int, n_jobs:int=-1):
        base_stocks = list(range(self.max_inventory + 1))
        caps = list(range(self.max_cap + 1))
        parameters_list = []
        for base_stock in base_stocks:
            for cap in caps:
                parameters_list.append({'base_stock': base_stock, 'cap': cap})
        log = evaluate_policy_multi_process(self, self.env, length, repeats, parameters_list, n_jobs=n_jobs)
        self.set_parameters(log["parameters"])
        return log

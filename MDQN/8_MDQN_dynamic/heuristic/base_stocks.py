from copy import deepcopy as copy
import numpy as np

from .basic import Agent, evaluate_policy_multi_process


class BaseStock(Agent):
    def __init__(self, env, parameters:dict=None):
        self.env = copy(env)
        self.max_inventory = self.env.max_inventory + (self.env.lead_time - 1) * self.env.max_order_quantity
        self.max_order_quantity = self.env.max_order_quantity
        if parameters is None:
            parameters = {'base_stock': 0}
        self.set_parameters(parameters)

    def act(self, obs):
        inventory_level = np.sum(obs)
        order_quantity = max(min(self.max_order_quantity, self.base_stock - inventory_level), 0)
        return int(order_quantity)
    
    def train(self, length:int=None, repeats:int=None):
        parameters_list = [{"base_stock": base_stock} for base_stock in range(self.max_inventory + 1)]
        log = evaluate_policy_multi_process(self, self.env, length, repeats, parameters_list)
        self.set_parameters(log["parameters"])
        return log


class CappedBaseStock(Agent):
    def __init__(self, env, parameters:dict=None):
        self.env = copy(env)
        self.max_inventory = self.env.max_inventory + (self.env.lead_time - 1) * self.env.max_order_quantity
        self.max_order_quantity = self.env.max_order_quantity
        if parameters is None:
            parameters = {'base_stock': 0, 'cap': 0}
        self.set_parameters(parameters)

    def act(self, obs):
        inventory_level = np.sum(obs)
        order_quantity = max(min(self.max_order_quantity, self.base_stock - inventory_level), 0)
        order_quantity = min(order_quantity, self.cap)
        return int(order_quantity)
    
    def train(self, length:int, repeats:int):
        base_stocks = list(range(self.max_inventory + 1))
        caps = list(range(self.max_order_quantity + 1))
        parameters_list = []
        for base_stock in base_stocks:
            for cap in caps:
                parameters_list.append({'base_stock': base_stock, 'cap': cap})
        log = evaluate_policy_multi_process(self, self.env, length, repeats, parameters_list)
        self.set_parameters(log["parameters"])
        return log

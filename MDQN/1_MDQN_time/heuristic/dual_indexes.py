from copy import deepcopy as copy
import numpy as np

from .basic import Agent, evaluate_policy_multi_process


class DualIndex(Agent):
    def __init__(self, env, parameters:dict=None):
        self.env = copy(env)
        self.max_inventory = self.env.max_inventory
        self.max_order_quantity = self.env.max_order_quantity
        self.lead_time_expediting = self.env.lead_time_expediting
        if parameters is None:
            parameters = {'base_stock_normal': 0,
                          'base_stock_expediting': 0}
        self.set_parameters(parameters)

    def act(self, obs):
        inventory_level_normal = np.sum(obs)
        inventory_level_expediting = np.sum(obs[:self.lead_time_expediting + 1])
        order_quantity_normal = max(min(self.max_order_quantity, self.base_stock_normal - inventory_level_normal), 0)
        order_quantity_expediting = max(min(self.max_order_quantity, self.base_stock_expediting - inventory_level_expediting), 0)
        order_quantity_normal = max(0, order_quantity_normal - order_quantity_expediting)
        action = order_quantity_normal * (self.max_order_quantity + 1) + order_quantity_expediting
        return action
    
    def train(self, length:int=None, repeats:int=1):
        parameters_list = []
        for base_stock_normal in range(self.max_inventory + 1):
            for base_stock_expediting in range(self.max_inventory + 1):
                parameters_list.append({'base_stock_normal': base_stock_normal,
                                        'base_stock_expediting': base_stock_expediting})
        log = evaluate_policy_multi_process(self, self.env, length, repeats, parameters_list)
        self.set_parameters(log["parameters"])
        return log


class CappedDualIndex(Agent):
    def __init__(self, env, parameters:dict=None):
        self.env = copy(env)
        self.max_inventory = self.env.max_inventory
        self.max_order_quantity = self.env.max_order_quantity
        self.lead_time_expediting = self.env.lead_time_expediting
        if parameters is None:
            parameters = {'base_stock_normal': 0,
                          'base_stock_expediting': 0,
                          'cap_normal': 0,
                          'cap_expediting': 0}
        self.set_parameters(parameters)

    def act(self, obs):
        # expediting
        inventory_level_expediting = np.sum(obs[:self.lead_time_expediting + 1])
        order_quantity_expediting = max(self.base_stock_expediting - inventory_level_expediting, 0)
        order_quantity_expediting = min(order_quantity_expediting, self.cap_expediting)
        # normal
        inventory_level_normal = np.sum(obs)
        order_quantity_normal = max(self.base_stock_normal - inventory_level_normal, 0)
        order_quantity_normal = max(0, order_quantity_normal - order_quantity_expediting)
        order_quantity_normal = min(order_quantity_normal, self.cap_normal)
        # action
        action = order_quantity_normal * (self.max_order_quantity + 1) + order_quantity_expediting
        return action
    
    def train(self, length:int=10000, repeats:int=10):
        parameters_list = []
        for base_stock_expediting in range(self.max_inventory + 1):
            for base_stock_normal in range(base_stock_expediting + 1, self.max_inventory + 1):
                for cap_normal in range(self.max_order_quantity + 1):
                    for cap_expediting in range(self.max_order_quantity + 1):
                        parameters_list.append({'base_stock_normal': base_stock_normal,
                                          'base_stock_expediting': base_stock_expediting,
                                          'cap_normal': cap_normal,
                                          'cap_expediting': cap_expediting})
        log = evaluate_policy_multi_process(self, self.env, length, repeats, parameters_list)
        self.set_parameters(log["parameters"])
        return log
    


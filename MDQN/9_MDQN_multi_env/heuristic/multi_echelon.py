from copy import deepcopy as copy

import numpy as np

from .basic import Agent, evaluate_policy_multi_process


class EchelonBaseStock(Agent):
    def __init__(self, env, parameters: dict = None):
        self.env = copy(env)
        self.max_order_quantity_retail = self.env.max_order_quantity_retail
        self.max_order_quantity_warehouse = self.env.max_order_quantity_warehouse
        self.lead_time_retail = self.env.lead_time_retail
        self.lead_time_supplier = self.env.lead_time_supplier
        self.max_s1 = (
            self.env.max_retail_inventory
            + self.env.max_internal_backlog
            + self.lead_time_retail * self.max_order_quantity_retail
        )
        self.max_s2_extra = (
            self.env.max_warehouse_inventory
            + self.lead_time_supplier * self.max_order_quantity_warehouse
        )
        if parameters is None:
            parameters = {
                "base_stock_retail_echelon": 0,
                "base_stock_system_echelon": 0,
            }
        self.set_parameters(parameters)

    def act(self, obs):
        obs = np.asarray(obs, dtype=np.int64)
        retailer_net_inventory = obs[0]
        internal_request_backlog = obs[1]
        pipeline_to_retailer = obs[2 : 2 + self.lead_time_retail]
        warehouse_on_hand = obs[2 + self.lead_time_retail]
        pipeline_to_warehouse = obs[3 + self.lead_time_retail :]

        eip1 = retailer_net_inventory + internal_request_backlog + int(np.sum(pipeline_to_retailer))
        eip2 = eip1 + warehouse_on_hand + int(np.sum(pipeline_to_warehouse))
        q_retail = int(
            np.clip(
                self.base_stock_retail_echelon - eip1,
                0,
                self.max_order_quantity_retail,
            )
        )
        q_warehouse = int(
            np.clip(
                self.base_stock_system_echelon - eip2,
                0,
                self.max_order_quantity_warehouse,
            )
        )
        return q_retail * (self.max_order_quantity_warehouse + 1) + q_warehouse

    def train(self, length: int = None, repeats: int = 1):
        parameters_list = []
        for s1 in range(self.max_s1 + 1):
            s2_max = s1 + self.max_s2_extra
            for s2 in range(s1, s2_max + 1):
                parameters_list.append(
                    {
                        "base_stock_retail_echelon": s1,
                        "base_stock_system_echelon": s2,
                    }
                )
        log = evaluate_policy_multi_process(self, self.env, length, repeats, parameters_list)
        self.set_parameters(log["parameters"])
        return log


class FixedOrderME(Agent):
    def __init__(self, env, parameters: dict = None):
        self.env = copy(env)
        self.max_order_quantity_retail = self.env.max_order_quantity_retail
        self.max_order_quantity_warehouse = self.env.max_order_quantity_warehouse
        if parameters is None:
            parameters = {"q_retail": 0, "q_warehouse": 0}
        self.set_parameters(parameters)

    def act(self, obs):
        del obs
        q_retail = int(np.clip(self.q_retail, 0, self.max_order_quantity_retail))
        q_warehouse = int(np.clip(self.q_warehouse, 0, self.max_order_quantity_warehouse))
        return q_retail * (self.max_order_quantity_warehouse + 1) + q_warehouse

    def train(self, length: int = None, repeats: int = 1):
        parameters_list = []
        for q_retail in range(self.max_order_quantity_retail + 1):
            for q_warehouse in range(self.max_order_quantity_warehouse + 1):
                parameters_list.append({"q_retail": q_retail, "q_warehouse": q_warehouse})
        log = evaluate_policy_multi_process(self, self.env, length, repeats, parameters_list)
        self.set_parameters(log["parameters"])
        return log

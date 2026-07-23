from collections import deque
from copy import deepcopy as copy

import gymnasium as gym
import numpy as np
from gymnasium import spaces


serial_multi_echelon_configs = [
    {
        "demand_mean": 3,
        "demand_dist": "poisson_truncated",
        "demand_max": 8,
        "seed": 0,
        "lead_time_retail": 1,
        "lead_time_supplier": 2,
        "holding_cost_rate_retail": 1.0,
        "holding_cost_rate_warehouse": 0.4,
        "shortage_cost_rate": 8.0,
        "max_order_quantity_retail": 6,
        "max_order_quantity_warehouse": 6,
        "max_retail_inventory": 12,
        "max_warehouse_inventory": 18,
        "max_internal_backlog": 12,
        "max_steps": 1000,
    },
    {
        "demand_mean": 3,
        "demand_dist": "poisson_truncated",
        "demand_max": 8,
        "seed": 0,
        "lead_time_retail": 2,
        "lead_time_supplier": 3,
        "holding_cost_rate_retail": 1.0,
        "holding_cost_rate_warehouse": 0.4,
        "shortage_cost_rate": 10.0,
        "max_order_quantity_retail": 6,
        "max_order_quantity_warehouse": 6,
        "max_retail_inventory": 12,
        "max_warehouse_inventory": 24,
        "max_internal_backlog": 16,
        "max_steps": 1000,
    },
]


class DemandGenerator:
    """Truncated Poisson demand for the serial multi-echelon environment."""

    def __init__(self, mean: int, demand_max: int, seed: int = 0, demand_dist: str = "poisson_truncated"):
        if demand_dist != "poisson_truncated":
            raise ValueError(f"Unsupported demand distribution: {demand_dist}.")
        self.mean = mean
        self.demand_max = demand_max
        self.demand_dist = demand_dist
        self.reset_seed(seed)

    def reset_seed(self, seed: int):
        self.rng = np.random.default_rng(seed=seed)

    def sample(self, size: int = 1):
        draws = self.rng.poisson(self.mean, size=size)
        return np.minimum(draws, self.demand_max).astype(np.int64)


class SerialMultiEchelonInventory(gym.Env):
    """Two-echelon serial inventory system with customer and internal backlogs."""

    def __init__(self, config: dict):
        super().__init__()
        self.lead_time_retail = config["lead_time_retail"]
        self.lead_time_supplier = config["lead_time_supplier"]
        self.holding_cost_rate_retail = config["holding_cost_rate_retail"]
        self.holding_cost_rate_warehouse = config["holding_cost_rate_warehouse"]
        self.shortage_cost_rate = config["shortage_cost_rate"]
        self.max_order_quantity_retail = config["max_order_quantity_retail"]
        self.max_order_quantity_warehouse = config["max_order_quantity_warehouse"]
        self.max_retail_inventory = config["max_retail_inventory"]
        self.max_warehouse_inventory = config["max_warehouse_inventory"]
        self.max_internal_backlog = config["max_internal_backlog"]
        self.max_customer_backlog = config.get("max_customer_backlog", self.max_internal_backlog)
        self.max_steps = config["max_steps"]
        self.reward_delay_time = self.lead_time_retail + self.lead_time_supplier
        self.demand_max = config["demand_max"]
        self.demand_generator = DemandGenerator(
            mean=config["demand_mean"],
            demand_max=config["demand_max"],
            seed=config["seed"],
            demand_dist=config["demand_dist"],
        )
        action_size = (self.max_order_quantity_retail + 1) * (self.max_order_quantity_warehouse + 1)
        self.action_space = spaces.Discrete(action_size)

        obs_low = np.array(
            [-self.max_customer_backlog, 0]
            + [0] * self.lead_time_retail
            + [0]
            + [0] * self.lead_time_supplier,
            dtype=np.int64,
        )
        obs_high = np.array(
            [self.max_retail_inventory, self.max_internal_backlog]
            + [self.max_internal_backlog] * self.lead_time_retail
            + [self.max_warehouse_inventory]
            + [self.max_order_quantity_warehouse] * self.lead_time_supplier,
            dtype=np.int64,
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.int64)
        self._set_seed(config["seed"])

    def _set_seed(self, seed: int):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)
        self.action_space.seed(seed)
        self.demand_generator.reset_seed(seed)

    def _initialize(self):
        self.retailer_net_inventory_last = 0
        self.internal_request_backlog_last = 0
        self.pipeline_to_retailer_last = deque(
            [0] * self.lead_time_retail, maxlen=self.lead_time_retail + 1
        )
        self.pipeline_to_warehouse_last = deque(
            [0] * self.lead_time_supplier, maxlen=self.lead_time_supplier + 1
        )
        self.warehouse_on_hand_last = 0

        self.retailer_net_inventory = 0
        self.internal_request_backlog = 0
        self.pipeline_to_retailer = deque([0] * self.lead_time_retail, maxlen=self.lead_time_retail + 1)
        self.pipeline_to_warehouse = deque(
            [0] * self.lead_time_supplier, maxlen=self.lead_time_supplier + 1
        )
        self.warehouse_on_hand = 0

        self.current_step = 0
        self.timely_cost = 0.0
        self.delayed_cost = 0.0
        self.holding_cost_retail = 0.0
        self.holding_cost_warehouse = 0.0
        self.shortage_cost = 0.0
        self.total_cost = 0.0
        self.retailer_request = 0
        self.warehouse_supplier_order = 0
        self.shipment_to_retailer = 0
        self.demand = 0

    def _decode_action(self, action: int):
        q_warehouse = action % (self.max_order_quantity_warehouse + 1)
        q_retail = action // (self.max_order_quantity_warehouse + 1)
        return q_retail, q_warehouse

    def _clip_retail_inventory(self):
        truncated = False
        if self.retailer_net_inventory > self.max_retail_inventory:
            self.retailer_net_inventory = self.max_retail_inventory
            truncated = True
        if self.retailer_net_inventory < -self.max_customer_backlog:
            self.retailer_net_inventory = -self.max_customer_backlog
            truncated = True
        return truncated

    def _get_obs(self):
        observation = [
            self.retailer_net_inventory,
            self.internal_request_backlog,
            *self.pipeline_to_retailer,
            self.warehouse_on_hand,
            *self.pipeline_to_warehouse,
        ]
        return np.array(observation, dtype=np.int64)

    def _get_info(self):
        return {
            "step": self.current_step,
            "costs": {
                "holding_retail": self.holding_cost_retail,
                "holding_warehouse": self.holding_cost_warehouse,
                "shortage": self.shortage_cost,
                "timely": self.timely_cost,
                "delayed": self.delayed_cost,
                "total": self.total_cost,
            },
            "retailer_net_inventory_last": self.retailer_net_inventory_last,
            "internal_request_backlog_last": self.internal_request_backlog_last,
            "pipeline_to_retailer_last": list(self.pipeline_to_retailer_last),
            "warehouse_on_hand_last": self.warehouse_on_hand_last,
            "pipeline_to_warehouse_last": list(self.pipeline_to_warehouse_last),
            "retailer_net_inventory": self.retailer_net_inventory,
            "internal_request_backlog": self.internal_request_backlog,
            "pipeline_to_retailer": list(self.pipeline_to_retailer),
            "warehouse_on_hand": self.warehouse_on_hand,
            "pipeline_to_warehouse": list(self.pipeline_to_warehouse),
            "retailer_request": self.retailer_request,
            "warehouse_supplier_order": self.warehouse_supplier_order,
            "shipment_to_retailer": self.shipment_to_retailer,
            "demand": self.demand,
        }

    def reset(self, seed: int = None):
        if seed is not None:
            super().reset(seed=seed)
            self._set_seed(seed)
        else:
            super().reset()
        self._initialize()
        self.demand_list = self.demand_generator.sample(self.max_steps)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.retailer_net_inventory_last = self.retailer_net_inventory
        self.internal_request_backlog_last = self.internal_request_backlog
        self.pipeline_to_retailer_last = copy(self.pipeline_to_retailer)
        self.warehouse_on_hand_last = self.warehouse_on_hand
        self.pipeline_to_warehouse_last = copy(self.pipeline_to_warehouse)

        action = int(action)
        if action not in self.action_space:
            raise ValueError(f"Invalid action {action}.")

        self.retailer_request, self.warehouse_supplier_order = self._decode_action(action)
        self.timely_cost = 0.0
        self.pipeline_to_warehouse.append(self.warehouse_supplier_order)
        self.warehouse_on_hand += self.pipeline_to_warehouse.popleft()
        self.internal_request_backlog += self.retailer_request

        truncated = False
        if self.warehouse_on_hand > self.max_warehouse_inventory:
            self.warehouse_on_hand = self.max_warehouse_inventory
            truncated = True
        if self.internal_request_backlog > self.max_internal_backlog:
            self.internal_request_backlog = self.max_internal_backlog
            truncated = True

        self.shipment_to_retailer = min(self.internal_request_backlog, self.warehouse_on_hand)
        self.internal_request_backlog -= self.shipment_to_retailer
        self.warehouse_on_hand -= self.shipment_to_retailer
        self.pipeline_to_retailer.append(self.shipment_to_retailer)

        self.retailer_net_inventory += self.pipeline_to_retailer.popleft()
        truncated = self._clip_retail_inventory() or truncated

        self.demand = int(self.demand_list[self.current_step])
        self.retailer_net_inventory -= self.demand
        truncated = self._clip_retail_inventory() or truncated

        self.holding_cost_retail = self.holding_cost_rate_retail * max(self.retailer_net_inventory, 0)
        self.holding_cost_warehouse = self.holding_cost_rate_warehouse * self.warehouse_on_hand
        self.shortage_cost = self.shortage_cost_rate * max(-self.retailer_net_inventory, 0)
        self.delayed_cost = (
            self.holding_cost_retail + self.holding_cost_warehouse + self.shortage_cost
        )
        self.total_cost = self.timely_cost + self.delayed_cost
        reward = -self.total_cost

        observation = self._get_obs()
        info = self._get_info()
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        return observation, reward, terminated, truncated, info

    def cal_action_distance(self, action_0, action_1):
        q_retail_0, q_warehouse_0 = self._decode_action(int(action_0))
        q_retail_1, q_warehouse_1 = self._decode_action(int(action_1))
        return abs(q_retail_0 - q_retail_1) + abs(q_warehouse_0 - q_warehouse_1)

    def close(self):
        pass

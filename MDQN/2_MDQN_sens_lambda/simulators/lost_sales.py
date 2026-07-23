import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from copy import deepcopy as copy


lost_sale_configs = [
    # task1
    {
        "mean": 5,
        "seed": 0,
        "demand_dist": "poisson",
        "lead_time": 2,
        "ordering_cost_rate": 0.1,
        "shortage_cost_rate": 3.0, # 3
        "holding_cost_rate": 1.0, # 1
        "max_order_quantity": 15,
        "max_inventory": 45,
        "max_steps": 1000
    },
    # task2
    {
        "mean": 5,
        "seed": 0,
        "demand_dist": "poisson",
        "lead_time": 6,
        "ordering_cost_rate": 0.1,
        "shortage_cost_rate": 3.0,
        "holding_cost_rate": 1.0,
        "max_order_quantity": 15,
        "max_inventory": 45,
        "max_steps": 1000
    },
    # task3
    {
        "mean": 5,
        "seed": 0,
        "demand_dist": "geometric",
        "lead_time": 2,
        "ordering_cost_rate": 0.1,
        "shortage_cost_rate": 8.0,
        "holding_cost_rate": 1.0,
        "max_order_quantity": 15,
        "max_inventory": 45,
        "max_steps": 1000
    },
    # task4
    {
        "mean": 5,
        "seed": 0,
        "demand_dist": "geometric",
        "lead_time": 6,
        "ordering_cost_rate": 0.1,
        "shortage_cost_rate": 8.0,
        "holding_cost_rate": 1.0,
        "max_order_quantity": 15,
        "max_inventory": 45,
        "max_steps": 1000
    }
]


class DemandGenerator:
    "Demand generator for lost sales inventory environment."
    def __init__(self, mean:int, seed:int=None, demand_dist:str="poisson"):
        if demand_dist == "poisson":
            self.dist_param = mean
        elif demand_dist == "geometric":
            self.dist_param = 1 / mean
        else:
            raise ValueError(f"Invalid demand distribution {self.demand_dist}.")
        self.demand_dist = demand_dist
        self.reset_seed(seed)

    def sample(self, size:int=1):
        return self.generator(self.dist_param, size)

    def reset_seed(self, seed:int):
        self.rng = np.random.default_rng(seed=seed)
        if self.demand_dist == "poisson":
            self.generator = self.rng.poisson
        else:
            self.generator = self.rng.geometric


class LostSalesInventory(gym.Env):
    """Lost sales inventory environment.
    Args:
        config (dict): Environment configuration.
    """
    def __init__(self, config:dict):
        super().__init__()
        # demand generator
        
        # parameters
        self.lead_time = config["lead_time"]
        self.ordering_cost_rate = config["ordering_cost_rate"]
        self.holding_cost_rate = config["holding_cost_rate"]
        self.shortage_cost_rate = config["shortage_cost_rate"]
        self.max_order_quantity = config["max_order_quantity"]
        self.max_inventory = config["max_inventory"]
        self.max_steps = config["max_steps"]
        self.reward_delay_time = self.lead_time
        # components
        self.demand_generator = DemandGenerator(config["mean"],
                                                 config["seed"],
                                                 config["demand_dist"])
        self.action_space = spaces.Discrete(self.max_order_quantity + 1)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(self.lead_time,), dtype=np.int64)
        # random seed
        self._set_seed(config["seed"])

    def _set_seed(self, seed:int):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.action_space.seed(seed)
        self.demand_generator.reset_seed(seed)

    def _initialize(self):
        # last state
        self.inventory_on_hand_last = 0
        self.inventory_on_order_last = deque([0] * self.lead_time, maxlen=self.lead_time + 1)
        # new state
        self.inventory_on_hand = 0
        self.inventory_on_order = deque([0] * self.lead_time, maxlen=self.lead_time + 1)
        # step
        self.current_step = 0
        # cost
        self.ordering_cost = 0
        self.shortage_cost = 0
        self.holding_cost = 0
        self.timely_cost = 0
        self.delayed_cost = 0
        self.total_cost = 0
        # demand
        self.demand = 0
        # action
        self.action = 0

    def _get_obs(self):
        observation = np.array(self.inventory_on_order)
        observation[0] += self.inventory_on_hand
        return observation

    def _get_info(self):
        return {
            "step": self.current_step,
            "costs": {
                "ordering": self.ordering_cost,
                "shortage": self.shortage_cost,
                "holding": self.holding_cost,
                "timely": self.timely_cost,
                "delayed": self.delayed_cost,
                "total": self.total_cost
            },
            "inventory_on_hand_last": self.inventory_on_hand_last,
            "inventory_on_order_last": self.inventory_on_order_last,            
            "inventory_on_hand": self.inventory_on_hand,
            "inventory_on_order": list(self.inventory_on_order),
            "demand": self.demand,
            "action": self.action
        }

    def reset(self, seed:int=None):
        if seed is not None:
            super().reset(seed=seed)
            self._set_seed(seed)
        else:
            super().reset()
        self._initialize()
        # demand
        self.demand_list = self.demand_generator.sample(self.max_steps)
        # init info and observation
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # save last state
        self.inventory_on_hand_last = self.inventory_on_hand
        self.inventory_on_order_last = copy(self.inventory_on_order)
        # check action
        action = int(action)
        if not action in self.action_space:
            raise ValueError(f"Invalid action {action}.")
        # order
        self.action = action
        self.ordering_cost = self.ordering_cost_rate * action
        self.inventory_on_order.append(action)
        # arrive
        self.inventory_on_hand += self.inventory_on_order.popleft()
        if self.inventory_on_hand > self.max_inventory:
            truncated = True
        else:
            truncated = False
        self.inventory_on_hand = min(self.inventory_on_hand, self.max_inventory)
        # sell
        self.demand = self.demand_list[self.current_step]
        self.sell_quantity = min(self.inventory_on_hand, self.demand)
        self.inventory_on_hand -= self.sell_quantity
        self.shortage_cost = self.shortage_cost_rate * (self.demand - self.sell_quantity)
        # hold
        self.holding_cost = self.holding_cost_rate * self.inventory_on_hand
        # cost and reward
        self.timely_cost = self.ordering_cost
        self.delayed_cost = self.holding_cost + self.shortage_cost
        self.total_cost = self.timely_cost + self.delayed_cost
        reward = -self.total_cost
        # step
        observation = self._get_obs()
        info = self._get_info()
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        return observation, reward, terminated, truncated, info

    def cal_action_distance(self, action_0, action_1):
        return abs(action_0 - action_1)

    def close(self):
        pass


# for test
if __name__ == "__main__":
    env = LostSalesInventory(lost_sale_configs[0])
    state, info = env.reset(0)
    for t in range(5):
        print(info)
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)

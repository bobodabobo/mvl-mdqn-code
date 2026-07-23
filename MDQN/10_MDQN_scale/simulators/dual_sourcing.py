import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from copy import deepcopy as copy


dual_sourcing_configs = [

    # task1
    {
        "seed": 0,
        "lead_time_expediting": 1,
        "lead_time_normal": 2,
        "ordering_cost_rate_normal": 0.1,
        "ordering_cost_rate_expediting": 1.1,
        "holding_cost_rate": 1,
        "shortage_cost_rate": 6,
        "max_order_quantity": 5,
        "max_inventory": 15,
        "max_steps": 1000
    },
    # task2
    {
        "seed": 1,
        "lead_time_expediting":1,
        "lead_time_normal": 2,
        "ordering_cost_rate_normal": 0.1,
        "ordering_cost_rate_expediting": 3.1,
        "holding_cost_rate": 1,
        "shortage_cost_rate": 6,
        "max_order_quantity": 5,
        "max_inventory": 15,
        "max_steps": 1000
    },
    # task3
    {
        "seed": 2,
        "lead_time_expediting":1,
        "lead_time_normal": 4,
        "ordering_cost_rate_normal": 0.1,
        "ordering_cost_rate_expediting": 1.1,
        "holding_cost_rate": 1,
        "shortage_cost_rate": 6,
        "max_order_quantity": 5,
        "max_inventory": 15,
        "max_steps": 1000
    },
    # task4
    {
        "seed": 3,
        "lead_time_expediting":1,
        "lead_time_normal": 4,
        "ordering_cost_rate_normal": 0.1,
        "ordering_cost_rate_expediting": 3.1,
        "holding_cost_rate": 1,
        "shortage_cost_rate": 6,
        "max_order_quantity": 5,
        "max_inventory": 15,
        "max_steps": 1000
    }
]


class DemandGenerator:
    "Demand generator for dual-sourcing environment."
    def __init__(self, seed:int=None):
        self.reset_seed(seed)

    def sample(self, size:int=1):
        return self.rng.integers(0, 5, size=size)

    def reset_seed(self, seed:int):
        self.rng = np.random.default_rng(seed=seed)


class DualSourcingInventory(gym.Env):
    """Dual-sourcing environment.
    Args:
        config (dict): Environment configuration.
    """
    def __init__(self, config:dict):
        super().__init__()
        # parameters
        self.lead_time_expediting = config["lead_time_expediting"]
        self.lead_time_normal = config["lead_time_normal"]
        self.ordering_cost_rate_normal = config["ordering_cost_rate_normal"]
        self.ordering_cost_rate_expediting = config["ordering_cost_rate_expediting"]
        self.holding_cost_rate = config["holding_cost_rate"]
        self.shortage_cost_rate = config["shortage_cost_rate"]
        self.max_order_quantity = config["max_order_quantity"]
        self.max_inventory = config["max_inventory"]
        self.max_steps = config["max_steps"]
        self.reward_delay_time = min(self.lead_time_expediting, self.lead_time_normal)
        # components
        self.demand_generator = DemandGenerator()
        self.action_space = spaces.Discrete((self.max_order_quantity + 1) ** 2)
        self.observation_space = spaces.Box(low=0, high=self.max_inventory + 1, shape=(self.lead_time_normal,), dtype=np.int64)
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
        self.inventory_on_order_last = deque([0] * self.lead_time_normal, maxlen=self.lead_time_normal + 1)
        # new state
        self.inventory_on_hand = 0
        self.inventory_on_order = deque([0] * self.lead_time_normal, maxlen=self.lead_time_normal + 1)
        # step
        self.current_step = 0
        # cost
        self.ordering_cost_normal = 0
        self.ordering_cost_expediting = 0
        self.shortage_cost = 0
        self.holding_cost = 0
        self.timely_cost = 0
        self.delayed_cost = 0
        self.total_cost = 0
        # demand
        self.demand = 0
        # action
        self.action = 0
        # order quantity
        self.order_quantity_normal = 0
        self.order_quantity_expediting = 0

    def _get_obs(self):
        observation = np.array(self.inventory_on_order)
        observation[0] += self.inventory_on_hand
        return observation

    def _get_info(self):
        return {
            "step": self.current_step,
            "costs": {
                "ordering_normal": self.ordering_cost_normal,
                "ordering_expediting": self.ordering_cost_expediting,
                "holding": self.holding_cost,
                "shortage": self.shortage_cost,
                "timely": self.timely_cost,
                "delayed": self.shortage_cost + self.holding_cost,
                "total": self.ordering_cost_normal + self.ordering_cost_expediting + self.holding_cost + self.shortage_cost
            },
            "inventory_on_hand_last": self.inventory_on_hand_last,
            "inventory_on_order_last": self.inventory_on_order_last,            
            "inventory_on_hand": self.inventory_on_hand,
            "inventory_on_order": self.inventory_on_order,
            "demand": self.demand,
            "action": self.action,
            "order_quantity_normal": self.order_quantity_normal,
            "order_quantity_expediting": self.order_quantity_expediting,
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
        self.order_quantity_normal = action // (self.max_order_quantity + 1)
        self.order_quantity_expediting = action % (self.max_order_quantity + 1)
        self.ordering_cost_normal = self.ordering_cost_rate_normal * self.order_quantity_normal
        self.ordering_cost_expediting = self.ordering_cost_rate_expediting * self.order_quantity_expediting
        self.inventory_on_order[self.lead_time_expediting] += self.order_quantity_expediting
        self.inventory_on_order.append(self.order_quantity_normal)
        # arrive
        self.inventory_on_hand += self.inventory_on_order.popleft()
        if self.inventory_on_hand >= self.max_inventory:
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
        # costs and reward
        self.timely_cost = self.ordering_cost_normal + self.ordering_cost_expediting
        self.delayed_cost = self.shortage_cost + self.holding_cost
        self.total_cost = self.timely_cost + self.delayed_cost
        reward = -self.total_cost
        # step
        observation = self._get_obs()
        info = self._get_info()
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        return observation, reward, terminated, truncated, info
    
    def cal_action_distance(self, action_0, action_1):
        order_quantity_normal_0 = action_0 // (self.max_order_quantity + 1)
        order_quantity_expediting_0 = action_0 % (self.max_order_quantity + 1)
        order_quantity_normal_1 = action_1 // (self.max_order_quantity + 1)
        order_quantity_expediting_1 = action_1 % (self.max_order_quantity + 1)
        distance = abs(order_quantity_normal_0 - order_quantity_normal_1)
        distance += abs(order_quantity_expediting_0 - order_quantity_expediting_1)
        return distance

    def close(self):
        pass


# for test
if __name__ == "__main__":
    env = DualSourcingInventory(dual_sourcing_configs[0])
    state, info = env.reset(0)
    for t in range(5):
        print(info)
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from copy import deepcopy as copy


perishable_configs = [
    # task1:
    {
        "mean": 4,
        "seed": 0,
        "lead_time": 1,
        "shelf_life": 2,
        "policy": "fifo",
        "ordering_cost_rate": 3,
        "shortage_cost_rate": 5, # 1
        "wastage_cost_rate": 10, # 10
        "holding_cost_rate": 0,
        "max_order_quantity": 12,
        "max_steps": 1000
    },
    # task2:
    {
        "mean": 4,
        "seed": 0,
        "lead_time": 2,
        "shelf_life": 2,
        "policy": "fifo",
        "ordering_cost_rate": 3,
        "shortage_cost_rate": 5,
        "wastage_cost_rate": 10,
        "holding_cost_rate": 0,
        "max_order_quantity": 12,
        "max_steps": 1000
    },
    # task3:
    {
        "mean": 4,
        "seed": 0,
        "lead_time": 1,
        "shelf_life": 2,
        "policy": "lifo",
        "ordering_cost_rate": 3,
        "shortage_cost_rate": 5,
        "wastage_cost_rate": 10,
        "holding_cost_rate": 0,
        "max_order_quantity": 12,
        "max_steps": 1000,
    },
    # task4:
    {
        "mean": 4,
        "seed": 0,
        "lead_time": 2,
        "shelf_life": 2,
        "policy": "lifo",
        "ordering_cost_rate": 3,
        "shortage_cost_rate": 5,
        "wastage_cost_rate": 10,
        "holding_cost_rate": 0,
        "max_order_quantity": 12,
        "max_steps": 1000
    }
]


class DemandGenerator:
    "Demand generator for perishable inventory environment."
    def __init__(self, mean:int, seed:int=None):
        self.reset_seed(seed)
        self.dist_param = mean

    def sample(self, size:int=1):
        return self.rng.poisson(self.dist_param, size)

    def reset_seed(self, seed:int):
        self.rng = np.random.default_rng(seed=seed)
        
        

class PerishableInventory(gym.Env):
    """
    Perishable inventory environment with FIFO/LIFO selling policies.

    Args:
        config (dict): Environment configuration.
    """
    def __init__(self, config:dict):
        super().__init__()
        # parameters
        self.lead_time = config["lead_time"]
        self.shelf_life = config["shelf_life"]
        self.policy = config["policy"].lower()
        if self.policy not in ['fifo', 'lifo']:
            raise ValueError(f"Invalid policy '{self.policy}'. Must be 'fifo' or 'lifo'.")
        self.max_order_quantity = config["max_order_quantity"]
        self.max_steps = config["max_steps"]
        self.max_inventory = self.max_order_quantity * (self.shelf_life + self.lead_time - 1) # for BSP-low-EW
        self.reward_delay_time = self.lead_time  # for MDQN
        # cost rates
        self.ordering_cost_rate = config["ordering_cost_rate"]
        self.shortage_cost_rate = config["shortage_cost_rate"]
        self.wastage_cost_rate = config["wastage_cost_rate"]
        self.holding_cost_rate = config["holding_cost_rate"]
        # components
        self.demand_generator = DemandGenerator(config["mean"], config["seed"])
        self.action_space = spaces.Discrete(self.max_order_quantity + 1)
        obs_space_dims = self.shelf_life + self.lead_time - 1
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(obs_space_dims,), dtype=np.int64)
        # random seed
        self._set_seed(config["seed"])

    def _set_seed(self, seed:int):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.action_space.seed(seed)
        self.demand_generator.reset_seed(seed)

    def _initialize(self):
        # last state
        self.inventory_on_hand_last = deque([0] * (self.shelf_life - 1), maxlen=self.shelf_life)
        self.inventory_on_order_last = deque([0] * self.lead_time, maxlen=self.lead_time + 1)
        # new state
        self.inventory_on_hand = deque([0] * (self.shelf_life - 1), maxlen=self.shelf_life)
        self.inventory_on_order = deque([0] * self.lead_time, maxlen=self.lead_time + 1)
        # step
        self.current_step = 0
        # cost
        self.ordering_cost = 0
        self.shortage_cost = 0
        self.wastage_cost = 0
        self.holding_cost = 0
        self.timely_cost = 0
        self.delayed_cost = 0
        self.total_cost = 0
        # demand
        self.demand = 0
        # action
        self.action = 0

    def _get_obs(self):
        observation = list(self.inventory_on_hand) + list(self.inventory_on_order)
        return np.array(observation)

    def _get_info(self):
        return {
            "step": self.current_step,
            "costs": {
                "ordering": self.ordering_cost,
                "shortage": self.shortage_cost,
                "wastage": self.wastage_cost,
                "holding": self.holding_cost,
                "timely": self.timely_cost,
                "delayed": self.delayed_cost,
                "total": self.total_cost
            },
            "inventory_on_hand_last": self.inventory_on_hand_last,
            "inventory_on_order_last": self.inventory_on_order_last,            
            "inventory_on_hand": list(self.inventory_on_hand),
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
        # demand list
        self.demand_list = self.demand_generator.sample(self.max_steps)
        # init obs
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # Save last state for info
        self.inventory_on_hand_last = copy(self.inventory_on_hand)
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
        arrived_quantity = self.inventory_on_order.popleft()
        self.inventory_on_hand.append(arrived_quantity)
        # sell (FIFO or LIFO)
        self.demand = self.demand_list[self.current_step]
        if self.policy == 'fifo':
            item_indices = range(self.shelf_life)
        else:
            item_indices = range(self.shelf_life - 1, -1, -1)
        remaining_demand = self.demand
        for i in item_indices:
            if remaining_demand <= 0:
                break
            sell_from_this_bin = min(remaining_demand, self.inventory_on_hand[i])
            self.inventory_on_hand[i] -= sell_from_this_bin
            remaining_demand -= sell_from_this_bin
        self.shortage_cost = self.shortage_cost_rate * remaining_demand
        # aging and Perishing
        self.perished_quantity = self.inventory_on_hand.popleft()
        self.wastage_cost = self.wastage_cost_rate * self.perished_quantity
        # hold
        total_on_hand = np.sum(self.inventory_on_hand)
        self.holding_cost = self.holding_cost_rate * total_on_hand
        # cost and reward
        self.timely_cost = self.ordering_cost
        self.delayed_cost = self.shortage_cost + self.wastage_cost + self.holding_cost
        self.total_cost = self.timely_cost + self.delayed_cost
        reward = -self.total_cost
        # step
        observation = self._get_obs()
        info = self._get_info()
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        return observation, reward, terminated, truncated, info
    
    def cal_action_distance(self, action_0, action_1):
        return abs(action_0 - action_1)

    def close(self):
        pass

# for test
if __name__ == '__main__':
    print("================== Running FIFO Example ==================")
    env = PerishableInventory(config=perishable_configs[0])
    obs, info = env.reset(0)
    for i in range(5):
        print(info)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
    
    print("================== Running LIFO Example ==================")
    env = PerishableInventory(config=perishable_configs[3])
    obs, info = env.reset(0)
    for i in range(5):
        print(info)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

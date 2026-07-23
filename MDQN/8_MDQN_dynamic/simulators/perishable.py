import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import deque
from copy import deepcopy as copy

from .dynamic_demand import DynamicDemandGenerator


perishable_configs = [
    # task1
    {
        "seed": 0,
        "lead_time": 1,
        "shelf_life": 2,
        "policy": "fifo",
        "ordering_cost_rate": 3,
        "shortage_cost_rate": 5,
        "wastage_cost_rate": 10,
        "holding_cost_rate": 0,
        "max_order_quantity": 12,
        "max_steps": 1000,
    },
    # task2
    {
        "seed": 0,
        "lead_time": 2,
        "shelf_life": 2,
        "policy": "fifo",
        "ordering_cost_rate": 3,
        "shortage_cost_rate": 5,
        "wastage_cost_rate": 10,
        "holding_cost_rate": 0,
        "max_order_quantity": 12,
        "max_steps": 1000,
    },
    # task3
    {
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
    # task4
    {
        "seed": 0,
        "lead_time": 2,
        "shelf_life": 2,
        "policy": "lifo",
        "ordering_cost_rate": 3,
        "shortage_cost_rate": 5,
        "wastage_cost_rate": 10,
        "holding_cost_rate": 0,
        "max_order_quantity": 12,
        "max_steps": 1000,
    },
]


class PerishableInventory(gym.Env):
    """Perishable inventory environment with dynamic periodic demand."""

    def __init__(self, config: dict):
        super().__init__()
        self.lead_time = config["lead_time"]
        self.shelf_life = config["shelf_life"]
        self.policy = config["policy"].lower()
        if self.policy not in ["fifo", "lifo"]:
            raise ValueError(f"Invalid policy '{self.policy}'. Must be 'fifo' or 'lifo'.")
        self.max_order_quantity = config["max_order_quantity"]
        self.max_steps = config["max_steps"]
        self.max_inventory = self.max_order_quantity * (self.shelf_life + self.lead_time - 1)
        self.reward_delay_time = self.lead_time
        self.ordering_cost_rate = config["ordering_cost_rate"]
        self.shortage_cost_rate = config["shortage_cost_rate"]
        self.wastage_cost_rate = config["wastage_cost_rate"]
        self.holding_cost_rate = config["holding_cost_rate"]
        self.inventory_obs_dim = self.shelf_life + self.lead_time - 1
        self.effective_lead_time = self.lead_time
        self.demand_generator = DynamicDemandGenerator(self.effective_lead_time, config["seed"])
        self.history_len = self.demand_generator.history_len
        self.period = self.demand_generator.period
        self.action_space = spaces.Discrete(self.max_order_quantity + 1)
        obs_dim = self.inventory_obs_dim + self.history_len
        self.observation_space = spaces.Box(low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self._set_seed(config["seed"])

    def _set_seed(self, seed: int):
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.action_space.seed(seed)
        self.demand_generator.reset_seed(seed)

    def _initialize(self):
        self.inventory_on_hand_last = deque([0.0] * (self.shelf_life - 1), maxlen=self.shelf_life)
        self.inventory_on_order_last = deque([0.0] * self.lead_time, maxlen=self.lead_time + 1)
        self.inventory_on_hand = deque([0.0] * (self.shelf_life - 1), maxlen=self.shelf_life)
        self.inventory_on_order = deque([0.0] * self.lead_time, maxlen=self.lead_time + 1)
        self.demand_history = deque([0.0] * self.history_len, maxlen=self.history_len)
        self.current_step = 0
        self.ordering_cost = 0.0
        self.shortage_cost = 0.0
        self.wastage_cost = 0.0
        self.holding_cost = 0.0
        self.timely_cost = 0.0
        self.delayed_cost = 0.0
        self.total_cost = 0.0
        self.demand = 0.0
        self.demand_center = 0.0
        self.phase_offset = 0
        self.action = 0

    def _get_inventory_obs(self):
        return np.array(list(self.inventory_on_hand) + list(self.inventory_on_order), dtype=np.float32)

    def _get_obs(self):
        inventory_obs = self._get_inventory_obs()
        demand_history = np.array(self.demand_history, dtype=np.float32)
        return np.concatenate((inventory_obs, demand_history)).astype(np.float32, copy=False)

    def extract_inventory_obs(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        return obs[: self.inventory_obs_dim].copy()

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
                "total": self.total_cost,
            },
            "inventory_on_hand_last": list(self.inventory_on_hand_last),
            "inventory_on_order_last": list(self.inventory_on_order_last),
            "inventory_on_hand": list(self.inventory_on_hand),
            "inventory_on_order": list(self.inventory_on_order),
            "demand": self.demand,
            "demand_center": self.demand_center,
            "phase_offset": self.phase_offset,
            "action": self.action,
        }

    def reset(self, seed: int | None = None):
        if seed is not None:
            super().reset(seed=seed)
            self._set_seed(seed)
        else:
            super().reset()
        self._initialize()
        self.phase_offset = int(self.rng.integers(0, self.period))
        demand_episode = self.demand_generator.sample_episode(self.max_steps, self.phase_offset)
        self.demand_history = deque(demand_episode["history"].tolist(), maxlen=self.history_len)
        self.demand_list = demand_episode["demand_list"]
        self.demand_center_list = demand_episode["demand_centers"]
        self.demand_center = float(self.demand_center_list[0])
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.inventory_on_hand_last = copy(self.inventory_on_hand)
        self.inventory_on_order_last = copy(self.inventory_on_order)
        action = int(action)
        if action not in self.action_space:
            raise ValueError(f"Invalid action {action}.")
        self.action = action
        self.ordering_cost = self.ordering_cost_rate * action
        self.inventory_on_order.append(float(action))
        arrived_quantity = self.inventory_on_order.popleft()
        self.inventory_on_hand.append(arrived_quantity)
        self.demand = float(self.demand_list[self.current_step])
        self.demand_center = float(self.demand_center_list[self.current_step])
        item_indices = range(self.shelf_life) if self.policy == "fifo" else range(self.shelf_life - 1, -1, -1)
        remaining_demand = self.demand
        for idx in item_indices:
            if remaining_demand <= 0:
                break
            sell_from_this_bin = min(remaining_demand, self.inventory_on_hand[idx])
            self.inventory_on_hand[idx] -= sell_from_this_bin
            remaining_demand -= sell_from_this_bin
        self.shortage_cost = self.shortage_cost_rate * remaining_demand
        self.perished_quantity = self.inventory_on_hand.popleft()
        self.wastage_cost = self.wastage_cost_rate * self.perished_quantity
        total_on_hand = np.sum(self.inventory_on_hand)
        self.holding_cost = self.holding_cost_rate * total_on_hand
        self.timely_cost = self.ordering_cost
        self.delayed_cost = self.shortage_cost + self.wastage_cost + self.holding_cost
        self.total_cost = self.timely_cost + self.delayed_cost
        reward = -self.total_cost
        self.demand_history.append(self.demand)
        observation = self._get_obs()
        info = self._get_info()
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        return observation, reward, terminated, truncated, info

    def cal_action_distance(self, action_0, action_1):
        return abs(int(action_0) - int(action_1))

    def close(self):
        pass


if __name__ == "__main__":
    print("================== Running FIFO Example ==================")
    env = PerishableInventory(config=perishable_configs[0])
    obs, info = env.reset(0)
    for _ in range(5):
        print(info)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

    print("================== Running LIFO Example ==================")
    env = PerishableInventory(config=perishable_configs[3])
    obs, info = env.reset(0)
    for _ in range(5):
        print(info)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
